"""Live domain metrics and progression-aware guardrail.

Promotes Orchid's progression-specific metrics from offline-eval-only to
first-class online signals:

* :class:`RollingProgressionMonitor` -- maintains per-policy rolling windows
  of (user, item, correct, pre_competence, post_competence, difficulty)
  tuples and exposes :func:`progression_gain`, :func:`category_coverage`,
  :func:`sequence_adherence`, :func:`stretch_fit`, and rolling acceptance
  rate. Values are mirrored to Prometheus gauges defined in
  :mod:`orchid_ranker.observability`.

* :class:`ProgressionGuardrail` -- domain-specific circuit breaker that halts
  the adaptive policy when rolling progression gain or category coverage drops
  below a floor, independently of SafeSwitchDR's generic acceptance gate. Can
  be composed *with* SafeSwitchDR via :meth:`combine_with` so either guardrail
  can halt rollout.

Why this matters: generic recommender stacks gate rollout on CTR or revenue;
progression domains need guardrails tied to *competence*, not engagement. A
ranker that keeps users clicking but stops progressing should be halted.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Set,
)

from orchid_ranker.evaluation import (
    category_coverage,
    progression_gain,
    sequence_adherence,
    stretch_fit,
)

# Prometheus is already a hard dep (observability.py imports unconditionally).
from orchid_ranker.observability import (
    DIFFICULTY_APPROPRIATENESS,
    PROFICIENCY_COVERAGE,
    PROGRESSION_GAIN,
    PROGRESSION_GUARDRAIL_HALTED,
    PROGRESSION_GUARDRAIL_TRIGGERS,
    ROLLING_ACCEPT_RATE,
    SEQUENCE_ADHERENCE,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ProgressionSnapshot",
    "RollingProgressionMonitor",
    "GuardrailConfig",
    "ProgressionGuardrail",
]


# ---------------------------------------------------------------------------
# Rolling monitor
# ---------------------------------------------------------------------------
@dataclass
class _Event:
    user_id: int
    item_id: int
    correct: bool
    pre_competence: float
    post_competence: float
    category: str
    difficulty: Optional[float]
    timestamp: float


@dataclass
class ProgressionSnapshot:
    """Current values for the four progression metrics + acceptance rate.

    All fields are in [0, 1]. ``sample_size`` is the number of events actually
    used to compute the snapshot (capped by the monitor's window size).
    """
    progression_gain: float
    category_coverage: float
    sequence_adherence: float
    stretch_fit: float
    accept_rate: float
    sample_size: int
    policy: str = "adaptive"

    # Backward-compatible aliases (deprecated)
    @property
    def proficiency_coverage(self) -> float:
        """Deprecated alias for ``category_coverage``."""
        import warnings
        warnings.warn(
            "Attribute 'proficiency_coverage' is deprecated, use 'category_coverage' instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.category_coverage

    @property
    def difficulty_appropriateness(self) -> float:
        """Deprecated alias for ``stretch_fit``."""
        import warnings
        warnings.warn(
            "Attribute 'difficulty_appropriateness' is deprecated, use 'stretch_fit' instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.stretch_fit

    def as_dict(self) -> Dict[str, float | int | str]:
        return {
            "progression_gain": self.progression_gain,
            "category_coverage": self.category_coverage,
            "sequence_adherence": self.sequence_adherence,
            "stretch_fit": self.stretch_fit,
            "accept_rate": self.accept_rate,
            "sample_size": self.sample_size,
            "policy": self.policy,
        }


class RollingProgressionMonitor:
    """Rolling-window monitor for progression metrics.

    Call :meth:`record` once per served (user, item) pair after the outcome
    is observed. The monitor maintains a bounded deque of recent events and
    recomputes metrics on demand (cheap: O(window_size)).

    Parameters
    ----------
    policy : str
        Label attached to all Prometheus gauges (e.g. ``"adaptive"``,
        ``"baseline"``). Lets you compare the adaptive loop against a control.
    window_size : int
        Max events retained. Older events are evicted FIFO. 200-2000 is a
        good range; larger windows smooth noise at the cost of slower
        reaction to drift.
    prerequisite_graph : dict, optional
        Item -> set-of-prerequisite-items. When provided, sequence_adherence
        is reported; otherwise it defaults to 1.0 (no DAG = all adherent).
    total_categories : set, optional
        Universe of categories for category_coverage. When absent, coverage
        is computed against the set of categories seen so far (i.e. grows
        with the window) -- less informative but safer for cold-start.
    total_skills : set, optional
        Deprecated alias for ``total_categories``.
    success_threshold : float
        Post-competence >= threshold counts as completed for coverage (default 0.7).
    mastery_threshold : float
        Deprecated alias for ``success_threshold``.
    stretch_width : float
        Width of the stretch zone for stretch_fit (default 0.25).
    zpd_width : float
        Deprecated alias for ``stretch_width``.
    emit_prometheus : bool
        If False, metrics are still computed but not pushed to gauges. Useful
        in tests or when multiple monitors would race on the same labels.
    """

    def __init__(
        self,
        *,
        policy: str = "adaptive",
        window_size: int = 500,
        prerequisite_graph: Optional[Mapping[int, Iterable[int]]] = None,
        total_categories: Optional[Iterable[Any]] = None,
        success_threshold: Optional[float] = None,
        stretch_width: Optional[float] = None,
        emit_prometheus: bool = True,
        # Deprecated aliases
        total_skills: Optional[Iterable[Any]] = None,
        mastery_threshold: Optional[float] = None,
        zpd_width: Optional[float] = None,
    ) -> None:
        import warnings as _w

        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        # Resolve deprecated params
        if total_skills is not None:
            _w.warn(
                "Parameter 'total_skills' is deprecated, use 'total_categories' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if total_categories is None:
                total_categories = total_skills
        if mastery_threshold is not None:
            _w.warn(
                "Parameter 'mastery_threshold' is deprecated, use 'success_threshold' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if success_threshold is None:
                success_threshold = mastery_threshold
        if zpd_width is not None:
            _w.warn(
                "Parameter 'zpd_width' is deprecated, use 'stretch_width' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if stretch_width is None:
                stretch_width = zpd_width
        # Apply defaults
        if success_threshold is None:
            success_threshold = 0.7
        if stretch_width is None:
            stretch_width = 0.25

        self.policy = str(policy)
        self.window_size = int(window_size)
        self.success_threshold = float(success_threshold)
        self.stretch_width = float(stretch_width)
        self.emit_prometheus = bool(emit_prometheus)
        self._prereq: Dict[int, Set[int]] = (
            {int(k): set(int(x) for x in v) for k, v in prerequisite_graph.items()}
            if prerequisite_graph is not None
            else {}
        )
        self._total_categories: Optional[Set[Any]] = (
            set(total_categories) if total_categories is not None else None
        )
        self._events: Deque[_Event] = deque(maxlen=self.window_size)
        # latest succeeded items / categories per user, for sequence_adherence
        self._user_succeeded_items: Dict[int, Set[int]] = defaultdict(set)
        self._lock = threading.RLock()

    # Backward-compatible property aliases
    @property
    def mastery_threshold(self) -> float:
        """Deprecated alias for ``success_threshold``."""
        return self.success_threshold

    @property
    def zpd_width(self) -> float:
        """Deprecated alias for ``stretch_width``."""
        return self.stretch_width

    # ------- recording -------
    def record(
        self,
        *,
        user_id: int,
        item_id: int,
        correct: bool | int | float,
        pre_competence: Optional[float] = None,
        post_competence: Optional[float] = None,
        category: Optional[str] = None,
        difficulty: Optional[float] = None,
        user_ability: Optional[float] = None,  # noqa: ARG002 -- reserved for future stretch-fit extension
        timestamp: Optional[float] = None,
        # Deprecated aliases
        pre_mastery: Optional[float] = None,
        post_mastery: Optional[float] = None,
        skill: Optional[str] = None,
    ) -> None:
        """Log one outcome. Non-blocking; metrics are updated synchronously."""
        import warnings as _w

        # Resolve deprecated params
        if pre_mastery is not None:
            _w.warn(
                "Parameter 'pre_mastery' is deprecated, use 'pre_competence' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if pre_competence is None:
                pre_competence = pre_mastery
        if post_mastery is not None:
            _w.warn(
                "Parameter 'post_mastery' is deprecated, use 'post_competence' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if post_competence is None:
                post_competence = post_mastery
        if skill is not None:
            _w.warn(
                "Parameter 'skill' is deprecated, use 'category' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if category is None:
                category = skill
        # Defaults
        if pre_competence is None:
            pre_competence = 0.0
        if post_competence is None:
            post_competence = 0.0
        if category is None:
            category = "__default__"

        ev = _Event(
            user_id=int(user_id),
            item_id=int(item_id),
            correct=bool(correct),
            pre_competence=float(pre_competence),
            post_competence=float(post_competence),
            category=str(category),
            difficulty=(None if difficulty is None else float(difficulty)),
            timestamp=float(timestamp if timestamp is not None else time.time()),
        )
        with self._lock:
            self._events.append(ev)
            # Track completed items per user for sequence_adherence. An item
            # counts as "completed" after a correct outcome OR after post
            # competence crosses the threshold.
            if ev.correct or ev.post_competence >= self.success_threshold:
                self._user_succeeded_items[ev.user_id].add(ev.item_id)
        if self.emit_prometheus:
            self._emit(self.snapshot())

    # ------- computation -------
    def snapshot(self) -> ProgressionSnapshot:
        """Compute the current metric values. O(window_size)."""
        with self._lock:
            events = list(self._events)
            user_succeeded = {u: set(s) for u, s in self._user_succeeded_items.items()}

        n = len(events)
        if n == 0:
            return ProgressionSnapshot(0.0, 0.0, 1.0, 1.0, 0.0, 0, self.policy)

        # progression_gain: average per-event (post - pre) / (1 - pre)
        gains = [
            progression_gain(ev.pre_competence, ev.post_competence) for ev in events
        ]
        avg_gain = float(sum(gains) / len(gains))

        # category_coverage: completed categories / total categories
        succeeded_categories: Set[Any] = {
            ev.category for ev in events
            if ev.post_competence >= self.success_threshold
        }
        if self._total_categories is not None and len(self._total_categories) > 0:
            coverage = category_coverage(
                achieved=succeeded_categories, total=self._total_categories,
            )
        else:
            seen = {ev.category for ev in events}
            coverage = category_coverage(achieved=succeeded_categories, total=seen)

        # sequence_adherence: per-user, average across users in window
        if self._prereq:
            per_user_rates = []
            seen_users = {ev.user_id for ev in events}
            for u in seen_users:
                recs = [ev.item_id for ev in events if ev.user_id == u]
                # Each user's "completed" set is taken from *earlier* in the window:
                # we approximate by using the set of items they got right
                # before their most recent interaction. Cheap and monotone.
                rate = sequence_adherence(
                    recommended_items=recs,
                    prerequisite_graph=self._prereq,
                    succeeded=user_succeeded.get(u, set()),
                )
                per_user_rates.append(rate)
            adherence = float(sum(per_user_rates) / max(1, len(per_user_rates)))
        else:
            adherence = 1.0

        # stretch_fit: fraction across events with difficulty
        diffs = [ev.difficulty for ev in events if ev.difficulty is not None]
        if diffs:
            # Use average user competence level across events as a rough anchor.
            competence = float(sum(ev.pre_competence for ev in events) / n)
            fit = stretch_fit(
                recommended_difficulties=diffs,
                user_competence=competence,
                stretch_width=self.stretch_width,
            )
        else:
            fit = 1.0

        accept_rate = float(sum(1 for ev in events if ev.correct) / n)

        return ProgressionSnapshot(
            progression_gain=avg_gain,
            category_coverage=coverage,
            sequence_adherence=adherence,
            stretch_fit=fit,
            accept_rate=accept_rate,
            sample_size=n,
            policy=self.policy,
        )

    # ------- Prometheus fan-out -------
    def _emit(self, snap: ProgressionSnapshot) -> None:
        try:
            PROGRESSION_GAIN.labels(policy=self.policy).set(snap.progression_gain)
            PROFICIENCY_COVERAGE.labels(policy=self.policy).set(snap.category_coverage)
            SEQUENCE_ADHERENCE.labels(policy=self.policy).set(snap.sequence_adherence)
            DIFFICULTY_APPROPRIATENESS.labels(policy=self.policy).set(snap.stretch_fit)
            ROLLING_ACCEPT_RATE.labels(policy=self.policy).set(snap.accept_rate)
        except Exception:  # noqa: BLE001
            # Prometheus errors must never crash the serving path.
            logger.exception("failed to emit progression metrics")

    def reset(self) -> None:
        """Clear the window and per-user completed-item tracking."""
        with self._lock:
            self._events.clear()
            self._user_succeeded_items.clear()


# ---------------------------------------------------------------------------
# Progression guardrail
# ---------------------------------------------------------------------------
@dataclass
class GuardrailConfig:
    """Thresholds for :class:`ProgressionGuardrail`.

    A guardrail trip requires *both*:
    (a) at least ``warmup_samples`` events in the rolling window, and
    (b) at least one of the configured thresholds violated.

    That means a fresh rollout is never halted on sparse data.

    Attributes
    ----------
    min_progression_gain : float
        Rolling progression_gain below this halts the policy. Set to a
        negative number (e.g. -1.0) to disable this check.
    min_accept_rate : float
        Rolling acceptance rate below this halts. Disable with 0.0.
    min_sequence_adherence : float
        Rolling sequence_adherence below this halts. 1.0 would only trip on
        *any* prerequisite violation; 0.9 is a reasonable default for systems
        that permit occasional out-of-order exploration.
    min_coverage : float
        Rolling category_coverage below this halts. Disable with 0.0.
    warmup_samples : int
        Minimum window size before any halt can fire.
    consecutive_violations : int
        Number of consecutive snapshots in violation before the halt fires.
        Prevents single-snapshot blips from tripping the guardrail.
    """
    min_progression_gain: float = 0.0
    min_accept_rate: float = 0.3
    min_sequence_adherence: float = 0.8
    min_coverage: float = 0.0
    warmup_samples: int = 50
    consecutive_violations: int = 3


class ProgressionGuardrail:
    """Domain-specific halt signal for the adaptive policy.

    Unlike :class:`SafeSwitchDR`, which gates on generic acceptance rate,
    this guardrail watches the four progression metrics Orchid actually
    cares about. It is a *halt signal*, not a traffic splitter: callers ask
    :meth:`should_allow_adaptive` before serving and record outcomes via the
    monitor. The guardrail does not sample the policy — compose with
    SafeSwitchDR for probabilistic rollout gating.

    Thread-safe; all state transitions go through one lock.

    Parameters
    ----------
    monitor : RollingProgressionMonitor
        Source of truth for the gated metrics.
    cfg : GuardrailConfig
        Thresholds.
    """

    def __init__(
        self,
        monitor: RollingProgressionMonitor,
        cfg: Optional[GuardrailConfig] = None,
    ) -> None:
        self.monitor = monitor
        self.cfg = cfg or GuardrailConfig()
        self._halted = False
        self._violation_streak = 0
        self._halt_reason: Optional[str] = None
        self._lock = threading.RLock()
        self._update_gauge()

    # ------- evaluation -------
    def _check(self, snap: ProgressionSnapshot) -> Optional[str]:
        """Return a reason string if any threshold is violated, else None."""
        if snap.sample_size < self.cfg.warmup_samples:
            return None
        if snap.progression_gain < self.cfg.min_progression_gain:
            return "progression_gain_below_floor"
        if snap.accept_rate < self.cfg.min_accept_rate:
            return "accept_rate_below_floor"
        if snap.sequence_adherence < self.cfg.min_sequence_adherence:
            return "sequence_adherence_violated"
        if snap.category_coverage < self.cfg.min_coverage:
            return "coverage_below_floor"
        return None

    def evaluate(self) -> bool:
        """Recompute from the monitor, update state, return ``should_allow_adaptive``."""
        snap = self.monitor.snapshot()
        with self._lock:
            if self._halted:
                return False
            reason = self._check(snap)
            if reason is not None:
                self._violation_streak += 1
                if self._violation_streak >= self.cfg.consecutive_violations:
                    self._halted = True
                    self._halt_reason = reason
                    logger.warning(
                        "ProgressionGuardrail halted: reason=%s snap=%s",
                        reason, snap.as_dict(),
                    )
                    try:
                        PROGRESSION_GUARDRAIL_TRIGGERS.labels(reason=reason).inc()
                    except Exception:  # noqa: BLE001
                        pass
            else:
                self._violation_streak = 0
            self._update_gauge()
            return not self._halted

    def should_allow_adaptive(self) -> bool:
        """Fast path: return current halt state without recomputing."""
        with self._lock:
            return not self._halted

    @property
    def is_halted(self) -> bool:
        with self._lock:
            return self._halted

    @property
    def halt_reason(self) -> Optional[str]:
        with self._lock:
            return self._halt_reason

    def restart(self, *, reset_window: bool = False) -> None:
        """Explicit restart after a halt. Mirrors :class:`SafeSwitchDR.restart`."""
        with self._lock:
            self._halted = False
            self._violation_streak = 0
            self._halt_reason = None
            if reset_window:
                self.monitor.reset()
            self._update_gauge()

    def _update_gauge(self) -> None:
        try:
            PROGRESSION_GUARDRAIL_HALTED.set(1.0 if self._halted else 0.0)
        except Exception:  # noqa: BLE001
            pass

    # ------- composition with SafeSwitchDR -------
    def combine_with(
        self, other_allow: Callable[[], bool]
    ) -> Callable[[], bool]:
        """Produce a combined gate: ``True`` iff *both* this and ``other_allow`` say so.

        Typical use::

            safeswitch = SafeSwitchDR(cfg)
            guardrail  = ProgressionGuardrail(monitor)
            gate = guardrail.combine_with(lambda: safeswitch.decide()[0])
            if gate():
                # serve adaptive
        """

        def combined() -> bool:
            if not self.should_allow_adaptive():
                return False
            return bool(other_allow())

        return combined
