"""Streaming adaptive ranker.

This module closes the gap between Orchid's "adaptive" positioning and its
runtime reality. A frozen :class:`TwoTowerRecommender` is wrapped with:

* :class:`BKTStateProvider` -- fresh user state (competence, fatigue,
  trust, engagement) assembled from live BKT (outcome tracing) posteriors
  and recent-interaction telemetry. Flows into the tower's FiLM gate on
  every rank call.
* :class:`OnlineUserAdapter` -- per-user residual embedding updated via SGD on
  each observed interaction. Lets a user's preferences drift without
  retraining the base tower.
* :class:`StreamingAdaptiveRanker` -- facade exposing ``observe`` and ``rank``
  with built-in adaptation-latency instrumentation so operators can verify the
  loop is actually adaptive in production.

The module is deliberately self-contained: it does not monkey-patch the base
tower and does not require changes to :mod:`orchid_ranker.agents.two_tower`.

Typical usage::

    from orchid_ranker.agents.two_tower import TwoTowerRecommender
    from orchid_ranker.streaming import StreamingAdaptiveRanker

    tower = TwoTowerRecommender(...).eval()          # frozen, pre-trained
    ranker = StreamingAdaptiveRanker(tower, user_feats, item_feats)

    # interaction arrives
    ranker.observe(user_id=7, item_id=42, correct=True)

    # next recommendation uses the *just-updated* state + residual
    top = ranker.rank(user_id=7, candidate_item_ids=[1, 2, 3, 42, 99], top_k=3)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from orchid_ranker.knowledge_tracing import BayesianKnowledgeTracing

logger = logging.getLogger(__name__)

__all__ = [
    "BKTStateProvider",
    "OnlineUserAdapter",
    "StreamingAdaptiveRanker",
    "AdaptationStats",
]


# ---------------------------------------------------------------------------
# BKT state provider
# ---------------------------------------------------------------------------
@dataclass
class _UserTelemetry:
    """Per-user rolling stats used to derive [fatigue, trust, engagement].

    Kept separate from the BKT tracker so non-competence state can evolve even
    for users whose BKT has not been initialised yet.
    """
    events: Deque[Tuple[float, bool]] = field(default_factory=lambda: deque(maxlen=64))
    last_seen: Optional[float] = None

    def record(self, correct: bool, ts: float) -> None:
        self.events.append((float(ts), bool(correct)))
        self.last_seen = float(ts)

    def trust(self) -> float:
        """Proxy for trust: recent correctness rate in [0, 1]."""
        if not self.events:
            return 0.5
        return float(sum(c for _, c in self.events) / len(self.events))

    def engagement(self, now: float, window: float = 300.0) -> float:
        """Proxy for engagement: interactions-per-minute in the last ``window`` seconds, squashed."""
        if not self.events:
            return 0.0
        recent = [t for t, _ in self.events if (now - t) <= window]
        if not recent:
            return 0.0
        rate_per_min = 60.0 * len(recent) / max(window, 1e-6)
        # squash with tanh so engagement ∈ [0, 1)
        return float(math.tanh(rate_per_min / 4.0))

    def fatigue(self, now: float, window: float = 900.0) -> float:
        """Proxy for fatigue: monotonically increases with sustained activity.

        Computed as a saturating function of recent event count in ``window``
        seconds. Resets toward zero after a gap of inactivity.
        """
        if not self.events:
            return 0.0
        recent = [t for t, _ in self.events if (now - t) <= window]
        # saturate: 0 at 0 events, ~0.9 at 30+ events in window
        return float(1.0 - math.exp(-len(recent) / 15.0))


class BKTStateProvider:
    """Maintains per-user BKT trackers and derives the user state vector.

    The tower's state_dim is 4 by default: ``[competence, fatigue, trust, engagement]``.
    Values are in [0, 1]. ``competence`` is the average BKT (outcome tracing)
    posterior across tracked categories (or a single default-category posterior).
    The remaining three components come from lightweight telemetry over recent
    interactions.

    Thread-safe: a single RLock guards trackers and telemetry. Designed for
    low-QPS serving; for high-QPS deployments, shard by user_id modulo N.

    Parameters
    ----------
    state_dim : int
        Must match the tower's ``state_dim``. Only 4 is currently supported.
    bkt_kwargs : dict, optional
        Forwarded to :class:`BayesianKnowledgeTracing` when instantiating a new
        per-user tracker.
    default_category : str
        Key used when an interaction does not specify a category.
    default_skill : str
        Deprecated alias for ``default_category``.
    """

    def __init__(
        self,
        state_dim: int = 4,
        bkt_kwargs: Optional[dict] = None,
        default_category: Optional[str] = None,
        *,
        # Deprecated alias
        default_skill: Optional[str] = None,
    ) -> None:
        import warnings as _w

        if state_dim != 4:
            raise ValueError(
                "BKTStateProvider currently only supports state_dim=4 "
                "([competence, fatigue, trust, engagement])."
            )
        if default_skill is not None:
            _w.warn(
                "Parameter 'default_skill' is deprecated, use 'default_category' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if default_category is None:
                default_category = default_skill
        if default_category is None:
            default_category = "__default__"

        self.state_dim = state_dim
        self.default_category = str(default_category)
        self._bkt_kwargs = dict(bkt_kwargs or {})
        # user_id -> { category -> tracker }
        self._trackers: Dict[int, Dict[str, BayesianKnowledgeTracing]] = defaultdict(dict)
        self._telemetry: Dict[int, _UserTelemetry] = defaultdict(_UserTelemetry)
        self._lock = threading.RLock()

    @property
    def default_skill(self) -> str:
        """Deprecated alias for ``default_category``."""
        return self.default_category

    # ---- mutation ----
    def observe(
        self,
        user_id: int,
        correct: bool,
        *,
        category: Optional[str] = None,
        timestamp: Optional[float] = None,
        # Deprecated alias
        skill: Optional[str] = None,
    ) -> float:
        """Record one interaction and return the updated competence for the category."""
        import warnings as _w

        if skill is not None:
            _w.warn(
                "Parameter 'skill' is deprecated, use 'category' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if category is None:
                category = skill
        category = str(category or self.default_category)
        ts = float(time.time() if timestamp is None else timestamp)
        with self._lock:
            bkts = self._trackers[int(user_id)]
            if category not in bkts:
                bkts[category] = BayesianKnowledgeTracing(**self._bkt_kwargs)
            p_known = bkts[category].update(bool(correct))
            self._telemetry[int(user_id)].record(correct, ts)
            return float(p_known)

    # ---- read ----
    def state_vec(
        self,
        user_id: int,
        *,
        now: Optional[float] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the current user state vector for ``user_id`` as a 1 x state_dim tensor."""
        t_now = float(time.time() if now is None else now)
        with self._lock:
            bkts = self._trackers.get(int(user_id), {})
            if bkts:
                comp = float(np.mean([bk.p_known for bk in bkts.values()]))
            else:
                # cold-start prior
                comp = float(self._bkt_kwargs.get("p_init", 0.1))
            tel = self._telemetry.get(int(user_id))
            if tel is None:
                fatigue = 0.0
                trust = 0.5
                engagement = 0.0
            else:
                fatigue = tel.fatigue(t_now)
                trust = tel.trust()
                engagement = tel.engagement(t_now)
        vec = torch.tensor(
            [[comp, fatigue, trust, engagement]],
            device=device,
            dtype=dtype,
        )
        return vec

    def competence(self, user_id: int, category: Optional[str] = None) -> float:
        """Return the current competence for a user's category."""
        category = str(category or self.default_category)
        with self._lock:
            bk = self._trackers.get(int(user_id), {}).get(category)
            return float(bk.p_known) if bk is not None else float(self._bkt_kwargs.get("p_init", 0.1))

    def mastery(self, user_id: int, skill: Optional[str] = None) -> float:
        """Deprecated alias for :meth:`competence`."""
        import warnings
        warnings.warn(
            "mastery() is deprecated, use competence() instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.competence(user_id, category=skill)


# ---------------------------------------------------------------------------
# Online residual adapter
# ---------------------------------------------------------------------------
class OnlineUserAdapter(nn.Module):
    """Per-user residual embedding trained online.

    Sits on top of a frozen two-tower's user representation: the effective
    user embedding at rank time is ``u_base + residual[user_id]``. The
    residual starts at zero, so the wrapped model's behaviour is unchanged
    for any user with no observed interactions.

    Update rule (one logistic SGD step per observation)::

        logit = dot(u_base + r_u, i)
        loss  = BCEWithLogits(logit, y) + 0.5 * l2 * ||r_u||²
        r_u -= lr * d loss / d r_u

    Only ``r_u`` receives gradient — the base tower is never touched. Runs on
    the caller's device; expected to be CPU-cheap for emb_dim up to 256.

    Parameters
    ----------
    num_users : int
    emb_dim : int
        Must match the two-tower's projection dimension.
    lr : float
        SGD learning rate for the residual update. 0.05 is a reasonable default
        for emb_dim around 32; scale inversely with dim.
    l2 : float
        Shrinkage coefficient. Non-zero values prevent long-inactive residuals
        from drifting and give newly-returning users a soft reset.
    clip : float
        Per-update L2 norm clip on the residual. Keeps online updates stable
        even under adversarial or noisy feedback.
    """

    def __init__(
        self,
        num_users: int,
        emb_dim: int,
        *,
        lr: float = 0.05,
        l2: float = 1e-3,
        clip: float = 1.0,
    ) -> None:
        super().__init__()
        if num_users <= 0 or emb_dim <= 0:
            raise ValueError("num_users and emb_dim must be positive")
        self.num_users = int(num_users)
        self.emb_dim = int(emb_dim)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.clip = float(clip)
        # Zero-initialised — frozen tower behaviour preserved until first observe()
        self.residual = nn.Embedding(num_users, emb_dim)
        nn.init.zeros_(self.residual.weight)
        self.residual.weight.requires_grad_(False)  # we update manually
        self._lock = threading.RLock()
        self._update_count: Dict[int, int] = defaultdict(int)

    @torch.no_grad()
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.residual(user_ids.long())

    @torch.no_grad()
    def observe(
        self,
        user_id: int,
        u_base: torch.Tensor,      # [emb_dim] — pre-residual user embedding
        item_emb: torch.Tensor,    # [emb_dim] — item embedding
        y: float,                  # observed label in [0, 1] (e.g. correct / accept)
    ) -> float:
        """Apply one SGD step and return the squared-norm of the updated residual."""
        if u_base.shape[-1] != self.emb_dim or item_emb.shape[-1] != self.emb_dim:
            raise ValueError("u_base and item_emb must have shape [emb_dim]")
        uid = int(user_id)
        if not 0 <= uid < self.num_users:
            raise IndexError(f"user_id {uid} out of range [0, {self.num_users})")

        with self._lock:
            r = self.residual.weight[uid].clone()
            u_full = u_base.detach().to(r.dtype).to(r.device) + r
            i = item_emb.detach().to(r.dtype).to(r.device)
            logit = torch.dot(u_full, i)
            # BCE-with-logits gradient w.r.t. u_full is (sigmoid(logit) - y) * i
            p = torch.sigmoid(logit)
            grad = (p - float(y)) * i + self.l2 * r
            r_new = r - self.lr * grad
            # norm clip on the residual itself
            n = torch.linalg.vector_norm(r_new)
            if float(n) > self.clip:
                r_new = r_new * (self.clip / float(n))
            self.residual.weight[uid] = r_new
            self._update_count[uid] += 1
            return float(torch.dot(r_new, r_new))

    def updates_for(self, user_id: int) -> int:
        return int(self._update_count.get(int(user_id), 0))

    def reset_user(self, user_id: int) -> None:
        with self._lock:
            self.residual.weight[int(user_id)] = torch.zeros(self.emb_dim)
            self._update_count.pop(int(user_id), None)


# ---------------------------------------------------------------------------
# Streaming ranker façade
# ---------------------------------------------------------------------------
@dataclass
class AdaptationStats:
    """Lightweight latency summary. Computed over the last N operations."""
    observe_p50_ms: float
    observe_p95_ms: float
    observe_p99_ms: float
    rank_p50_ms: float
    rank_p95_ms: float
    rank_p99_ms: float
    observations: int
    ranks: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "observe_p50_ms": self.observe_p50_ms,
            "observe_p95_ms": self.observe_p95_ms,
            "observe_p99_ms": self.observe_p99_ms,
            "rank_p50_ms": self.rank_p50_ms,
            "rank_p95_ms": self.rank_p95_ms,
            "rank_p99_ms": self.rank_p99_ms,
            "observations": self.observations,
            "ranks": self.ranks,
        }


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


class StreamingAdaptiveRanker:
    """Wraps a frozen two-tower with a streaming update path.

    The wrapped tower is *not* modified — this class only reads from it. A
    per-user :class:`OnlineUserAdapter` and a :class:`BKTStateProvider` sit on
    the side and are updated on every ``observe`` call. ``rank`` reads the
    freshest state + residual for the given user, runs one tower forward, and
    returns the top-k ranked item ids.

    Adaptation latency — the wall-clock gap between an ``observe`` completing
    and a subsequent ``rank`` reflecting that observation — is bounded above
    by one ``observe`` plus one ``rank`` latency. Both are instrumented and
    exposed via :meth:`stats`.

    Parameters
    ----------
    tower : nn.Module
        Any model exposing an ``infer(user_vec, item_matrix, user_ids, item_ids,
        state_vec) -> Tensor[B, K]`` method with a ``user_emb``, ``item_emb``
        and projection stack matching the :class:`TwoTowerRecommender` API.
    user_features : torch.Tensor
        Shape (num_users, user_dim). Frozen user side-info features.
    item_features : torch.Tensor
        Shape (num_items, item_dim). Frozen item side-info features.
    emb_dim : int, optional
        Override the adapter's embedding dimension. Defaults to the tower's.
    lr : float, optional
        Residual adapter learning rate.
    l2 : float, optional
        Residual adapter L2 shrinkage.
    bkt_kwargs : dict, optional
        Forwarded to the BKT trackers.
    history_cap : int, optional
        How many recent observe/rank latencies to retain for stats.
    user_id_map : mapping, optional
        External user ID to internal row index. When omitted, user IDs are
        interpreted as row indices.
    item_id_map : mapping, optional
        External item ID to internal row index. When omitted, item IDs are
        interpreted as row indices.
    index_item_map : mapping, optional
        Internal row index to external item ID for rank outputs.
    """

    def __init__(
        self,
        tower: nn.Module,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        *,
        emb_dim: Optional[int] = None,
        lr: float = 0.05,
        l2: float = 1e-3,
        bkt_kwargs: Optional[dict] = None,
        history_cap: int = 1024,
        monitor: Optional[object] = None,
        item_difficulties: Optional[Sequence[float]] = None,
        scaling_config: Optional[object] = None,
        user_id_map: Optional[Mapping[int, int]] = None,
        item_id_map: Optional[Mapping[int, int]] = None,
        index_item_map: Optional[Mapping[int, int]] = None,
    ) -> None:
        if not hasattr(tower, "infer"):
            raise TypeError("tower must expose an `infer` method")

        num_users = int(user_features.shape[0])
        emb_dim = int(emb_dim or getattr(tower, "emb_dim", user_features.shape[1]))
        state_dim = int(getattr(tower, "state_dim", 4))

        device = str(getattr(tower, "device", "cpu"))
        self.tower = tower.eval()
        # freeze tower gradients — this path never trains the tower
        for p in self.tower.parameters():
            p.requires_grad_(False)

        self.user_features = user_features.to(device)
        self.item_features = item_features.to(device)
        self.device = device
        self._user_id_map = {int(k): int(v) for k, v in (user_id_map or {}).items()}
        self._item_id_map = {int(k): int(v) for k, v in (item_id_map or {}).items()}
        self._index_item_map = {int(k): int(v) for k, v in (index_item_map or {}).items()}

        # Use sparse/sharded backends when a ScalingConfig is provided,
        # falling back to the default dense implementations otherwise.
        if scaling_config is not None:
            from orchid_ranker.scaling import (
                ShardedBKTStateProvider,
                SparseOnlineUserAdapter,
            )
            max_active = getattr(scaling_config, "max_active_users", 1_000_000)
            num_shards = getattr(scaling_config, "num_state_shards", 16)
            max_per_shard = max(1, max_active // max(num_shards, 1))

            self.state = ShardedBKTStateProvider(
                num_shards=num_shards,
                state_dim=state_dim,
                bkt_kwargs=bkt_kwargs,
                max_users_per_shard=max_per_shard,
            )
            self.adapter = SparseOnlineUserAdapter(
                emb_dim=emb_dim, lr=lr, l2=l2,
                max_active_users=max_active,
                device=device,
            )
            logger.info(
                "StreamingAdaptiveRanker: using sparse scaling backend "
                "(max_active=%d, shards=%d)",
                max_active, num_shards,
            )
        else:
            self.state = BKTStateProvider(state_dim=state_dim, bkt_kwargs=bkt_kwargs)
            self.adapter = OnlineUserAdapter(
                num_users=num_users, emb_dim=emb_dim, lr=lr, l2=l2,
            ).to(device)

        self._observe_ms: Deque[float] = deque(maxlen=history_cap)
        self._rank_ms: Deque[float] = deque(maxlen=history_cap)
        self._lock = threading.RLock()
        # Optional live progression monitor. Duck-typed to avoid a hard import
        # cycle with live_metrics.py — anything with a ``record(...)`` method
        # matching the RollingProgressionMonitor signature works.
        self.monitor = monitor
        if item_difficulties is not None:
            self._item_difficulties: Optional[List[float]] = [
                float(d) for d in item_difficulties
            ]
            if len(self._item_difficulties) != int(item_features.shape[0]):
                raise ValueError(
                    "item_difficulties length must equal number of items "
                    f"({len(self._item_difficulties)} vs {item_features.shape[0]})"
                )
        else:
            self._item_difficulties = None

    # ---- internal helpers ----
    def _user_index(self, user_id: int) -> int:
        external = int(user_id)
        if not self._user_id_map:
            return external
        try:
            return self._user_id_map[external]
        except KeyError as exc:
            raise KeyError(f"Unknown user_id={external}") from exc

    def _item_index(self, item_id: int) -> int:
        external = int(item_id)
        if not self._item_id_map:
            return external
        try:
            return self._item_id_map[external]
        except KeyError as exc:
            raise KeyError(f"Unknown item_id={external}") from exc

    def _candidate_pairs(self, candidate_item_ids: Iterable[int]) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        seen: set[int] = set()
        for raw_item_id in candidate_item_ids:
            external = int(raw_item_id)
            internal = self._item_id_map.get(external) if self._item_id_map else external
            if internal is None or internal in seen:
                continue
            output_id = external if self._item_id_map else self._index_item_map.get(internal, internal)
            pairs.append((output_id, internal))
            seen.add(internal)
        return pairs

    @torch.no_grad()
    def _user_base_emb(self, user_id: int, state_vec: torch.Tensor) -> torch.Tensor:
        """Compute the tower's pre-residual user embedding for residual SGD."""
        uid_t = torch.tensor([int(user_id)], dtype=torch.long, device=self.device)
        x_u = self.user_features[uid_t]
        u = self.tower._user_context_vec(x_u, uid_t, state_vec.to(self.device))  # [1, emb_dim]
        return u.squeeze(0).detach()

    @torch.no_grad()
    def _item_emb_for(self, item_id: int) -> torch.Tensor:
        """Tower item embedding for a single item id."""
        iid_t = torch.tensor([int(item_id)], dtype=torch.long, device=self.device)
        I_base = self.tower.item_net(self.item_features.index_select(0, iid_t))
        I = self.tower.item_proj(I_base) + self.tower.item_emb(iid_t)
        return I.squeeze(0).detach()

    # ---- public API ----
    def observe(
        self,
        user_id: int,
        item_id: int,
        correct: bool | int | float,
        *,
        category: Optional[str] = None,
        timestamp: Optional[float] = None,
        # Deprecated alias
        skill: Optional[str] = None,
    ) -> Dict[str, float]:
        """Record a single interaction. O(emb_dim) work; typically sub-millisecond.

        Returns a dict with the updated competence for the category and the
        residual norm after the update.
        """
        import warnings as _w

        if skill is not None:
            _w.warn(
                "Parameter 'skill' is deprecated, use 'category' instead. "
                "Will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            if category is None:
                category = skill

        t0 = time.perf_counter()
        user_idx = self._user_index(user_id)
        item_idx = self._item_index(item_id)
        y = float(1.0 if bool(correct) else 0.0)
        # pre-competence snapshot *before* outcome tracing update -- needed for progression_gain
        pre_competence = self.state.competence(user_idx, category=category)
        # 1. update BKT + telemetry
        p_known = self.state.observe(
            user_idx, bool(correct), category=category, timestamp=timestamp,
        )
        # 2. compute u_base and item_emb from frozen tower
        sv = self.state.state_vec(user_idx, device=self.device)
        u_base = self._user_base_emb(user_idx, sv)
        i_emb = self._item_emb_for(item_idx)
        # 3. online residual SGD step
        residual_norm_sq = self.adapter.observe(user_idx, u_base, i_emb, y)
        dt_ms = 1000.0 * (time.perf_counter() - t0)
        with self._lock:
            self._observe_ms.append(dt_ms)
        # 4. fan out to the live monitor (never allowed to break the path)
        if self.monitor is not None:
            difficulty = None
            if self._item_difficulties is not None and 0 <= int(item_idx) < len(self._item_difficulties):
                difficulty = self._item_difficulties[int(item_idx)]
            try:
                self.monitor.record(  # type: ignore[attr-defined]
                    user_id=user_id,
                    item_id=item_id,
                    correct=bool(correct),
                    pre_competence=float(pre_competence),
                    post_competence=float(p_known),
                    category=str(category or "__default__"),
                    difficulty=difficulty,
                    timestamp=timestamp,
                )
            except Exception:  # noqa: BLE001
                # Monitor hooks are best-effort; never crash serving.
                pass
        return {
            "p_known": float(p_known),
            "pre_competence": float(pre_competence),
            "residual_norm_sq": float(residual_norm_sq),
            "latency_ms": float(dt_ms),
        }

    @torch.no_grad()
    def rank(
        self,
        user_id: int,
        candidate_item_ids: Iterable[int],
        *,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Return the top-k (item_id, score) pairs for ``user_id``.

        State and residual are pulled fresh on every call. There is no cache
        — by design — so this call reflects the most recent ``observe``.
        """
        t0 = time.perf_counter()
        user_idx = self._user_index(user_id)
        pairs = self._candidate_pairs(candidate_item_ids)
        if not pairs:
            return []
        external_ids = [external for external, _internal in pairs]
        cand = [internal for _external, internal in pairs]
        uid_t = torch.tensor([int(user_idx)], dtype=torch.long, device=self.device)
        iid_t = torch.tensor(cand, dtype=torch.long, device=self.device)
        sv = self.state.state_vec(user_idx, device=self.device)
        x_u = self.user_features[uid_t]

        # Base tower logits [1, K]
        logits = self.tower.infer(
            user_vec=x_u,
            item_matrix=self.item_features,
            user_ids=uid_t,
            item_ids=iid_t,
            state_vec=sv,
        ).view(-1)

        # Residual contribution: dot(r_u, i_emb) — add to each candidate score
        r_u = self.adapter(uid_t).view(-1)  # [emb_dim]
        if float(torch.linalg.vector_norm(r_u)) > 0.0:
            I_base = self.tower.item_net(self.item_features.index_select(0, iid_t))
            I = self.tower.item_proj(I_base) + self.tower.item_emb(iid_t)  # [K, emb_dim]
            bonus = (I @ r_u).view(-1)
            logits = logits + bonus

        k = max(1, min(int(top_k), len(cand)))
        vals, idxs = torch.topk(logits, k=k)
        out = [(external_ids[int(i)], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]
        dt_ms = 1000.0 * (time.perf_counter() - t0)
        with self._lock:
            self._rank_ms.append(dt_ms)
        return out

    # ---- introspection ----
    def stats(self) -> AdaptationStats:
        with self._lock:
            obs = list(self._observe_ms)
            rnk = list(self._rank_ms)
        return AdaptationStats(
            observe_p50_ms=_pct(obs, 50),
            observe_p95_ms=_pct(obs, 95),
            observe_p99_ms=_pct(obs, 99),
            rank_p50_ms=_pct(rnk, 50),
            rank_p95_ms=_pct(rnk, 95),
            rank_p99_ms=_pct(rnk, 99),
            observations=len(obs),
            ranks=len(rnk),
        )

    def competence(self, user_id: int, category: Optional[str] = None) -> float:
        """Return the current competence for a user's category."""
        return self.state.competence(self._user_index(user_id), category=category)

    def mastery(self, user_id: int, skill: Optional[str] = None) -> float:
        """Deprecated alias for :meth:`competence`."""
        import warnings
        warnings.warn(
            "mastery() is deprecated, use competence() instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.state.competence(self._user_index(user_id), category=skill)

    def updates_for(self, user_id: int) -> int:
        return self.adapter.updates_for(self._user_index(user_id))
