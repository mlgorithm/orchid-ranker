"""Spaced-repetition scheduling utilities for adaptive learning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

__all__ = [
    "FSRSReviewState",
    "FSRSScheduler",
    "ReviewRecommendation",
]

# FSRS-5 forgetting curve constants. Retrievability follows the power law
# R(t, S) = (1 + FACTOR * t / S) ** DECAY, calibrated so that R(S) = 0.9 — i.e.
# stability S is, by definition, the time at which retrievability falls to 0.9.
_DECAY = -0.5
_FACTOR = 0.9 ** (1.0 / _DECAY) - 1.0  # = 19/81 ≈ 0.234567...

# FSRS-5 default weights (the 19 fitted parameters w0..w18). These are the
# published defaults; callers can override via FSRSScheduler(weights=...).
_FSRS5_DEFAULT_WEIGHTS = (
    0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604, 0.0046,
    1.54575, 0.1192, 1.01925, 1.9395, 0.11, 0.29605, 2.2698, 0.2315,
    2.9898, 0.51655, 0.6621,
)
_MIN_STABILITY = 0.01


@dataclass(frozen=True)
class FSRSReviewState:
    """Memory state for one learner-item pair."""

    stability: float = 1.0
    difficulty: float = 5.0
    due_at: datetime | None = None
    last_review_at: datetime | None = None
    repetitions: int = 0
    lapses: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["due_at"] = None if self.due_at is None else self.due_at.isoformat()
        data["last_review_at"] = None if self.last_review_at is None else self.last_review_at.isoformat()
        return data


@dataclass(frozen=True)
class ReviewRecommendation:
    """Review urgency score for a learner-item memory state."""

    item_id: Any
    retrievability: float
    urgency: float
    due: bool
    due_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["due_at"] = None if self.due_at is None else self.due_at.isoformat()
        return data


class FSRSScheduler:
    """Small FSRS-style scheduler for retention-aware adaptive policies.

    This is a lightweight serving component, not a full FSRS reproduction. It
    captures the key operating behavior Orchid needs: estimate forgetting risk,
    update stability/difficulty after a review grade, and rank due reviews.
    """

    def __init__(
        self,
        *,
        request_retention: float = 0.90,
        min_interval_days: float = 0.25,
        max_interval_days: float = 365.0,
        weights: tuple[float, ...] = _FSRS5_DEFAULT_WEIGHTS,
    ) -> None:
        if not 0.0 < request_retention < 1.0:
            raise ValueError("request_retention must be inside (0, 1)")
        if min_interval_days <= 0.0:
            raise ValueError("min_interval_days must be positive")
        if max_interval_days < min_interval_days:
            raise ValueError("max_interval_days must be >= min_interval_days")
        if len(weights) < 19:
            raise ValueError("weights must provide at least 19 FSRS parameters")
        self.request_retention = float(request_retention)
        self.min_interval_days = float(min_interval_days)
        self.max_interval_days = float(max_interval_days)
        self.w = tuple(float(value) for value in weights)

    def retrievability(self, state: FSRSReviewState, *, now: datetime | None = None) -> float:
        """Return retention probability under the FSRS power-law forgetting curve.

        ``R(t, S) = (1 + FACTOR * t / S) ** DECAY`` with fixed ``DECAY = -0.5``,
        so ``R(S) = 0.9`` by construction. Heavier-tailed than the SM-2
        exponential, which is the point of FSRS.
        """
        if state.last_review_at is None:
            return 0.0
        active_now = _utc(now)
        elapsed_days = max(0.0, (active_now - _utc(state.last_review_at)).total_seconds() / 86400.0)
        stability = max(float(state.stability), _MIN_STABILITY)
        return float((1.0 + _FACTOR * elapsed_days / stability) ** _DECAY)

    def interval_for(self, stability: float) -> float:
        """Days until retrievability decays to ``request_retention`` (clipped).

        Inverts the power curve: ``I = (S / FACTOR) * (rr ** (1 / DECAY) - 1)``.
        Higher ``request_retention`` ⇒ shorter intervals.
        """
        s = max(float(stability), _MIN_STABILITY)
        interval = (s / _FACTOR) * (self.request_retention ** (1.0 / _DECAY) - 1.0)
        return _clip(interval, self.min_interval_days, self.max_interval_days)

    # --- FSRS-5 stability / difficulty update equations -------------------
    def _initial_difficulty(self, grade: int) -> float:
        return _clip(self.w[4] - np.exp(self.w[5] * (grade - 1.0)) + 1.0, 1.0, 10.0)

    def _next_difficulty(self, difficulty: float, grade: int) -> float:
        delta = -self.w[6] * (grade - 3.0)
        damped = difficulty + delta * (10.0 - difficulty) / 9.0  # linear damping
        reverted = self.w[7] * self._initial_difficulty(4) + (1.0 - self.w[7]) * damped
        return _clip(reverted, 1.0, 10.0)

    def _next_stability_recall(self, difficulty: float, stability: float, retrievability: float, grade: int) -> float:
        hard_penalty = self.w[15] if grade == 2 else 1.0
        easy_bonus = self.w[16] if grade == 4 else 1.0
        growth = (
            np.exp(self.w[8])
            * (11.0 - difficulty)
            * stability ** (-self.w[9])
            * (np.exp(self.w[10] * (1.0 - retrievability)) - 1.0)
            * hard_penalty
            * easy_bonus
        )
        return float(stability * (1.0 + growth))

    def _next_stability_forget(self, difficulty: float, stability: float, retrievability: float) -> float:
        post_lapse = (
            self.w[11]
            * difficulty ** (-self.w[12])
            * ((stability + 1.0) ** self.w[13] - 1.0)
            * np.exp(self.w[14] * (1.0 - retrievability))
        )
        # A lapse can never increase stability.
        return float(min(post_lapse, stability))

    def review(
        self,
        state: FSRSReviewState | None,
        *,
        grade: int,
        now: datetime | None = None,
    ) -> FSRSReviewState:
        """Update memory state from a 1-4 review grade."""
        if grade not in {1, 2, 3, 4}:
            raise ValueError("grade must be one of 1, 2, 3, 4")
        active_now = _utc(now)
        recall = grade >= 2

        first_review = state is None or state.last_review_at is None or state.repetitions <= 0
        if first_review:
            # FSRS initial stability S0(g) = w[g-1]; initial difficulty D0(g).
            stability = max(_MIN_STABILITY, self.w[grade - 1])
            difficulty = self._initial_difficulty(grade)
            lapses = 0 if recall else 1
            repetitions = 1
        else:
            assert state is not None
            r = self.retrievability(state, now=active_now)
            difficulty = self._next_difficulty(float(state.difficulty), grade)
            if recall:
                stability = self._next_stability_recall(difficulty, max(float(state.stability), _MIN_STABILITY), r, grade)
                lapses = state.lapses
            else:
                stability = self._next_stability_forget(difficulty, max(float(state.stability), _MIN_STABILITY), r)
                lapses = state.lapses + 1
            repetitions = state.repetitions + 1

        stability = _clip(stability, _MIN_STABILITY, self.max_interval_days)
        interval_days = self.interval_for(stability)
        return FSRSReviewState(
            stability=float(stability),
            difficulty=float(difficulty),
            due_at=active_now + timedelta(days=float(interval_days)),
            last_review_at=active_now,
            repetitions=repetitions,
            lapses=lapses,
        )

    def recommend_reviews(
        self,
        states: dict[Any, FSRSReviewState],
        *,
        now: datetime | None = None,
        top_k: int = 10,
    ) -> list[ReviewRecommendation]:
        """Rank learner-item states by forgetting urgency."""
        if top_k <= 0:
            return []
        active_now = _utc(now)
        recs = []
        for item_id, state in states.items():
            r = self.retrievability(state, now=active_now)
            due = state.due_at is None or _utc(state.due_at) <= active_now
            due_bonus = 0.25 if due else 0.0
            urgency = float(np.clip(1.0 - r + due_bonus, 0.0, 1.25))
            recs.append(ReviewRecommendation(item_id=item_id, retrievability=r, urgency=urgency, due=due, due_at=state.due_at))
        recs.sort(key=lambda rec: (rec.urgency, str(rec.item_id)), reverse=True)
        return recs[: min(int(top_k), len(recs))]


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))
