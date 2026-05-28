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
    ) -> None:
        if not 0.0 < request_retention < 1.0:
            raise ValueError("request_retention must be inside (0, 1)")
        if min_interval_days <= 0.0:
            raise ValueError("min_interval_days must be positive")
        if max_interval_days < min_interval_days:
            raise ValueError("max_interval_days must be >= min_interval_days")
        self.request_retention = float(request_retention)
        self.min_interval_days = float(min_interval_days)
        self.max_interval_days = float(max_interval_days)

    def retrievability(self, state: FSRSReviewState, *, now: datetime | None = None) -> float:
        """Return retention probability under exponential half-life decay."""
        if state.last_review_at is None:
            return 0.0
        active_now = _utc(now)
        elapsed_days = max(0.0, (active_now - _utc(state.last_review_at)).total_seconds() / 86400.0)
        stability = max(float(state.stability), 1e-6)
        return float(np.exp(np.log(self.request_retention) * elapsed_days / stability))

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
        old = state or FSRSReviewState(last_review_at=active_now)
        recall = grade >= 2
        retrievability = self.retrievability(old, now=active_now) if old.last_review_at is not None else 0.0
        difficulty = _clip(float(old.difficulty) + (3.0 - float(grade)) * 0.35, 1.0, 10.0)
        if not recall:
            stability = max(self.min_interval_days, float(old.stability) * 0.45)
            lapses = old.lapses + 1
        else:
            grade_bonus = {2: 1.10, 3: 1.75, 4: 2.50}[grade]
            recall_bonus = 1.0 + max(0.0, 1.0 - retrievability)
            difficulty_penalty = 1.0 + (difficulty - 5.0) / 20.0
            stability = max(self.min_interval_days, float(old.stability) * grade_bonus * recall_bonus / difficulty_penalty)
            lapses = old.lapses
        stability = _clip(stability, self.min_interval_days, self.max_interval_days)
        interval_days = _clip(stability, self.min_interval_days, self.max_interval_days)
        return FSRSReviewState(
            stability=float(stability),
            difficulty=float(difficulty),
            due_at=active_now + timedelta(days=float(interval_days)),
            last_review_at=active_now,
            repetitions=old.repetitions + 1,
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
