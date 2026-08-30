"""Small-data empirical tracer used before knowledge tracing is well supported.

The adaptive-learning product needs to be useful during a pilot, when a
transformer-sized knowledge tracer would be more expressive than the data can
justify.  ``EmpiricalTracer`` is a deliberately transparent hierarchical
baseline: it combines global, item, learner, and learner-item correctness
rates with beta-style smoothing and updates immediately after each outcome.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .adaptive_schema import normalize_timestamps

__all__ = ["EmpiricalTracer"]


class EmpiricalTracer:
    """Predict correctness from smoothed global, item, and learner outcomes.

    This is an adaptive-learning fallback, not a replacement for a fitted
    sequence model.  It has no hidden training loop, works with sparse pilot
    data, and exposes the same prediction/observation surface used by Orchid's
    learning policies.
    """

    def __init__(
        self,
        *,
        correct_threshold: float = 0.5,
        item_prior_strength: float = 8.0,
        user_prior_strength: float = 8.0,
        user_item_prior_strength: float = 4.0,
    ) -> None:
        if not 0.0 <= float(correct_threshold) <= 1.0:
            raise ValueError("correct_threshold must be in [0, 1]")
        for name, value in (
            ("item_prior_strength", item_prior_strength),
            ("user_prior_strength", user_prior_strength),
            ("user_item_prior_strength", user_item_prior_strength),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite value")
        self.correct_threshold = float(correct_threshold)
        self.item_prior_strength = float(item_prior_strength)
        self.user_prior_strength = float(user_prior_strength)
        self.user_item_prior_strength = float(user_item_prior_strength)
        self.item_ids_: list[Any] = []
        self.is_fitted: bool = False
        self._global_successes: float = 0.0
        self._global_count: float = 0.0
        self._item_successes: dict[Any, float] = {}
        self._item_count: dict[Any, float] = {}
        self._user_successes: dict[Any, float] = {}
        self._user_count: dict[Any, float] = {}
        self._user_item_successes: dict[tuple[Any, Any], float] = {}
        self._user_item_count: dict[tuple[Any, Any], float] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        correct_col: str = "correct",
        timestamp_col: Optional[str] = None,
        **_: Any,
    ) -> "EmpiricalTracer":
        """Fit the smoothed empirical state from completed interactions."""
        required = {user_col, item_col, correct_col}
        if timestamp_col is not None:
            required.add(timestamp_col)
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing required columns: {sorted(missing)}")
        if interactions.empty:
            raise ValueError("interactions DataFrame is empty")

        self.item_ids_ = sorted(interactions[item_col].drop_duplicates().tolist(), key=lambda value: str(value))
        self._global_successes = 0.0
        self._global_count = 0.0
        self._item_successes = {}
        self._item_count = {}
        self._user_successes = {}
        self._user_count = {}
        self._user_item_successes = {}
        self._user_item_count = {}
        work = interactions.copy()
        if timestamp_col is not None:
            work[timestamp_col] = normalize_timestamps(work[timestamp_col], timestamp_col)
        work["__orchid_order__"] = np.arange(len(work))
        order = [user_col]
        if timestamp_col is not None:
            order.append(timestamp_col)
        order.append("__orchid_order__")
        work = work.sort_values(order, kind="mergesort")
        for user_id, item_id, correct in work[[user_col, item_col, correct_col]].itertuples(index=False, name=None):
            self.observe(user_id, item_id, correct)
        self.is_fitted = True
        return self

    def predict_correct(self, user_id: Any, item_id: Any) -> float:
        """Return a hierarchical, smoothed probability of a correct outcome."""
        self._require_fitted()
        global_rate = self._global_rate()
        item_rate = self._smoothed_rate(
            self._item_successes.get(item_id, 0.0),
            self._item_count.get(item_id, 0.0),
            prior=global_rate,
            strength=self.item_prior_strength,
        )
        user_rate = self._smoothed_rate(
            self._user_successes.get(user_id, 0.0),
            self._user_count.get(user_id, 0.0),
            prior=global_rate,
            strength=self.user_prior_strength,
        )
        base_rate = 0.65 * item_rate + 0.35 * user_rate
        pair = (user_id, item_id)
        pair_rate = self._smoothed_rate(
            self._user_item_successes.get(pair, 0.0),
            self._user_item_count.get(pair, 0.0),
            prior=base_rate,
            strength=self.user_item_prior_strength,
        )
        pair_count = self._user_item_count.get(pair, 0.0)
        personalization = min(0.70, pair_count / (pair_count + self.user_item_prior_strength))
        return _clamp_probability((1.0 - personalization) * base_rate + personalization * pair_rate)

    def predict_many(self, user_id: Any, item_ids: Sequence[Any]) -> dict[Any, float]:
        """Return correctness predictions for each supplied candidate item."""
        self._require_fitted()
        return {item_id: self.predict_correct(user_id, item_id) for item_id in item_ids}

    def observe(self, user_id: Any, item_id: Any, correct: Any, *, timestamp: Optional[Any] = None) -> int:
        """Update empirical counts after one completed interaction."""
        del timestamp
        label = _label(correct, threshold=self.correct_threshold)
        pair = (user_id, item_id)
        self._global_successes += label
        self._global_count += 1.0
        self._item_successes[item_id] = self._item_successes.get(item_id, 0.0) + label
        self._item_count[item_id] = self._item_count.get(item_id, 0.0) + 1.0
        self._user_successes[user_id] = self._user_successes.get(user_id, 0.0) + label
        self._user_count[user_id] = self._user_count.get(user_id, 0.0) + 1.0
        self._user_item_successes[pair] = self._user_item_successes.get(pair, 0.0) + label
        self._user_item_count[pair] = self._user_item_count.get(pair, 0.0) + 1.0
        if item_id not in self.item_ids_:
            self.item_ids_.append(item_id)
        return int(self._user_count[user_id])

    def _global_rate(self) -> float:
        return self._smoothed_rate(
            self._global_successes,
            self._global_count,
            prior=0.5,
            strength=self.item_prior_strength,
        )

    @staticmethod
    def _smoothed_rate(successes: float, count: float, *, prior: float, strength: float) -> float:
        return _clamp_probability((float(successes) + float(strength) * float(prior)) / (float(count) + float(strength)))

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("EmpiricalTracer must be fitted before use")


def _label(value: Any, *, threshold: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("correct labels must be finite")
    return float(numeric >= threshold)


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
