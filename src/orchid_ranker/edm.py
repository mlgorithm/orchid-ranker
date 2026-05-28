"""Classical educational data mining baselines for adaptive learning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

__all__ = [
    "AFMTracer",
    "EDMTrainingReport",
    "PFATracer",
]


@dataclass(frozen=True)
class EDMTrainingReport:
    """Training summary for a classical EDM tracer."""

    n_examples: int
    n_users: int
    n_items: int
    n_concepts: int
    positive_rate: float
    fallback_only: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PFATracer:
    """Performance Factors Analysis baseline.

    PFA is an interpretable logistic model over learner concept history. Orchid
    uses it as a small-data baseline alongside BKT, DKT, and attention-based KT.
    """

    def __init__(self, *, C: float = 1.0, max_iter: int = 200) -> None:
        if C <= 0.0:
            raise ValueError("C must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.vectorizer_: Optional[DictVectorizer] = None
        self.model_: Optional[LogisticRegression] = None
        self.report_: Optional[EDMTrainingReport] = None
        self.global_prior_: float = 0.5
        self.item_to_concept_: dict[Any, Any] = {}
        self._successes: dict[tuple[Any, Any], int] = {}
        self._failures: dict[tuple[Any, Any], int] = {}

    @property
    def is_fitted(self) -> bool:
        return self.report_ is not None

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        correct_col: str = "correct",
        timestamp_col: Optional[str] = None,
        concept_col: Optional[str] = None,
    ) -> "PFATracer":
        """Fit from chronological learner outcomes."""
        required = {user_col, item_col, correct_col}
        if timestamp_col is not None:
            required.add(timestamp_col)
        if concept_col is not None:
            required.add(concept_col)
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing required columns: {sorted(missing)}")

        work = _ordered(interactions, user_col=user_col, timestamp_col=timestamp_col)
        concept_series = work[concept_col] if concept_col is not None else work[item_col]
        self.item_to_concept_ = dict(zip(work[item_col], concept_series))
        labels = [_label(value) for value in work[correct_col].tolist()]
        self.global_prior_ = float(np.mean(labels)) if labels else 0.5

        features: list[dict[str, Any]] = []
        y: list[int] = []
        successes: dict[tuple[Any, Any], int] = {}
        failures: dict[tuple[Any, Any], int] = {}
        for user_id, item_id, concept_id, correct in zip(
            work[user_col].tolist(),
            work[item_col].tolist(),
            concept_series.tolist(),
            labels,
        ):
            features.append(self._features(user_id, item_id, concept_id, successes, failures))
            y.append(correct)
            key = (user_id, concept_id)
            if correct:
                successes[key] = successes.get(key, 0) + 1
            else:
                failures[key] = failures.get(key, 0) + 1

        self._successes = successes
        self._failures = failures
        fallback_only = len(set(y)) < 2
        self.vectorizer_ = DictVectorizer(sparse=True)
        if fallback_only:
            self.model_ = None
        else:
            X = self.vectorizer_.fit_transform(features)
            self.model_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="liblinear").fit(X, y)

        self.report_ = EDMTrainingReport(
            n_examples=len(y),
            n_users=int(work[user_col].nunique()),
            n_items=int(work[item_col].nunique()),
            n_concepts=int(pd.Series(concept_series).nunique()),
            positive_rate=self.global_prior_,
            fallback_only=fallback_only,
        )
        return self

    def predict_correct(self, user_id: Any, item_id: Any) -> float:
        return float(self.predict_many(user_id, [item_id])[item_id])

    def predict_many(self, user_id: Any, item_ids: Sequence[Any]) -> dict[Any, float]:
        self._require_fitted()
        if not item_ids:
            return {}
        features = [
            self._features(user_id, item_id, self.item_to_concept_.get(item_id, item_id), self._successes, self._failures)
            for item_id in item_ids
        ]
        if self.model_ is None or self.vectorizer_ is None:
            return {item_id: self.global_prior_ for item_id in item_ids}
        X = self.vectorizer_.transform(features)
        probs = self.model_.predict_proba(X)[:, 1]
        return {item_id: float(np.clip(prob, 0.0, 1.0)) for item_id, prob in zip(item_ids, probs)}

    def observe(self, user_id: Any, item_id: Any, correct: Any) -> int:
        self._require_fitted()
        concept_id = self.item_to_concept_.get(item_id, item_id)
        key = (user_id, concept_id)
        if _label(correct):
            self._successes[key] = self._successes.get(key, 0) + 1
        else:
            self._failures[key] = self._failures.get(key, 0) + 1
        return self._successes.get(key, 0) + self._failures.get(key, 0)

    def diagnostics(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.report_ is not None
        return self.report_.to_dict()

    def _features(
        self,
        user_id: Any,
        item_id: Any,
        concept_id: Any,
        successes: dict[tuple[Any, Any], int],
        failures: dict[tuple[Any, Any], int],
    ) -> dict[str, Any]:
        del item_id
        key = (user_id, concept_id)
        return {
            "bias": 1.0,
            f"concept={concept_id}": 1.0,
            "successes": float(successes.get(key, 0)),
            "failures": float(failures.get(key, 0)),
        }

    def _require_fitted(self) -> None:
        if self.report_ is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use")


class AFMTracer(PFATracer):
    """Additive Factors Model baseline.

    AFM uses concept identity and total opportunities rather than separate
    success/failure factors. It is a concise, interpretable comparator for
    smaller adaptive-learning datasets.
    """

    def _features(
        self,
        user_id: Any,
        item_id: Any,
        concept_id: Any,
        successes: dict[tuple[Any, Any], int],
        failures: dict[tuple[Any, Any], int],
    ) -> dict[str, Any]:
        del item_id
        key = (user_id, concept_id)
        attempts = successes.get(key, 0) + failures.get(key, 0)
        return {
            "bias": 1.0,
            f"concept={concept_id}": 1.0,
            "attempts": float(attempts),
        }


def _ordered(interactions: pd.DataFrame, *, user_col: str, timestamp_col: Optional[str]) -> pd.DataFrame:
    work = interactions.copy()
    work["__orchid_order__"] = np.arange(len(work))
    sort_cols = [user_col]
    if timestamp_col is not None:
        sort_cols.append(timestamp_col)
    sort_cols.append("__orchid_order__")
    return work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("correct labels must be finite")
    return int(numeric >= 0.5)
