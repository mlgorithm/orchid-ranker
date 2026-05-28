"""Item-response-theory utilities for adaptive testing and placement."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

__all__ = [
    "IRTAdaptiveSelector",
    "IRTItem",
    "IRTRecommendation",
]


@dataclass(frozen=True)
class IRTItem:
    """Parameters for a 1PL/2PL/3PL item-response model."""

    item_id: Any
    difficulty: float
    discrimination: float = 1.0
    guessing: float = 0.0
    concept_id: Optional[Any] = None


@dataclass(frozen=True)
class IRTRecommendation:
    """Next-item recommendation from an IRT adaptive selector."""

    item_id: Any
    score: float
    p_correct: float
    information: float
    difficulty: float
    discrimination: float
    guessing: float
    concept_id: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IRTAdaptiveSelector:
    """Small IRT engine for placement, mastery checks, and adaptive tests.

    The selector keeps a scalar learner ability estimate ``theta`` and ranks
    candidate items by Fisher information, optionally filtered by prerequisite
    concepts. It supports Rasch/1PL behavior by leaving discrimination at 1.0,
    2PL by setting discrimination per item, and 3PL by setting a non-zero
    guessing parameter.
    """

    def __init__(
        self,
        *,
        initial_theta: float = 0.0,
        learning_rate: float = 0.35,
        min_theta: float = -6.0,
        max_theta: float = 6.0,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if min_theta >= max_theta:
            raise ValueError("min_theta must be smaller than max_theta")
        self.theta = float(initial_theta)
        self.learning_rate = float(learning_rate)
        self.min_theta = float(min_theta)
        self.max_theta = float(max_theta)
        self.items_: dict[Any, IRTItem] = {}
        self.history_: list[tuple[Any, int, float]] = []

    @property
    def is_fitted(self) -> bool:
        return bool(self.items_)

    def fit_items(
        self,
        items: pd.DataFrame | Sequence[IRTItem] | Sequence[Mapping[str, Any]],
        *,
        item_col: str = "item_id",
        difficulty_col: str = "difficulty",
        discrimination_col: Optional[str] = None,
        guessing_col: Optional[str] = None,
        concept_col: Optional[str] = None,
    ) -> "IRTAdaptiveSelector":
        """Load item parameters from a dataframe, mappings, or ``IRTItem`` objects."""
        self.items_ = {}
        for item in _iter_items(
            items,
            item_col=item_col,
            difficulty_col=difficulty_col,
            discrimination_col=discrimination_col,
            guessing_col=guessing_col,
            concept_col=concept_col,
        ):
            _validate_item(item)
            self.items_[item.item_id] = item
        if not self.items_:
            raise ValueError("items must contain at least one item")
        return self

    def probability(self, item_id: Any, *, theta: Optional[float] = None) -> float:
        """Return the probability of a correct response for one item."""
        self._require_fitted()
        item = self._item(item_id)
        ability = self.theta if theta is None else float(theta)
        base = _sigmoid(item.discrimination * (ability - item.difficulty))
        return _clamp01(item.guessing + (1.0 - item.guessing) * base)

    def information(self, item_id: Any, *, theta: Optional[float] = None) -> float:
        """Return item information at the current or supplied ability."""
        self._require_fitted()
        item = self._item(item_id)
        p = self.probability(item_id, theta=theta)
        if p <= 1e-9 or p >= 1.0:
            return 0.0
        if item.guessing > 0.0:
            adjusted = max((p - item.guessing) / max(1.0 - item.guessing, 1e-9), 1e-9)
            return float(item.discrimination**2 * adjusted**2 * (1.0 - p) / p)
        return float(item.discrimination**2 * p * (1.0 - p))

    def observe(self, item_id: Any, correct: Any) -> float:
        """Update ability from one response and return the new ``theta``."""
        self._require_fitted()
        item = self._item(item_id)
        label = 1 if bool(correct) else 0
        p = self.probability(item_id)
        grad = item.discrimination * (label - p)
        self.theta = min(self.max_theta, max(self.min_theta, self.theta + self.learning_rate * grad))
        self.history_.append((item_id, label, self.theta))
        return self.theta

    def recommend(
        self,
        candidate_item_ids: Optional[Sequence[Any]] = None,
        *,
        top_k: int = 1,
        prerequisite_by_concept: Optional[Mapping[Any, Sequence[Any]]] = None,
        mastered_concepts: Optional[Sequence[Any]] = None,
    ) -> list[IRTRecommendation]:
        """Rank items by information subject to optional prerequisite constraints."""
        self._require_fitted()
        if top_k <= 0:
            return []
        candidates = list(candidate_item_ids) if candidate_item_ids is not None else list(self.items_)
        mastered = set(mastered_concepts or [])
        prerequisites = {concept: set(reqs) for concept, reqs in dict(prerequisite_by_concept or {}).items()}
        recs = []
        for item_id in candidates:
            if item_id not in self.items_:
                continue
            item = self.items_[item_id]
            if item.concept_id in prerequisites and not prerequisites[item.concept_id].issubset(mastered):
                continue
            p_correct = self.probability(item_id)
            info = self.information(item_id)
            recs.append(
                IRTRecommendation(
                    item_id=item_id,
                    score=info,
                    p_correct=p_correct,
                    information=info,
                    difficulty=item.difficulty,
                    discrimination=item.discrimination,
                    guessing=item.guessing,
                    concept_id=item.concept_id,
                )
            )
        recs.sort(key=lambda rec: (rec.score, -abs(rec.difficulty - self.theta), str(rec.item_id)), reverse=True)
        return recs[: min(int(top_k), len(recs))]

    def _item(self, item_id: Any) -> IRTItem:
        try:
            return self.items_[item_id]
        except KeyError as exc:
            raise KeyError(f"Unknown item_id={item_id!r}") from exc

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("IRTAdaptiveSelector.fit_items must be called before use")


def _iter_items(
    items: pd.DataFrame | Sequence[IRTItem] | Sequence[Mapping[str, Any]],
    *,
    item_col: str,
    difficulty_col: str,
    discrimination_col: Optional[str],
    guessing_col: Optional[str],
    concept_col: Optional[str],
) -> list[IRTItem]:
    if isinstance(items, pd.DataFrame):
        missing = [column for column in [item_col, difficulty_col] if column not in items.columns]
        if missing:
            raise ValueError(f"items missing required columns: {missing}")
        result = []
        for _, row in items.iterrows():
            result.append(
                IRTItem(
                    item_id=row[item_col],
                    difficulty=float(row[difficulty_col]),
                    discrimination=float(row[discrimination_col]) if discrimination_col else 1.0,
                    guessing=float(row[guessing_col]) if guessing_col else 0.0,
                    concept_id=row[concept_col] if concept_col else None,
                )
            )
        return result

    result = []
    for raw in items:
        if isinstance(raw, IRTItem):
            result.append(raw)
        else:
            result.append(
                IRTItem(
                    item_id=raw[item_col],
                    difficulty=float(raw[difficulty_col]),
                    discrimination=float(raw[discrimination_col]) if discrimination_col else float(raw.get("discrimination", 1.0)),
                    guessing=float(raw[guessing_col]) if guessing_col else float(raw.get("guessing", 0.0)),
                    concept_id=raw[concept_col] if concept_col else raw.get("concept_id"),
                )
            )
    return result


def _validate_item(item: IRTItem) -> None:
    if item.discrimination <= 0.0:
        raise ValueError("item discrimination must be positive")
    if not 0.0 <= item.guessing < 1.0:
        raise ValueError("item guessing must be in [0, 1)")
    if not all(math.isfinite(value) for value in [item.difficulty, item.discrimination, item.guessing]):
        raise ValueError("IRT item parameters must be finite")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
