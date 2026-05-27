"""Personalized contextual bandit baselines for adaptive serving."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "BanditScore",
    "PersonalizedLinUCB",
]


@dataclass(frozen=True)
class BanditScore:
    """Mean, uncertainty, and optimistic score for one candidate action."""

    item_id: Any
    mean: float
    bonus: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PersonalizedLinUCB:
    """Personalized LinUCB over ``phi(user, item)`` features.

    The feature map concatenates user features, item features, and their
    elementwise interaction when dimensions match. This replaces item-only UCB
    baselines for adaptive-learning scenarios where learner state matters.
    """

    def __init__(self, *, alpha: float = 1.0, l2: float = 1.0) -> None:
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if l2 <= 0.0:
            raise ValueError("l2 must be positive")
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.A_: np.ndarray | None = None
        self.b_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.A_ is not None and self.b_ is not None

    def score(
        self,
        user_features: Sequence[float],
        item_features: Mapping[Any, Sequence[float]],
    ) -> dict[Any, BanditScore]:
        if not item_features:
            return {}
        user = _vector(user_features, "user_features")
        phis = {item_id: self.phi(user, _vector(features, "item_features")) for item_id, features in item_features.items()}
        self._ensure_dim(len(next(iter(phis.values()))))
        assert self.A_ is not None and self.b_ is not None
        A_inv = np.linalg.inv(self.A_)
        theta = A_inv @ self.b_
        scores: dict[Any, BanditScore] = {}
        for item_id, phi in phis.items():
            mean = float(phi @ theta)
            bonus = float(self.alpha * np.sqrt(max(0.0, phi @ A_inv @ phi)))
            scores[item_id] = BanditScore(item_id=item_id, mean=mean, bonus=bonus, score=mean + bonus)
        return scores

    def recommend(
        self,
        user_features: Sequence[float],
        item_features: Mapping[Any, Sequence[float]],
        *,
        top_k: int = 1,
    ) -> list[Any]:
        if top_k <= 0:
            return []
        scores = self.score(user_features, item_features)
        ranked = sorted(scores, key=lambda item_id: (scores[item_id].score, str(item_id)), reverse=True)
        return ranked[: min(int(top_k), len(ranked))]

    def update(
        self,
        user_features: Sequence[float],
        item_features: Sequence[float],
        reward: float,
    ) -> None:
        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError("reward must be finite")
        user = _vector(user_features, "user_features")
        item = _vector(item_features, "item_features")
        phi = self.phi(user, item)
        self._ensure_dim(len(phi))
        assert self.A_ is not None and self.b_ is not None
        self.A_ += np.outer(phi, phi)
        self.b_ += reward_value * phi

    def phi(self, user_features: np.ndarray, item_features: np.ndarray) -> np.ndarray:
        if user_features.ndim != 1 or item_features.ndim != 1:
            raise ValueError("user_features and item_features must be one-dimensional")
        parts = [user_features, item_features]
        if user_features.shape == item_features.shape:
            parts.append(user_features * item_features)
        return np.concatenate(parts).astype(np.float64)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "l2": self.l2,
            "is_fitted": self.is_fitted,
            "feature_dim": None if self.A_ is None else int(self.A_.shape[0]),
        }

    def _ensure_dim(self, dim: int) -> None:
        if self.A_ is None or self.b_ is None:
            self.A_ = np.eye(int(dim), dtype=np.float64) * self.l2
            self.b_ = np.zeros((int(dim),), dtype=np.float64)
            return
        if self.A_.shape != (dim, dim):
            raise ValueError(f"feature dimension changed from {self.A_.shape[0]} to {dim}")


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr
