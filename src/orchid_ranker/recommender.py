"""High-level Surprise-like recommender interface for Orchid Ranker."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .baselines import (
    ALSBaseline,
    ExplicitMFBaseline,
    ImplicitALSBaseline,
    ImplicitBPRBaseline,
    LinUCBBaseline,
    UserKNNBaseline,
    NeuralMatrixFactorizationBaseline,
    PopularityBaseline,
    RandomBaseline,
)
from .utils.validation import ValidationError, validate_interactions_frame, validate_item_features


@dataclass
class Recommendation:
    item_id: int
    score: float


SUPPORTED_STRATEGIES: Tuple[str, ...] = (
    "als",
    "explicit_mf",
    "linucb",
    "popularity",
    "random",
    "implicit_als",
    "implicit_bpr",
    "neural_mf",
    "user_knn",
)


class OrchidRecommender:
    """Convenience wrapper offering a Surprise-like API.

    Parameters
    ----------
    strategy:
        One of ``{"als", "linucb", "popularity", "random", "implicit_als", "implicit_bpr", "neural_mf", "user_knn"}``.
    device:
        Torch device string. Defaults to CPU.
    strategy_kwargs:
        Extra keyword arguments forwarded to the underlying baseline.
    """

    def __init__(
        self,
        strategy: str = "als",
        *,
        device: Optional[str] = None,
        validate_inputs: bool = True,
        **strategy_kwargs,
    ) -> None:
        normalised = strategy.lower()
        if normalised not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Supported strategies: {', '.join(SUPPORTED_STRATEGIES)}"
            )
        self.strategy = normalised
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.strategy_kwargs = strategy_kwargs
        self._validate_inputs = bool(validate_inputs)
        self._baseline = None
        self._user2idx: Dict[int, int] = {}
        self._idx2user: Dict[int, int] = {}
        self._item2idx: Dict[int, int] = {}
        self._idx2item: Dict[int, int] = {}
        self._seen_items: Dict[int, set[int]] = {}
        self._item_features: Optional[np.ndarray] = None
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def _build_mappings(self, interactions: pd.DataFrame, user_col: str, item_col: str) -> None:
        users = interactions[user_col].astype(int).unique()
        items = interactions[item_col].astype(int).unique()
        self._user2idx = {int(uid): idx for idx, uid in enumerate(sorted(users))}
        self._idx2user = {idx: uid for uid, idx in self._user2idx.items()}
        self._item2idx = {int(iid): idx for idx, iid in enumerate(sorted(items))}
        self._idx2item = {idx: iid for iid, idx in self._item2idx.items()}

    def _init_seen_items(self, interactions: pd.DataFrame, user_col: str, item_col: str) -> None:
        self._seen_items = {
            self._user2idx[int(uid)]: {self._item2idx[int(iid)] for iid in grp[item_col].astype(int).values}
            for uid, grp in interactions.groupby(user_col)
            if int(uid) in self._user2idx
        }

    # ------------------------------------------------------------------
    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None,
        item_features: Optional[np.ndarray] = None,
    ) -> "OrchidRecommender":
        """Fit the recommender on implicit or explicit feedback."""
        if interactions.empty:
            raise ValueError("interactions DataFrame is empty")

        if self._validate_inputs:
            required_cols = {user_col, item_col}
            if rating_col is not None:
                required_cols.add(rating_col)
            try:
                validate_interactions_frame(interactions, required_columns=required_cols)
            except ValidationError as exc:
                raise ValueError(str(exc)) from exc

        interactions = interactions.copy()
        interactions[user_col] = interactions[user_col].astype(int)
        interactions[item_col] = interactions[item_col].astype(int)

        if rating_col is None:
            interactions["__label__"] = 1.0
            rating_col = "__label__"
        else:
            interactions[rating_col] = interactions[rating_col].astype(float)

        self._build_mappings(interactions, user_col, item_col)
        self._init_seen_items(interactions, user_col, item_col)

        num_users = len(self._user2idx)
        num_items = len(self._item2idx)
        labels = interactions[rating_col].astype(float).values
        user_idx = interactions[user_col].map(self._user2idx).astype(int).values
        item_idx = interactions[item_col].map(self._item2idx).astype(int).values

        strategy = self.strategy
        if strategy == "als":
            self._baseline = ALSBaseline(num_users, num_items, device=self.device, **self.strategy_kwargs)
            self._baseline.fit(user_idx, item_idx, labels)
        elif strategy == "explicit_mf":
            # Treat provided labels as explicit 1–5 (or real) ratings and optimise MSE
            self._baseline = ExplicitMFBaseline(
                num_users=num_users,
                num_items=num_items,
                device=self.device,
                **self.strategy_kwargs,
            )
            self._baseline.fit(user_idx, item_idx, labels)
        elif strategy == "popularity":
            popularity = interactions.groupby(item_col)[rating_col].mean().to_dict()
            popularity_idx = {self._item2idx[int(iid)]: float(score) for iid, score in popularity.items()}
            self._baseline = PopularityBaseline(popularity_idx, device=self.device)
        elif strategy == "random":
            self._baseline = RandomBaseline(self.device)
        elif strategy == "linucb":
            if item_features is None:
                raise ValueError("item_features must be provided for linucb strategy")
            if self._validate_inputs:
                try:
                    validate_item_features(item_features, num_items)
                except ValidationError as exc:
                    raise ValueError(str(exc)) from exc
            self._item_features = item_features.astype(np.float32)
            self._baseline = LinUCBBaseline(
                alpha=float(self.strategy_kwargs.get("alpha", 1.5)),
                item_features=self._item_features,
                device=self.device,
            )
            # use average rating per item as reward proxy
            reward_dict = interactions.groupby(item_idx)[rating_col].mean().to_dict()
            reward_idx = {int(idx): float(val) for idx, val in reward_dict.items()}
            self._baseline.fit(reward_idx)
        elif strategy == "implicit_als":
            self._baseline = ImplicitALSBaseline(**self.strategy_kwargs)
            self._baseline.fit(
                user_idx,
                item_idx,
                labels,
                num_users=num_users,
                num_items=num_items,
            )
        elif strategy == "implicit_bpr":
            self._baseline = ImplicitBPRBaseline(**self.strategy_kwargs)
            self._baseline.fit(
                user_idx,
                item_idx,
                labels,
                num_users=num_users,
                num_items=num_items,
            )
        elif strategy == "neural_mf":
            self._baseline = NeuralMatrixFactorizationBaseline(
                num_users=num_users,
                num_items=num_items,
                device=self.device,
                **self.strategy_kwargs,
            )
            self._baseline.fit(user_idx, item_idx, labels)
        elif strategy == "user_knn":
            matrix = np.zeros((num_users, num_items), dtype=np.float32)
            counts = np.zeros((num_users, num_items), dtype=np.float32)
            np.add.at(matrix, (user_idx, item_idx), labels.astype(np.float32))
            np.add.at(counts, (user_idx, item_idx), 1.0)
            non_zero = counts > 0
            if np.any(non_zero):
                matrix[non_zero] = matrix[non_zero] / counts[non_zero]
            self._baseline = UserKNNBaseline(
                matrix,
                device=self.device,
                k=int(self.strategy_kwargs.get("k", 20)),
            )
        else:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. Expected one of 'als', 'linucb', 'popularity', "
                "'random', 'implicit_als', 'implicit_bpr', 'neural_mf', 'user_knn'."
            )

        self._logger.info("fitted strategy=%s users=%d items=%d", self.strategy, num_users, num_items)

        return self

    # ------------------------------------------------------------------
    def _scores_for_user(self, user_idx: int, candidate_idx: Sequence[int]) -> torch.Tensor:
        if self._baseline is None:
            raise RuntimeError("Recommender has not been fit yet")

        item_tensor = torch.tensor(candidate_idx, dtype=torch.long, device=self.device)
        kwargs = {"item_ids": item_tensor}
        if hasattr(self._baseline, "user_matrix") or isinstance(self._baseline, ALSBaseline):
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            kwargs["user_ids"] = user_tensor
        logits = self._baseline.infer(**kwargs)
        return logits.squeeze(0)

    # ------------------------------------------------------------------
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the relevance score for a specific (user, item) pair."""
        if user_id not in self._user2idx or item_id not in self._item2idx:
            raise KeyError("Unknown user_id or item_id provided to predict")
        user_idx = self._user2idx[user_id]
        item_idx = self._item2idx[item_id]
        score = self._scores_for_user(user_idx, [item_idx])[0].detach().cpu().item()
        return float(score)

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        *,
        filter_seen: bool = True,
    ) -> List[Recommendation]:
        """Return top-k item recommendations for a user."""
        if user_id not in self._user2idx:
            raise KeyError(f"Unknown user_id {user_id}. Have you called fit?")
        user_idx = self._user2idx[user_id]
        all_items = list(self._item2idx.values())
        if filter_seen and user_idx in self._seen_items:
            seen = self._seen_items[user_idx]
            candidate_idx = [idx for idx in all_items if idx not in seen]
        else:
            candidate_idx = all_items

        if not candidate_idx:
            return []

        scores = self._scores_for_user(user_idx, candidate_idx).detach().cpu().numpy()
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Recommendation(item_id=int(self._idx2item[candidate_idx[i]]), score=float(scores[i]))
            for i in order
        ]

    # ------------------------------------------------------------------
    def all_items(self) -> List[int]:
        return list(self._item2idx.keys())

    def all_users(self) -> List[int]:
        return list(self._user2idx.keys())
