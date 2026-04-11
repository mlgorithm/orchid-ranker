"""High-level Surprise-like recommender interface for Orchid Ranker."""
from __future__ import annotations

__all__ = [
    "Recommendation",
    "OrchidRecommender",
    "SUPPORTED_STRATEGIES",
    "STRATEGY_GUIDE",
]

import difflib
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._compat import require_torch
require_torch("OrchidRecommender")
import torch  # noqa: E402

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
    """Single item recommendation with score.

    Parameters
    ----------
    item_id : int
        Identifier for the recommended item.
    score : float
        Relevance or ranking score for the item.
    """
    item_id: int
    score: float

    def __repr__(self) -> str:
        return f"Recommendation(item_id={self.item_id}, score={self.score:.4f})"


SUPPORTED_STRATEGIES: Tuple[str, ...] = (
    "als",
    "auto",
    "explicit_mf",
    "linucb",
    "popularity",
    "random",
    "implicit_als",
    "implicit_bpr",
    "neural_mf",
    "user_knn",
)

STRATEGY_GUIDE: Dict[str, str] = {
    "als": "Fast alternating least squares. Best for implicit feedback with sparse data.",
    "auto": "Automatic strategy selection. Picks explicit_mf for ratings, als for binary data.",
    "explicit_mf": "SVD-style matrix factorization. Best for explicit ratings (e.g., 1-5 scale).",
    "linucb": "Contextual bandit algorithm. Requires item features. Balances exploration/exploitation.",
    "popularity": "Returns items sorted by popularity. No personalization; good baseline.",
    "random": "Random recommendation. Useful as a sanity-check baseline.",
    "implicit_als": "ALS for implicit feedback. Optimizes for ranking using implicit signals.",
    "implicit_bpr": "Bayesian Personalized Ranking. Optimizes pairwise ranking for implicit feedback.",
    "neural_mf": "Deep neural networks for matrix factorization. Captures non-linear patterns.",
    "user_knn": "User-based collaborative filtering. Recommends items liked by similar users.",
}


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
        """Initialize an OrchidRecommender with a specific strategy.

        Parameters
        ----------
        strategy : str, optional
            Recommendation strategy. One of "als", "explicit_mf", "linucb",
            "popularity", "random", "implicit_als", "implicit_bpr", "neural_mf",
            or "user_knn" (default: "als").
        device : str, optional
            Torch device string (e.g., "cpu", "cuda", "cuda:0").
            If None, automatically selects CUDA if available, else CPU.
        validate_inputs : bool, optional
            Whether to validate input DataFrames during fit (default: True).
        **strategy_kwargs
            Additional keyword arguments passed to the underlying baseline model.
            For example: n_factors, learning_rate, k (for user_knn), alpha (for linucb).

        Raises
        ------
        ValueError
            If strategy is not in the supported strategies list.

        Examples
        --------
        >>> rec = OrchidRecommender(strategy="als")
        >>> rec_knn = OrchidRecommender(strategy="user_knn", k=20)
        >>> rec_linucb = OrchidRecommender(strategy="linucb", alpha=1.5)
        """
        normalised = strategy.lower()
        if normalised not in SUPPORTED_STRATEGIES:
            # Build helpful error message with strategies and descriptions
            strategies_help = "\n".join(
                f"  {s}: {STRATEGY_GUIDE.get(s, 'N/A')}"
                for s in sorted(SUPPORTED_STRATEGIES)
            )
            # Suggest close matches for typos
            suggestions = difflib.get_close_matches(normalised, SUPPORTED_STRATEGIES, n=3, cutoff=0.6)
            suggestion_text = ""
            if suggestions:
                suggestion_text = f"\nDid you mean: {', '.join(suggestions)}?"
            raise ValueError(
                f"Unknown strategy '{strategy}'.{suggestion_text}\n\n"
                f"Supported strategies:\n{strategies_help}"
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
        self._resolved_strategy: Optional[str] = None
        self._logger = logging.getLogger(__name__)

    def __repr__(self) -> str:
        strategy = self._resolved_strategy or self.strategy
        n_users = len(self._user2idx)
        n_items = len(self._item2idx)
        fitted = "fitted" if self._baseline is not None else "not fitted"
        return (f"OrchidRecommender(strategy='{strategy}', {fitted}, "
                f"users={n_users}, items={n_items}, device='{self.device}')")

    @property
    def is_fitted(self) -> bool:
        """Whether the recommender has been fit on data."""
        return self._baseline is not None

    @classmethod
    def quick_fit(
        cls,
        interactions: pd.DataFrame,
        *,
        strategy: str = "auto",
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None,
        **kwargs,
    ) -> "OrchidRecommender":
        """Create and fit a recommender in one call.

        Parameters
        ----------
        interactions : pd.DataFrame
            Interactions DataFrame.
        strategy : str, optional
            Strategy name, or "auto" to detect from data (default: "auto").
        user_col : str, optional
            Column name for user IDs (default: "user_id").
        item_col : str, optional
            Column name for item IDs (default: "item_id").
        rating_col : str, optional
            Column name for ratings. If None, uses implicit feedback.
        **kwargs
            Additional keyword arguments passed to the strategy.

        Returns
        -------
        OrchidRecommender
            Fitted recommender ready for predictions.

        Examples
        --------
        >>> rec = OrchidRecommender.quick_fit(df, rating_col="rating")
        >>> recs = rec.recommend(user_id=1, top_k=5)
        """
        rec = cls(strategy=strategy, **kwargs)
        rec.fit(interactions, user_col=user_col, item_col=item_col, rating_col=rating_col)
        return rec

    @classmethod
    def available_strategies(cls) -> str:
        """Return a formatted table of available strategies and descriptions.

        Returns
        -------
        str
            Formatted table of all supported strategies and their descriptions.

        Examples
        --------
        >>> print(OrchidRecommender.available_strategies())
        Available Recommendation Strategies
        ====================================
        als         | Fast alternating least squares. ...
        """
        lines = ["Available Recommendation Strategies", "=" * 60]
        max_name_len = max(len(s) for s in SUPPORTED_STRATEGIES)
        for strategy in sorted(SUPPORTED_STRATEGIES):
            description = STRATEGY_GUIDE.get(strategy, "N/A")
            lines.append(f"{strategy:<{max_name_len}} | {description}")
        return "\n".join(lines)

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
        """Fit the recommender on implicit or explicit feedback.

        Trains the underlying baseline model on user-item interaction data.
        Supports implicit feedback (presence/absence) or explicit ratings.

        Parameters
        ----------
        interactions : pd.DataFrame
            Interactions DataFrame with at least user_col and item_col.
            If rating_col is provided, it should contain rating/feedback values.
        user_col : str, optional
            Column name for user IDs (default: "user_id").
        item_col : str, optional
            Column name for item IDs (default: "item_id").
        rating_col : str, optional
            Column name for explicit ratings/feedback. If None, treats all
            interactions as implicit feedback with uniform weight 1.0 (default: None).
        item_features : np.ndarray, optional
            Feature matrix of shape (num_items, feature_dim) for linucb strategy.
            Required if strategy=="linucb", ignored otherwise (default: None).

        Returns
        -------
        self : OrchidRecommender
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If interactions DataFrame is empty or missing required columns.
        ValueError
            If linucb strategy is selected but item_features is not provided.

        Examples
        --------
        >>> rec = OrchidRecommender(strategy="als")
        >>> rec.fit(interactions_df)
        >>> rec.fit(interactions_df, rating_col="rating")
        """
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
            # Reject non-finite ratings (inf, nan)
            ratings_arr = interactions[rating_col].values
            if not np.all(np.isfinite(ratings_arr)):
                raise ValueError("ratings contain non-finite values (inf or NaN)")

        self._build_mappings(interactions, user_col, item_col)
        self._init_seen_items(interactions, user_col, item_col)

        num_users = len(self._user2idx)
        num_items = len(self._item2idx)
        labels = interactions[rating_col].astype(float).values
        user_idx = interactions[user_col].map(self._user2idx).astype(int).values
        item_idx = interactions[item_col].map(self._item2idx).astype(int).values

        strategy = self.strategy
        if strategy == "auto":
            # Detect whether data is explicit ratings or binary/implicit
            unique_vals = np.unique(labels)
            is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0})
            if is_binary:
                strategy = "als"
                self._logger.info("auto: detected binary feedback -> als")
            else:
                strategy = "explicit_mf"
                self._logger.info("auto: detected explicit ratings (range %.1f-%.1f) -> explicit_mf",
                                  labels.min(), labels.max())
            self._resolved_strategy = strategy
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
                raise ValueError(
                    "item_features is required for the 'linucb' strategy. "
                    "Pass a np.ndarray of shape (num_items, feature_dim) to fit()."
                )
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

        item_tensor = torch.as_tensor(candidate_idx, dtype=torch.long, device=self.device)
        kwargs = {"item_ids": item_tensor}
        if hasattr(self._baseline, "user_matrix") or isinstance(self._baseline, ALSBaseline):
            user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            kwargs["user_ids"] = user_tensor
        logits = self._baseline.infer(**kwargs)
        return logits.squeeze(0)

    # ------------------------------------------------------------------
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the relevance score for a specific (user, item) pair.

        Parameters
        ----------
        user_id : int
            User identifier (must exist in training data).
        item_id : int
            Item identifier (must exist in training data).

        Returns
        -------
        float
            Predicted relevance score.

        Raises
        ------
        RuntimeError
            If recommender has not been fit.
        KeyError
            If user_id or item_id is unknown.
        """
        if self._baseline is None:
            raise RuntimeError("Recommender has not been fit yet. Call fit() first.")
        if user_id not in self._user2idx:
            raise KeyError(f"Unknown user_id={user_id}. Known users: call all_users() to see valid IDs.")
        if item_id not in self._item2idx:
            raise KeyError(f"Unknown item_id={item_id}. Known items: call all_items() to see valid IDs.")
        user_idx = self._user2idx[user_id]
        item_idx = self._item2idx[item_id]
        score = self._scores_for_user(user_idx, [item_idx])[0].item()
        return float(score)

    def predict_many(self, user_ids: Sequence[int], item_ids: Sequence[int]) -> np.ndarray:
        """Vectorized prediction for matching sequences of user_ids and item_ids.

        Computes relevance scores for (user, item) pairs in parallel.
        More efficient than calling predict() multiple times.

        Parameters
        ----------
        user_ids : Sequence[int]
            User identifiers (length N).
        item_ids : Sequence[int]
            Item identifiers (length N). Must match length of user_ids.

        Returns
        -------
        np.ndarray
            Shape (N,), dtype float32. Predicted scores for each (user, item) pair
            in the same order as input sequences.

        Raises
        ------
        ValueError
            If user_ids and item_ids have different lengths.
        KeyError
            If any user_id or item_id is unknown.
        RuntimeError
            If recommender has not been fit.

        Examples
        --------
        >>> scores = rec.predict_many([1, 1, 2], [5, 6, 5])
        >>> scores.shape
        (3,)
        """
        if len(user_ids) != len(item_ids):
            raise ValueError("user_ids and item_ids must have the same length")
        if len(user_ids) == 0:
            return np.zeros(0, dtype=np.float32)

        try:
            user_idx = np.asarray([self._user2idx[int(u)] for u in user_ids], dtype=np.int64)
            item_idx = np.asarray([self._item2idx[int(i)] for i in item_ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(
                f"Unknown user_id or item_id in predict_many: {exc}. "
                f"Call all_users()/all_items() to see valid IDs."
            ) from exc

        outputs = np.empty(len(user_idx), dtype=np.float32)
        for u in np.unique(user_idx):
            mask = np.where(user_idx == u)[0]
            candidates = item_idx[mask].tolist()
            scores_t = self._scores_for_user(int(u), candidates)
            scores = scores_t.cpu().numpy().astype(np.float32) if scores_t.device.type != 'cpu' else scores_t.detach().numpy().astype(np.float32)
            outputs[mask] = scores
        return outputs

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        *,
        filter_seen: bool = True,
    ) -> List[Recommendation]:
        """Generate top-k item recommendations for a user.

        Parameters
        ----------
        user_id : int
            User identifier (must exist in training data).
        top_k : int, optional
            Number of recommendations to return (default: 10).
        filter_seen : bool, optional
            Whether to exclude items already seen by the user (default: True).

        Returns
        -------
        list of Recommendation
            Top-k recommended items with scores, in descending score order.

        Raises
        ------
        RuntimeError
            If recommender has not been fit.
        KeyError
            If user_id is unknown.
        """
        if self._baseline is None:
            raise RuntimeError("Recommender has not been fit yet. Call fit() first.")
        if user_id not in self._user2idx:
            raise KeyError(f"Unknown user_id {user_id}. Have you called fit?")
        user_idx = self._user2idx[user_id]
        all_items_arr = np.fromiter(self._item2idx.values(), dtype=np.int64, count=len(self._item2idx))
        if filter_seen and user_idx in self._seen_items:
            seen = self._seen_items[user_idx]
            seen_arr = np.fromiter(seen, dtype=np.int64, count=len(seen))
            candidate_idx = np.setdiff1d(all_items_arr, seen_arr).tolist()
        else:
            candidate_idx = all_items_arr.tolist()

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
        """Get all known item IDs.

        Returns list of all item identifiers that were present in the training data
        used to fit the recommender.

        Returns
        -------
        list of int
            All item IDs in the training data, sorted in ascending order.

        Examples
        --------
        >>> items = rec.all_items()
        >>> len(items) > 0
        True
        """
        return sorted(self._item2idx.keys())

    def all_users(self) -> List[int]:
        """Get all known user IDs.

        Returns list of all user identifiers that were present in the training data
        used to fit the recommender.

        Returns
        -------
        list of int
            All user IDs in the training data, sorted in ascending order.

        Examples
        --------
        >>> users = rec.all_users()
        >>> len(users) > 0
        True
        """
        return sorted(self._user2idx.keys())

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the fitted recommender to disk.

        Convenience method that delegates to serialization.save_model().
        Saves the strategy, user/item mappings, and fitted model state.

        Parameters
        ----------
        path : str or Path
            Destination file path for the checkpoint.

        Raises
        ------
        RuntimeError
            If recommender has not been fit yet.
        RuntimeError
            If checkpoint writing fails.

        Examples
        --------
        >>> rec = OrchidRecommender(strategy="als")
        >>> rec.fit(interactions_df)
        >>> rec.save("model.pt")
        """
        from .serialization import save_model
        save_model(self, path)

    @classmethod
    def load(cls, path: str) -> "OrchidRecommender":
        """Load a previously saved OrchidRecommender from disk.

        Convenience classmethod that delegates to serialization.load_model().
        Restores the model in fitted state with all internal mappings and state.

        Parameters
        ----------
        path : str or Path
            Path to the saved checkpoint file.

        Returns
        -------
        OrchidRecommender
            The restored model in fitted state, ready for inference.

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.
        RuntimeError
            If checkpoint loading or restoration fails.

        Examples
        --------
        >>> rec = OrchidRecommender.load("model.pt")
        >>> predictions = rec.predict(user_id=1, item_id=5)
        """
        from .serialization import load_model
        model = load_model(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is {type(model).__name__}, expected OrchidRecommender")
        return model
