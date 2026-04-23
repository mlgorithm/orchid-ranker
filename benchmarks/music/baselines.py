"""Benchmark baselines for the Last.fm 1K music evaluation.

Each baseline conforms to :class:`BaselineRecommender` so that ``run.py``
can treat them uniformly.  Grid search over hyperparameters is handled by
:func:`grid_search`, evaluated via :func:`compute_ndcg_at_k`.

This module mirrors ``benchmarks/movielens_1m/baselines.py`` exactly in
structure, adapted for purely implicit (listen-event) feedback.

Baselines
---------
1. PopularityBaseline       -- global popularity, no personalization.
2. OrchidFrozenBaseline     -- OrchidRecommender (strategy="als"), fit once.
3. OrchidAdaptiveBaseline   -- OrchidRecommender (strategy="neural_mf") + as_streaming().
4. ImplicitALSBaseline      -- OrchidRecommender (strategy="implicit_als").
5. ImplicitBPRBaseline      -- OrchidRecommender (strategy="implicit_bpr").
"""
from __future__ import annotations

import itertools
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NDCG helper
# ---------------------------------------------------------------------------

def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted cumulative gain for the first *k* entries."""
    relevances = np.asarray(relevances, dtype=np.float64)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2, dtype=np.float64))
    return float(np.sum(relevances / discounts))


def compute_ndcg_at_k(
    recommender: "BaselineRecommender",
    val_df: pd.DataFrame,
    k: int = 10,
    *,
    user_id_to_idx: Optional[Dict] = None,
    item_id_to_idx: Optional[Dict] = None,
) -> float:
    """Mean NDCG@k for *recommender* evaluated against *val_df*.

    For each user present in *val_df*, the recommender is asked for top-k
    recommendations.  A val item is considered relevant when ``label >= 1``
    (for implicit data, all interactions are relevant).

    Parameters
    ----------
    recommender : BaselineRecommender
        A fitted baseline.
    val_df : pd.DataFrame
        Validation interactions with columns ``user_id``, ``item_id``, ``label``.
    k : int
        Cutoff for NDCG computation.
    user_id_to_idx : dict, optional
        Mapping from original user id to internal index.
    item_id_to_idx : dict, optional
        Mapping from original item id to internal index.

    Returns
    -------
    float
        Mean NDCG@k across users.
    """
    # Build per-user relevant item sets from validation data.
    user_relevant: Dict[Any, Set[Any]] = defaultdict(set)
    for _, row in val_df.iterrows():
        if row["label"] >= 1:
            user_relevant[row["user_id"]].add(row["item_id"])

    ndcg_scores: list[float] = []
    for user_id, rel_items in user_relevant.items():
        if not rel_items:
            continue
        user_idx = user_id_to_idx[user_id] if user_id_to_idx else user_id
        try:
            recs = recommender.recommend(user_idx, k=k)
        except (KeyError, IndexError):
            # User not in training set -- skip.
            continue

        # Build binary relevance vector aligned with the recommendation list.
        relevances = np.array(
            [1.0 if item_idx in rel_items else 0.0 for item_idx in recs],
            dtype=np.float64,
        )
        dcg = _dcg_at_k(relevances, k)
        ideal = _dcg_at_k(np.ones(min(len(rel_items), k), dtype=np.float64), k)
        ndcg_scores.append(dcg / ideal if ideal > 0 else 0.0)

    if not ndcg_scores:
        return 0.0
    return float(np.mean(ndcg_scores))


# ---------------------------------------------------------------------------
# Grid search helper
# ---------------------------------------------------------------------------

def grid_search(
    baseline_cls: type,
    param_grid: Dict[str, Sequence[Any]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    item_features: np.ndarray,
    num_users: int,
    num_items: int,
    user_id_to_idx: Dict,
    item_id_to_idx: Dict,
    *,
    k: int = 10,
    fixed_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], "BaselineRecommender"]:
    """Exhaustive grid search over *param_grid* for *baseline_cls*.

    Each combination is instantiated, fitted, and evaluated via NDCG@k on
    the validation set.  The best model and its parameters are returned.

    Returns
    -------
    tuple of (best_params dict, fitted BaselineRecommender)
    """
    fixed_kwargs = fixed_kwargs or {}
    keys = sorted(param_grid.keys())
    values = [param_grid[key] for key in keys]
    combos = list(itertools.product(*values))

    best_ndcg = -1.0
    best_params: Dict[str, Any] = {}
    best_model: Optional[BaselineRecommender] = None

    total = len(combos)
    logger.info(
        "grid_search: %s -- %d combinations over %s",
        baseline_cls.__name__, total, keys,
    )

    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        merged = {**fixed_kwargs, **params}
        t0 = time.perf_counter()
        model = baseline_cls(**merged)
        model.fit(
            train_df, val_df, item_features,
            num_users, num_items,
            user_id_to_idx, item_id_to_idx,
        )
        ndcg = compute_ndcg_at_k(model, val_df, k=k)
        elapsed = time.perf_counter() - t0
        logger.info(
            "  [%d/%d] %s  NDCG@%d=%.4f  (%.1fs)",
            idx, total, params, k, ndcg, elapsed,
        )
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_params = params
            best_model = model

    logger.info(
        "grid_search: best %s params=%s  NDCG@%d=%.4f",
        baseline_cls.__name__, best_params, k, best_ndcg,
    )
    assert best_model is not None, "grid_search produced no model"
    return best_params, best_model


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaselineRecommender(ABC):
    """Common interface for all benchmark recommenders."""

    name: str

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        item_features: np.ndarray,
        num_users: int,
        num_items: int,
        user_id_to_idx: Dict,
        item_id_to_idx: Dict,
    ) -> None:
        """Train on the data.  *val_df* is available for hyperparameter selection."""

    @abstractmethod
    def recommend(
        self,
        user_idx: Any,
        k: int = 10,
        exclude: Optional[Set] = None,
    ) -> List:
        """Return top-k **item IDs** for a user, excluding items in *exclude*."""

    @abstractmethod
    def score(self, user_idx: Any, item_idx: Any) -> float:
        """Return a relevance score for a (user, item) pair."""


# ---------------------------------------------------------------------------
# 1. Popularity baseline
# ---------------------------------------------------------------------------

class PopularityBaseline(BaselineRecommender):
    """Ranks items by training-set popularity (count of listen interactions).

    No personalisation: every user receives the same ranking.
    """

    name = "popularity"

    def __init__(self) -> None:
        self._item_scores: Dict[Any, float] = {}
        self._ranked_items: List = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        item_features: np.ndarray,
        num_users: int,
        num_items: int,
        user_id_to_idx: Dict,
        item_id_to_idx: Dict,
    ) -> None:
        # For implicit data, every row is a positive interaction
        counts = train_df.groupby("item_id").size()
        self._item_scores = counts.to_dict()
        self._ranked_items = sorted(
            self._item_scores.keys(),
            key=lambda iid: self._item_scores[iid],
            reverse=True,
        )
        logger.info(
            "PopularityBaseline fitted: %d items with interactions",
            len(self._ranked_items),
        )

    def recommend(
        self,
        user_idx: Any,
        k: int = 10,
        exclude: Optional[Set] = None,
    ) -> List:
        exclude = exclude or set()
        result: list = []
        for item_id in self._ranked_items:
            if item_id in exclude:
                continue
            result.append(item_id)
            if len(result) >= k:
                break
        return result

    def score(self, user_idx: Any, item_idx: Any) -> float:
        return float(self._item_scores.get(item_idx, 0.0))


# ---------------------------------------------------------------------------
# Helper: wrap OrchidRecommender for the benchmark interface
# ---------------------------------------------------------------------------

def _fit_orchid(
    strategy: str,
    train_df: pd.DataFrame,
    strategy_kwargs: Dict[str, Any],
    seed: int = 42,
) -> Any:
    """Create, fit, and return an ``OrchidRecommender``.

    Uses the *original* user_id / item_id columns -- the recommender
    builds its own internal mappings.  For implicit data, omit rating_col.
    """
    from orchid_ranker.recommender import OrchidRecommender

    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    rec = OrchidRecommender(strategy=strategy, **strategy_kwargs)
    # Implicit feedback -- no rating column
    rec.fit(train_df, user_col="user_id", item_col="item_id")
    return rec


# ---------------------------------------------------------------------------
# 2. OrchidFrozenBaseline (ALS, fit-once)
# ---------------------------------------------------------------------------

class OrchidFrozenBaseline(BaselineRecommender):
    """OrchidRecommender with strategy="als", fit once and frozen.

    Grid search over ``n_factors`` and ``regularization``; best model chosen
    on validation NDCG@10.
    """

    name = "orchid_frozen_als"

    _PARAM_GRID: Dict[str, list] = {
        "n_factors": [32, 64, 128],
        "regularization": [0.01, 0.1],
    }

    def __init__(
        self,
        n_factors: int = 64,
        regularization: float = 0.01,
    ) -> None:
        self.n_factors = n_factors
        self.regularization = regularization
        self._rec: Any = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        item_features: np.ndarray,
        num_users: int,
        num_items: int,
        user_id_to_idx: Dict,
        item_id_to_idx: Dict,
    ) -> None:
        self._rec = _fit_orchid(
            strategy="als",
            train_df=train_df,
            strategy_kwargs={
                "emb_dim": self.n_factors,
                "lr": self.regularization,
            },
        )

    def recommend(
        self,
        user_idx: Any,
        k: int = 10,
        exclude: Optional[Set] = None,
    ) -> List:
        recs = self._rec.recommend(user_idx, top_k=k, filter_seen=True)
        items = [r.item_id for r in recs]
        if exclude:
            items = [i for i in items if i not in exclude]
        return items[:k]

    def score(self, user_idx: Any, item_idx: Any) -> float:
        try:
            return self._rec.predict(user_idx, item_idx)
        except (KeyError, RuntimeError):
            return 0.0


# ---------------------------------------------------------------------------
# 3. OrchidAdaptiveBaseline (neural_mf + streaming adapter)
# ---------------------------------------------------------------------------

class OrchidAdaptiveBaseline(BaselineRecommender):
    """OrchidRecommender with ``strategy="neural_mf"`` wrapped via ``.as_streaming()``.

    The neural_mf strategy exposes a neural tower, which is required by
    ``as_streaming()``.  Grid search over ``lr`` and ``l2`` for the
    streaming adapter.
    """

    name = "orchid_adaptive"

    _PARAM_GRID: Dict[str, list] = {
        "lr": [0.01, 0.05, 0.1],
        "l2": [1e-4, 1e-3, 1e-2],
    }

    def __init__(self, lr: float = 0.05, l2: float = 1e-3) -> None:
        self.lr = lr
        self.l2 = l2
        self._rec: Any = None
        self._streamer: Any = None
        self._user_id_to_idx: Dict = {}
        self._item_id_to_idx: Dict = {}
        self._idx_to_item: Dict = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        item_features: np.ndarray,
        num_users: int,
        num_items: int,
        user_id_to_idx: Dict,
        item_id_to_idx: Dict,
    ) -> None:
        self._user_id_to_idx = dict(user_id_to_idx)
        self._item_id_to_idx = dict(item_id_to_idx)
        self._idx_to_item = {v: k for k, v in item_id_to_idx.items()}

        self._rec = _fit_orchid(
            strategy="neural_mf",
            train_df=train_df,
            strategy_kwargs={"emb_dim": 64, "epochs": 5},
        )
        try:
            self._streamer = self._rec.as_streaming(lr=self.lr, l2=self.l2)
        except RuntimeError:
            logger.warning(
                "as_streaming() failed; falling back to frozen neural_mf "
                "recommendations (no online adaptation)."
            )
            self._streamer = None

    def recommend(
        self,
        user_idx: Any,
        k: int = 10,
        exclude: Optional[Set] = None,
    ) -> List:
        if self._streamer is not None:
            internal_uid = self._rec._user2idx.get(user_idx)
            if internal_uid is None:
                return []
            all_item_indices = list(self._rec._item2idx.values())
            if exclude:
                exclude_internal = {
                    self._rec._item2idx[eid]
                    for eid in exclude
                    if eid in self._rec._item2idx
                }
                all_item_indices = [
                    i for i in all_item_indices if i not in exclude_internal
                ]
            ranked = self._streamer.rank(
                internal_uid, all_item_indices, top_k=k,
            )
            return [self._rec._idx2item[iid] for iid, _ in ranked]
        # Fallback: frozen OrchidRecommender.
        recs = self._rec.recommend(user_idx, top_k=k, filter_seen=True)
        items = [r.item_id for r in recs]
        if exclude:
            items = [i for i in items if i not in exclude]
        return items[:k]

    def score(self, user_idx: Any, item_idx: Any) -> float:
        try:
            return self._rec.predict(user_idx, item_idx)
        except (KeyError, RuntimeError):
            return 0.0


# ---------------------------------------------------------------------------
# 4. ImplicitALS baseline
# ---------------------------------------------------------------------------

class ImplicitALSBenchmarkBaseline(BaselineRecommender):
    """OrchidRecommender with strategy="implicit_als".

    Grid search over ``factors``, ``regularization``, and ``iterations``.
    """

    name = "implicit_als"

    _PARAM_GRID: Dict[str, list] = {
        "factors": [64, 128],
        "regularization": [0.01, 0.1],
        "iterations": [15, 30],
    }

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self._rec: Any = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        item_features: np.ndarray,
        num_users: int,
        num_items: int,
        user_id_to_idx: Dict,
        item_id_to_idx: Dict,
    ) -> None:
        self._rec = _fit_orchid(
            strategy="implicit_als",
            train_df=train_df,
            strategy_kwargs={
                "factors": self.factors,
                "regularization": self.regularization,
                "iterations": self.iterations,
            },
        )

    def recommend(
        self,
        user_idx: Any,
        k: int = 10,
        exclude: Optional[Set] = None,
    ) -> List:
        recs = self._rec.recommend(user_idx, top_k=k, filter_seen=True)
        items = [r.item_id for r in recs]
        if exclude:
            items = [i for i in items if i not in exclude]
        return items[:k]

    def score(self, user_idx: Any, item_idx: Any) -> float:
        try:
            return self._rec.predict(user_idx, item_idx)
        except (KeyError, RuntimeError):
            return 0.0


# ---------------------------------------------------------------------------
# 5. ImplicitBPR baseline
# ---------------------------------------------------------------------------

class ImplicitBPRBenchmarkBaseline(BaselineRecommender):
    """OrchidRecommender with strategy="implicit_bpr".

    Grid search over ``factors`` and ``learning_rate``.
    """

    name = "implicit_bpr"

    _PARAM_GRID: Dict[str, list] = {
        "factors": [64, 128],
        "learning_rate": [0.01, 0.05],
    }

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
    ) -> None:
        self.factors = factors
        self.learning_rate = learning_rate
        self._rec: Any = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        item_features: np.ndarray,
        num_users: int,
        num_items: int,
        user_id_to_idx: Dict,
        item_id_to_idx: Dict,
    ) -> None:
        self._rec = _fit_orchid(
            strategy="implicit_bpr",
            train_df=train_df,
            strategy_kwargs={
                "factors": self.factors,
                "learning_rate": self.learning_rate,
            },
        )

    def recommend(
        self,
        user_idx: Any,
        k: int = 10,
        exclude: Optional[Set] = None,
    ) -> List:
        recs = self._rec.recommend(user_idx, top_k=k, filter_seen=True)
        items = [r.item_id for r in recs]
        if exclude:
            items = [i for i in items if i not in exclude]
        return items[:k]

    def score(self, user_idx: Any, item_idx: Any) -> float:
        try:
            return self._rec.predict(user_idx, item_idx)
        except (KeyError, RuntimeError):
            return 0.0


# ---------------------------------------------------------------------------
# Registry -- convenience list used by run.py
# ---------------------------------------------------------------------------

ALL_BASELINES: List[type] = [
    PopularityBaseline,
    OrchidFrozenBaseline,
    OrchidAdaptiveBaseline,
    ImplicitALSBenchmarkBaseline,
    ImplicitBPRBenchmarkBaseline,
]
"""All baseline classes in evaluation order."""
