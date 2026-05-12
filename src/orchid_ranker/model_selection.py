"""Model selection and cross-validation utilities for Orchid Ranker."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .evaluation import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

#: Supported metric names for cross_validate, evaluate_on_holdout, compare_models.
EVALUATION_METRICS = frozenset({"precision@5", "recall@5", "ndcg@10", "map@10"})

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .recommender import OrchidRecommender

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data splitting utilities
# ---------------------------------------------------------------------------


def train_test_split(
    interactions: pd.DataFrame,
    test_size: float = 0.2,
    by_user: bool = True,
    random_state: Optional[int] = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: Optional[str] = "timestamp",
    chronological: Optional[bool] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train and test sets.

    Supports per-user stratified splits, chronological per-user splits when
    timestamp data is present, and global random splits.

    Parameters
    ----------
    interactions : pd.DataFrame
        Interactions DataFrame with at least user_col and item_col.
    test_size : float, optional
        Fraction of data to use for testing (default: 0.2).
        When by_user=True, this fraction applies per user.
        When by_user=False, this is the global fraction.
    by_user : bool, optional
        If True, hold out test_size fraction of each user's interactions.
        If False, perform a global random split (default: True).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    user_col : str, optional
        Column name for user IDs (default: "user_id").
    item_col : str, optional
        Column name for item IDs (default: "item_id").
    time_col : str, optional
        Timestamp column used for chronological splits. If left as the default
        and a ``timestamp`` column exists, by-user splits are chronological.
    chronological : bool, optional
        Force or disable chronological splitting. ``None`` auto-enables it for
        by-user splits when ``time_col`` exists.

    Returns
    -------
    train_df : pd.DataFrame
        Training interactions.
    test_df : pd.DataFrame
        Test interactions.

    Raises
    ------
    ValueError
        If test_size not in (0, 1) or interactions is empty.

    Examples
    --------
    >>> train, test = train_test_split(interactions, test_size=0.2, by_user=True)
    >>> print(f"Train: {len(train)}, Test: {len(test)}")
    """
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be in (0, 1)")

    if interactions.empty:
        raise ValueError("interactions DataFrame is empty")

    # Validate required columns exist
    missing_cols = []
    if user_col not in interactions.columns:
        missing_cols.append(user_col)
    if item_col not in interactions.columns:
        missing_cols.append(item_col)
    if missing_cols:
        raise ValueError(
            f"Column(s) {missing_cols} not found in interactions DataFrame. "
            f"Available columns: {list(interactions.columns)}"
        )

    interactions = interactions.copy()
    if chronological is None:
        chronological = bool(by_user and time_col is not None and time_col in interactions.columns)
    if chronological and time_col is None:
        raise ValueError("chronological split requires time_col")
    if chronological and time_col not in interactions.columns:
        raise ValueError(
            f"time_col '{time_col}' not found in interactions DataFrame. "
            f"Available columns: {list(interactions.columns)}"
        )

    rng = np.random.RandomState(random_state)

    if by_user:
        if chronological:
            return chronological_user_split(
                interactions,
                test_size=test_size,
                user_col=user_col,
                time_col=str(time_col),
            )
        # Hold out test_size fraction per user
        train_idx: List[int] = []
        test_idx: List[int] = []
        for user_id, group in interactions.groupby(user_col):
            user_indices = group.index.tolist()
            if len(user_indices) <= 1:
                train_idx.extend(user_indices)
                continue
            n_test = min(len(user_indices) - 1, max(1, int(len(user_indices) * test_size)))
            test_indices = rng.choice(
                user_indices, size=n_test, replace=False
            ).tolist()
            train_indices = [i for i in user_indices if i not in test_indices]
            train_idx.extend(train_indices)
            test_idx.extend(test_indices)
        train_df = interactions.loc[train_idx].reset_index(drop=True)
        test_df = interactions.loc[test_idx].reset_index(drop=True)
    else:
        if chronological:
            ordered = interactions.sort_values(str(time_col), kind="mergesort")
            n_test = max(1, int(len(ordered) * test_size))
            split_at = max(1, len(ordered) - n_test)
            train_df = ordered.iloc[:split_at].reset_index(drop=True)
            test_df = ordered.iloc[split_at:].reset_index(drop=True)
            return train_df, test_df
        # Global random split
        n_test = int(len(interactions) * test_size)
        test_idx_arr = rng.choice(
            len(interactions), size=n_test, replace=False
        )
        train_idx_arr = np.setdiff1d(
            np.arange(len(interactions)), test_idx_arr
        )
        train_df = interactions.iloc[train_idx_arr].reset_index(drop=True)
        test_df = interactions.iloc[test_idx_arr].reset_index(drop=True)

    return train_df, test_df


def chronological_user_split(
    interactions: pd.DataFrame,
    *,
    test_size: float = 0.2,
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's history by time, holding out only future events.

    Unlike a random within-user split, this guarantees that no training event
    for a user occurs after that user's test events. Single-interaction users
    remain in train so downstream recommenders do not face impossible cold
    starts by default.
    """
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be in (0, 1)")
    if interactions.empty:
        raise ValueError("interactions DataFrame is empty")
    missing_cols = [col for col in (user_col, time_col) if col not in interactions.columns]
    if missing_cols:
        raise ValueError(
            f"Column(s) {missing_cols} not found in interactions DataFrame. "
            f"Available columns: {list(interactions.columns)}"
        )

    ordered = interactions.sort_values([user_col, time_col], kind="mergesort")
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in ordered.groupby(user_col, sort=False):
        if len(group) <= 1:
            train_parts.append(group)
            continue
        n_test = min(len(group) - 1, max(1, int(len(group) * test_size)))
        split_at = len(group) - n_test
        train_parts.append(group.iloc[:split_at])
        test_parts.append(group.iloc[split_at:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ordered.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ordered.iloc[0:0].copy()
    return train_df, test_df


def _resolve_rating_col(
    interactions: pd.DataFrame,
    rating_col: Optional[str],
) -> Optional[str]:
    """Resolve the rating column for training/evaluation helpers."""
    if rating_col is not None:
        if rating_col not in interactions.columns:
            raise ValueError(
                f"rating_col '{rating_col}' not found in interactions DataFrame. "
                f"Available columns: {list(interactions.columns)}"
            )
        return rating_col
    return "rating" if "rating" in interactions.columns else None


def _build_user_stratified_folds(
    interactions: pd.DataFrame,
    *,
    k: int,
    random_state: Optional[int],
    user_col: str,
    time_col: Optional[str] = "timestamp",
    chronological: Optional[bool] = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Build CV folds by holding out interactions within each user.

    This keeps evaluated users present in the training set and avoids
    impossible cold-start folds for recommenders that cannot score unseen users.
    Users with fewer than two interactions are retained in train for every fold.
    """
    if chronological is None:
        chronological = bool(time_col is not None and time_col in interactions.columns)
    if chronological:
        if time_col is None or time_col not in interactions.columns:
            raise ValueError(
                f"time_col '{time_col}' not found in interactions DataFrame. "
                f"Available columns: {list(interactions.columns)}"
            )
        return _build_chronological_user_folds(
            interactions,
            k=k,
            user_col=user_col,
            time_col=str(time_col),
        )

    rng = np.random.RandomState(random_state)
    fold_test_idx: List[List[int]] = [[] for _ in range(k)]

    for _, group in interactions.groupby(user_col):
        user_indices = group.index.to_numpy(dtype=np.int64, copy=True)
        if user_indices.size <= 1:
            continue
        rng.shuffle(user_indices)
        n_user_folds = min(k, user_indices.size)
        for pos, idx in enumerate(user_indices):
            fold_test_idx[pos % n_user_folds].append(int(idx))

    all_indices = set(int(i) for i in interactions.index.tolist())
    fold_data: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for test_idx in fold_test_idx:
        if not test_idx:
            continue
        test_idx_set = set(test_idx)
        train_idx = sorted(all_indices - test_idx_set)
        test_idx_sorted = sorted(test_idx_set)
        train_df = interactions.loc[train_idx].reset_index(drop=True)
        test_df = interactions.loc[test_idx_sorted].reset_index(drop=True)
        if not train_df.empty and not test_df.empty:
            fold_data.append((train_df, test_df))

    return fold_data


def _build_chronological_user_folds(
    interactions: pd.DataFrame,
    *,
    k: int,
    user_col: str,
    time_col: str,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Build leak-free sequential folds with per-user train-before-test cutoffs."""
    ordered = interactions.sort_values([user_col, time_col], kind="mergesort")
    all_train: list[list[int]] = [[] for _ in range(k)]
    all_test: list[list[int]] = [[] for _ in range(k)]

    for _, group in ordered.groupby(user_col, sort=False):
        user_indices = group.index.to_numpy(dtype=np.int64, copy=True)
        n = user_indices.size
        if n <= 1:
            for train_idx in all_train:
                train_idx.extend(int(i) for i in user_indices)
            continue
        n_user_folds = min(k, n - 1)
        cutoffs = np.linspace(1, n, num=n_user_folds + 1, dtype=int)
        for fold_id in range(n_user_folds):
            start = int(cutoffs[fold_id])
            end = int(cutoffs[fold_id + 1])
            if end <= start:
                end = min(n, start + 1)
            train_part = user_indices[:start]
            test_part = user_indices[start:end]
            all_train[fold_id].extend(int(i) for i in train_part)
            all_test[fold_id].extend(int(i) for i in test_part)

    fold_data: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_idx, test_idx in zip(all_train, all_test):
        if not train_idx or not test_idx:
            continue
        train_df = interactions.loc[sorted(set(train_idx))].reset_index(drop=True)
        test_df = interactions.loc[sorted(set(test_idx))].reset_index(drop=True)
        if not train_df.empty and not test_df.empty:
            fold_data.append((train_df, test_df))
    return fold_data


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    interactions: pd.DataFrame,
    strategy: str,
    k: int = 5,
    metrics: Optional[List[str]] = None,
    strategy_kwargs: Optional[Dict] = None,
    random_state: Optional[int] = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: Optional[str] = None,
    time_col: Optional[str] = "timestamp",
    chronological: Optional[bool] = None,
) -> Dict[str, Dict[str, float]]:
    """Perform k-fold cross-validation for a recommender strategy.

    Splits each user's interactions across folds so evaluated users remain
    present in training, then returns mean and std of metrics across folds.

    Parameters
    ----------
    interactions : pd.DataFrame
        Interactions DataFrame with at least user_col and item_col.
    strategy : str
        Recommender strategy name (e.g., "als", "random", "popularity").
    k : int, optional
        Number of folds (default: 5).
    metrics : list of str, optional
        Metric names to compute. Options: "precision@5", "recall@5", "ndcg@10",
        "map@10". If None, defaults to all (default: None).
    strategy_kwargs : dict, optional
        Keyword arguments passed to OrchidRecommender (default: None).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    user_col : str, optional
        Column name for user IDs (default: "user_id").
    item_col : str, optional
        Column name for item IDs (default: "item_id").
    rating_col : str, optional
        Column name for explicit ratings/feedback. If None and a ``rating``
        column exists, it is used automatically.
    time_col : str, optional
        Timestamp column used for chronological folds. If left as the default
        and a ``timestamp`` column exists, folds are chronological and
        train-before-test per user.
    chronological : bool, optional
        Force or disable chronological folds. ``None`` auto-enables them when
        ``time_col`` exists.

    Returns
    -------
    results : dict
        Dictionary with structure {metric_name: {"mean": float, "std": float}}.

    Raises
    ------
    ValueError
        If k < 2, interactions is empty, or unknown metric provided.

    Examples
    --------
    >>> results = cross_validate(interactions, "als", k=5)
    >>> print(f"Precision@5: {results['precision@5']['mean']:.4f} ± "
    ...       f"{results['precision@5']['std']:.4f}")
    """
    if interactions.empty:
        raise ValueError("interactions DataFrame is empty")

    if k < 2:
        raise ValueError("k must be >= 2")

    if metrics is None:
        metrics = ["precision@5", "recall@5", "ndcg@10", "map@10"]

    valid_metrics = {"precision@5", "recall@5", "ndcg@10", "map@10"}
    for metric in metrics:
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported: {valid_metrics}"
            )

    strategy_kwargs = strategy_kwargs or {}
    resolved_rating_col = _resolve_rating_col(interactions, rating_col)
    eligible_users = int((interactions.groupby(user_col).size() >= 2).sum())
    if eligible_users == 0:
        raise ValueError(
            "cross_validate requires at least one user with two or more interactions"
        )

    if k > len(interactions):
        logger.warning(
            "k=%d exceeds number of interactions (%d). Clamping k to %d.",
            k, len(interactions), len(interactions),
        )
        k = len(interactions)

    fold_data = _build_user_stratified_folds(
        interactions,
        k=k,
        random_state=random_state,
        user_col=user_col,
        time_col=time_col,
        chronological=chronological,
    )
    if not fold_data:
        raise ValueError("Unable to build non-empty cross-validation folds from the provided interactions")

    # Run cross-validation
    fold_results: Dict[str, List[float]] = {metric: [] for metric in metrics}

    for fold_idx, (train_df, test_df) in enumerate(fold_data):
        if train_df.empty or test_df.empty:
            continue

        # Fit model on training fold
        from .recommender import OrchidRecommender
        model = OrchidRecommender(strategy=strategy, **strategy_kwargs)
        model.fit(
            train_df,
            user_col=user_col,
            item_col=item_col,
            rating_col=resolved_rating_col,
        )

        # Evaluate on test fold
        fold_scores = evaluate_on_holdout(
            model,
            test_df,
            metrics=metrics,
            k=10,
            user_col=user_col,
            item_col=item_col,
        )

        for metric, score in fold_scores.items():
            fold_results[metric].append(score)

    # Compute mean and std
    results = {}
    for metric in metrics:
        scores = fold_results[metric]
        if scores:
            results[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
        else:
            results[metric] = {"mean": 0.0, "std": 0.0}

    return results


# ---------------------------------------------------------------------------
# Evaluation on holdout data
# ---------------------------------------------------------------------------


def evaluate_on_holdout(
    model: OrchidRecommender,
    test_interactions: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    k: int = 10,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> Dict[str, float]:
    """Evaluate a fitted recommender model on held-out test data.

    Generates top-k recommendations for each user in the test set and
    computes ranking metrics against ground-truth test interactions.

    Parameters
    ----------
    model : OrchidRecommender
        A fitted recommender model.
    test_interactions : pd.DataFrame
        Test interactions DataFrame with user_col and item_col.
        Contains ground-truth relevant items for each user.
    metrics : list of str, optional
        Metric names to compute. Options: "precision@5", "recall@5", "ndcg@10",
        "map@10". If None, defaults to all (default: None).
    k : int, optional
        Number of recommendations to generate per user (default: 10).
    user_col : str, optional
        Column name for user IDs (default: "user_id").
    item_col : str, optional
        Column name for item IDs (default: "item_id").

    Returns
    -------
    scores : dict
        Dictionary mapping metric names to computed mean scores
        (averaged across all users).

    Raises
    ------
    ValueError
        If unknown metric is requested.

    Examples
    --------
    >>> model = OrchidRecommender("als")
    >>> model.fit(train_df)
    >>> scores = evaluate_on_holdout(model, test_df, k=10)
    >>> print(f"Precision@5: {scores['precision@5']:.4f}")
    """
    if metrics is None:
        metrics = ["precision@5", "recall@5", "ndcg@10", "map@10"]

    valid_metrics = {"precision@5", "recall@5", "ndcg@10", "map@10"}
    for metric in metrics:
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported: {valid_metrics}"
            )

    if test_interactions.empty:
        return {metric: 0.0 for metric in metrics}

    # Build ground-truth relevant items per user
    relevant_items = {}
    for user_id, group in test_interactions.groupby(user_col):
        relevant_items[int(user_id)] = set(
            int(iid) for iid in group[item_col].values
        )

    # Generate recommendations for users with test data
    recommendations = {}
    for user_id in relevant_items.keys():
        try:
            recs = model.recommend(user_id, top_k=k, filter_seen=True)
            recommendations[user_id] = [rec.item_id for rec in recs]
        except (KeyError, RuntimeError):
            # User or model not available
            recommendations[user_id] = []

    # Compute metrics
    scores = {}
    for metric_name in metrics:
        metric_scores = []

        for user_id, relevant in relevant_items.items():
            recommended = recommendations.get(user_id, [])
            if metric_name == "precision@5":
                score = precision_at_k(recommended, relevant, 5)
            elif metric_name == "recall@5":
                score = recall_at_k(recommended, relevant, 5)
            elif metric_name == "ndcg@10":
                rel_dict = {item: 1.0 for item in relevant}
                score = ndcg_at_k(recommended, rel_dict, 10)
            elif metric_name == "map@10":
                score = average_precision(recommended, relevant, 10)
            else:
                continue

            metric_scores.append(score)

        scores[metric_name] = (
            float(np.mean(metric_scores)) if metric_scores else 0.0
        )

    return scores


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_models(
    interactions: pd.DataFrame,
    strategies: Sequence[str],
    k: int = 5,
    metrics: Optional[List[str]] = None,
    strategy_configs: Optional[List[Dict]] = None,
    random_state: Optional[int] = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compare multiple recommender strategies via cross-validation.

    Runs k-fold cross-validation for each strategy and returns a DataFrame
    with rows=strategies and columns=metrics. Useful for model selection
    and benchmarking different approaches.

    Parameters
    ----------
    interactions : pd.DataFrame
        Interactions DataFrame with at least user_col and item_col.
    strategies : sequence of str
        List of strategy names to compare (e.g., ["als", "random", "popularity"]).
    k : int, optional
        Number of cross-validation folds (default: 5).
    metrics : list of str, optional
        Metric names to compute. Options: "precision@5", "recall@5", "ndcg@10",
        "map@10". If None, defaults to all (default: None).
    strategy_configs : list of dict, optional
        One kwargs dict per strategy. If shorter than strategies list,
        remaining strategies use empty dicts (default: None).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    user_col : str, optional
        Column name for user IDs (default: "user_id").
    item_col : str, optional
        Column name for item IDs (default: "item_id").
    rating_col : str, optional
        Column name for explicit ratings/feedback. If None and a ``rating``
        column exists, it is used automatically.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with strategies as row indices and metrics as columns.
        Each cell contains "mean ± std" formatted string, e.g., "0.45 ± 0.05".

    Raises
    ------
    ValueError
        If interactions is empty or k < 2.

    Examples
    --------
    >>> strategies = ["als", "random", "popularity"]
    >>> df = compare_models(interactions, strategies, k=5)
    >>> print(df)
                precision@5  recall@5     ndcg@10     map@10
    als         0.45 ± 0.05 0.32 ± 0.04 0.61 ± 0.06 0.54 ± 0.05
    random      0.10 ± 0.01 0.07 ± 0.01 0.12 ± 0.01 0.08 ± 0.01
    popularity  0.35 ± 0.04 0.25 ± 0.03 0.48 ± 0.05 0.42 ± 0.04
    """
    if metrics is None:
        metrics = ["precision@5", "recall@5", "ndcg@10", "map@10"]

    if strategy_configs is None:
        strategy_configs = [{} for _ in strategies]
    else:
        # Pad with empty dicts if needed
        strategy_configs = list(strategy_configs)
        while len(strategy_configs) < len(strategies):
            strategy_configs.append({})

    results_by_strategy = {}

    for strategy, kwargs in zip(strategies, strategy_configs):
        logger.info(f"Cross-validating strategy: {strategy}")
        cv_results = cross_validate(
            interactions,
            strategy=strategy,
            k=k,
            metrics=metrics,
            strategy_kwargs=kwargs,
            random_state=random_state,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
        )
        results_by_strategy[strategy] = cv_results

    # Format results into DataFrame
    data = {}
    for strategy, cv_results in results_by_strategy.items():
        row = {}
        for metric in metrics:
            if metric in cv_results:
                mean = cv_results[metric]["mean"]
                std = cv_results[metric]["std"]
                row[metric] = f"{mean:.4f} ± {std:.4f}"
            else:
                row[metric] = "N/A"
        data[strategy] = row

    results_df = pd.DataFrame.from_dict(data, orient="index")
    results_df = results_df[metrics]  # Ensure column order

    # Log summary
    logger.info("=" * 80)
    logger.info("Model Comparison Summary")
    logger.info("=" * 80)
    logger.info("\n" + results_df.to_string())
    logger.info("=" * 80)

    return results_df


__all__ = [
    "EVALUATION_METRICS",
    "chronological_user_split",
    "train_test_split",
    "cross_validate",
    "evaluate_on_holdout",
    "compare_models",
]
