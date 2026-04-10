"""Model selection and cross-validation utilities for Orchid Ranker."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision,
)
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train and test sets.

    Supports both per-user stratified splits and global random splits.

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

    interactions = interactions.copy()
    rng = np.random.RandomState(random_state)

    if by_user:
        # Hold out test_size fraction per user
        train_idx = []
        test_idx = []
        for user_id, group in interactions.groupby(user_col):
            user_indices = group.index.tolist()
            n_test = max(1, int(len(user_indices) * test_size))
            test_indices = rng.choice(
                user_indices, size=n_test, replace=False
            ).tolist()
            train_indices = [i for i in user_indices if i not in test_indices]
            train_idx.extend(train_indices)
            test_idx.extend(test_indices)
        train_df = interactions.loc[train_idx].reset_index(drop=True)
        test_df = interactions.loc[test_idx].reset_index(drop=True)
    else:
        # Global random split
        n_test = int(len(interactions) * test_size)
        test_idx = rng.choice(
            len(interactions), size=n_test, replace=False
        )
        train_idx = np.setdiff1d(
            np.arange(len(interactions)), test_idx
        )
        train_df = interactions.iloc[train_idx].reset_index(drop=True)
        test_df = interactions.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df


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
) -> Dict[str, Dict[str, float]]:
    """Perform k-fold cross-validation for a recommender strategy.

    Splits data by user (each fold holds out a fraction of each user's
    interactions) and returns mean and std of metrics across folds.

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
    rng = np.random.RandomState(random_state)
    n_unique_users = interactions[user_col].nunique()

    if k > n_unique_users:
        logger.warning(
            f"k={k} exceeds number of unique users ({n_unique_users}). "
            f"Clamping k to {n_unique_users}."
        )
        k = n_unique_users

    # Create fold splits (stratified by user)
    fold_data = []
    users = interactions[user_col].unique()
    fold_size = max(1, len(users) // k)

    for fold_idx in range(k):
        start_idx = fold_idx * fold_size
        end_idx = start_idx + fold_size if fold_idx < k - 1 else len(users)
        test_users = set(users[start_idx:end_idx])

        test_mask = interactions[user_col].isin(test_users)
        train_df = interactions[~test_mask].reset_index(drop=True)
        test_df = interactions[test_mask].reset_index(drop=True)

        fold_data.append((train_df, test_df))

    # Run cross-validation
    fold_results = {metric: [] for metric in metrics}

    for fold_idx, (train_df, test_df) in enumerate(fold_data):
        if train_df.empty or test_df.empty:
            continue

        # Fit model on training fold
        model = OrchidRecommender(strategy=strategy, **strategy_kwargs)
        model.fit(train_df, user_col=user_col, item_col=item_col)

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
            recs = model.recommend(user_id, top_k=k, filter_seen=False)
            recommendations[user_id] = [rec.item_id for rec in recs]
        except (KeyError, RuntimeError):
            # User or model not available
            recommendations[user_id] = []

    # Compute metrics
    scores = {}
    metric_funcs = {
        "precision@5": (precision_at_k, 5),
        "recall@5": (recall_at_k, 5),
        "ndcg@10": (ndcg_at_k, 10),
        "map@10": (average_precision, 10),
    }

    for metric_name in metrics:
        if metric_name not in metric_funcs:
            continue

        func, default_k = metric_funcs[metric_name]
        metric_scores = []

        for user_id, relevant in relevant_items.items():
            recommended = recommendations.get(user_id, [])

            if metric_name in ("ndcg@10", "map@10"):
                # These need full relevance dicts or sets
                if metric_name == "ndcg@10":
                    rel_dict = {item: 1.0 for item in relevant}
                    score = func(recommended, rel_dict, default_k)
                else:
                    score = func(recommended, relevant, default_k)
            else:
                score = func(recommended, relevant, default_k)

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
    strategy_kwargs_list: Optional[List[Dict]] = None,
    random_state: Optional[int] = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
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
    strategy_kwargs_list : list of dict, optional
        List of kwargs dicts (one per strategy). If shorter than strategies,
        remaining strategies use empty dicts (default: None).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    user_col : str, optional
        Column name for user IDs (default: "user_id").
    item_col : str, optional
        Column name for item IDs (default: "item_id").

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

    if strategy_kwargs_list is None:
        strategy_kwargs_list = [{} for _ in strategies]
    else:
        # Pad with empty dicts if needed
        while len(strategy_kwargs_list) < len(strategies):
            strategy_kwargs_list.append({})

    results_by_strategy = {}

    for strategy, kwargs in zip(strategies, strategy_kwargs_list):
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
    "train_test_split",
    "cross_validate",
    "evaluate_on_holdout",
    "compare_models",
]
