"""Hyperparameter tuning utilities for OrchidRecommender."""

from __future__ import annotations

import logging
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from .recommender import OrchidRecommender

from .evaluation import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


class GridSearchCV:
    """Exhaustive search over hyperparameter grid for OrchidRecommender.

    Performs an exhaustive search over all combinations of hyperparameters,
    evaluating each combination using k-fold cross-validation. Tracks the
    best-performing configuration and provides access to detailed results.

    Parameters
    ----------
    strategy : str
        Recommender strategy name. Must be one of the supported strategies
        in OrchidRecommender.SUPPORTED_STRATEGIES.
    param_grid : dict
        Dictionary mapping parameter names (str) to lists of values to try.
        Example: {"factors": [10, 20, 50], "regularization": [0.01, 0.1]}
    cv : int, default=3
        Number of cross-validation folds.
    scoring : str, default="ndcg@10"
        Metric to optimize. Options: "precision@k", "recall@k", "ndcg@k", "map@k".
        Use the format "metric@k" where k is the cut-off rank
        (e.g., "ndcg@10", "precision@5").
    random_state : int, default=42
        Random state for fold splitting and reproducibility.
    verbose : int, default=0
        Verbosity level. Use > 0 for progress logging.

    Attributes
    ----------
    best_params_ : dict
        Dictionary of parameter values that produced the best CV score.
    best_score_ : float
        Mean cross-validation score achieved by best_params_.
    results_ : pd.DataFrame
        DataFrame containing all grid combinations, their parameters,
        and mean CV scores. Columns: param_*, mean_score.
    best_model_ : OrchidRecommender
        Fitted OrchidRecommender using best_params_ on full training data.
    n_iter_ : int
        Number of parameter combinations evaluated.

    Examples
    --------
    >>> param_grid = {
    ...     "factors": [10, 20, 50],
    ...     "regularization": [0.01, 0.1, 1.0]
    ... }
    >>> grid = GridSearchCV(
    ...     strategy="als",
    ...     param_grid=param_grid,
    ...     cv=5,
    ...     scoring="ndcg@10"
    ... )
    >>> grid.fit(interactions_df)
    >>> print(grid.best_params_)
    >>> best_model = grid.best_model()
    >>> recommendations = best_model.recommend(user_id=1)
    """

    def __init__(
        self,
        strategy: str,
        param_grid: Dict[str, List[Any]],
        cv: int = 3,
        scoring: str = "ndcg@10",
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.strategy = strategy
        self.param_grid = param_grid
        self.cv = max(2, int(cv))
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose

        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = -np.inf
        self.results_: Optional[pd.DataFrame] = None
        self.best_model_: Optional[OrchidRecommender] = None
        self.n_iter_: int = 0

        self._validate_scoring(scoring)

    @staticmethod
    def _validate_scoring(scoring: str) -> None:
        """Validate scoring metric format."""
        valid_metrics = {"precision", "recall", "ndcg", "map"}
        if "@" not in scoring:
            raise ValueError(
                f"Scoring format must be 'metric@k' (e.g., 'ndcg@10'), got '{scoring}'"
            )
        metric, k_str = scoring.split("@")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported: {', '.join(valid_metrics)}"
            )
        try:
            int(k_str)
        except ValueError:
            raise ValueError(f"Invalid k value in scoring '{scoring}'")

    def _parse_scoring(self) -> tuple[str, int]:
        """Parse scoring string into (metric, k)."""
        metric, k_str = self.scoring.split("@")
        return metric, int(k_str)

    def _compute_fold_score(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        params: Dict[str, Any],
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None,
    ) -> float:
        """Fit model on train fold and evaluate on test fold."""
        from .recommender import OrchidRecommender
        metric, k = self._parse_scoring()

        try:
            model = OrchidRecommender(strategy=self.strategy, **params)
            model.fit(train_data, user_col=user_col, item_col=item_col, rating_col=rating_col)

            # Compute metric on test set
            scores = []
            test_users = test_data[user_col].unique()

            for user_id in test_users:
                user_test = test_data[test_data[user_col] == user_id]
                relevant = set(user_test[item_col].astype(int).unique())

                if not relevant:
                    continue

                try:
                    recs = model.recommend(int(user_id), top_k=k, filter_seen=False)
                    recommended = [rec.item_id for rec in recs]

                    if metric == "precision":
                        score = precision_at_k(recommended, relevant, k)
                    elif metric == "recall":
                        score = recall_at_k(recommended, relevant, k)
                    elif metric == "ndcg":
                        rel_dict = {item: 1.0 for item in relevant}
                        score = ndcg_at_k(recommended, rel_dict, k)
                    elif metric == "map":
                        score = average_precision(recommended, relevant, k)
                    else:
                        score = 0.0

                    scores.append(score)
                except KeyError:
                    # User not in training data
                    continue

            return float(np.mean(scores)) if scores else 0.0

        except (ValueError, RuntimeError, KeyError) as e:
            if self.verbose > 0:
                logger.warning(f"Error evaluating params {params}: {e}")
            return 0.0

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None,
    ) -> GridSearchCV:
        """Run grid search with k-fold cross-validation.

        Parameters
        ----------
        interactions : pd.DataFrame
            Interactions DataFrame with user_col and item_col columns.
        user_col : str, default="user_id"
            Name of user ID column.
        item_col : str, default="item_id"
            Name of item ID column.
        rating_col : str, optional
            Name of rating/label column. If None, treats as implicit feedback.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If interactions is empty, param_grid is empty, cv < 2, or no combinations generated.
        """
        if interactions.empty:
            raise ValueError("interactions DataFrame is empty")

        if not self.param_grid:
            raise ValueError("param_grid must not be empty")

        if self.cv < 2:
            raise ValueError("cv must be >= 2")

        # Generate all parameter combinations
        param_names = sorted(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        combinations = list(product(*param_values))
        self.n_iter_ = len(combinations)

        if self.n_iter_ == 0:
            raise ValueError("No parameter combinations generated from param_grid")

        if self.verbose > 0:
            logger.info(f"Running grid search with {self.n_iter_} parameter combinations")

        # Setup k-fold splits
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        results = []

        for combo_idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            fold_scores = []

            # Cross-validation loop
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(interactions)):
                train_data = interactions.iloc[train_idx].reset_index(drop=True)
                test_data = interactions.iloc[test_idx].reset_index(drop=True)

                score = self._compute_fold_score(
                    train_data, test_data, params,
                    user_col=user_col,
                    item_col=item_col,
                    rating_col=rating_col,
                )
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))

            if self.verbose > 0:
                logger.info(
                    f"[{combo_idx + 1}/{self.n_iter_}] "
                    f"params={params} mean_score={mean_score:.6f}"
                )

            result_row = {**params, "mean_score": mean_score}
            results.append(result_row)

            # Track best score
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params.copy()

        # Store results as DataFrame
        self.results_ = pd.DataFrame(results)

        if self.verbose > 0:
            logger.info(
                f"Best score: {self.best_score_:.6f} "
                f"with params: {self.best_params_}"
            )

        # Fit final model on full dataset with best params
        from .recommender import OrchidRecommender
        self.best_model_ = OrchidRecommender(strategy=self.strategy, **self.best_params_)
        self.best_model_.fit(
            interactions,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
        )

        return self

    def best_model(self) -> OrchidRecommender:
        """Return the fitted OrchidRecommender with best hyperparameters.

        Returns
        -------
        OrchidRecommender
            Recommender fitted with best_params_ on the full training set.

        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        """
        if self.best_model_ is None:
            raise RuntimeError("GridSearchCV has not been fit yet. Call fit() first.")
        return self.best_model_


class RandomSearchCV:
    """Random search over hyperparameter space for OrchidRecommender.

    Performs random sampling over the hyperparameter space, evaluating each
    sampled configuration using k-fold cross-validation. Useful for large
    search spaces where exhaustive search is computationally expensive.

    Parameters
    ----------
    strategy : str
        Recommender strategy name. Must be one of the supported strategies
        in OrchidRecommender.SUPPORTED_STRATEGIES.
    param_distributions : dict
        Dictionary mapping parameter names (str) to lists of values or
        distributions to sample from. Example: {"factors": [10, 20, 50]}.
    n_iter : int, default=10
        Number of parameter combinations to sample.
    cv : int, default=3
        Number of cross-validation folds.
    scoring : str, default="ndcg@10"
        Metric to optimize. Options: "precision@k", "recall@k", "ndcg@k", "map@k".
        Use the format "metric@k" where k is the cut-off rank.
    random_state : int, default=42
        Random state for sampling and fold splitting.
    verbose : int, default=0
        Verbosity level. Use > 0 for progress logging.

    Attributes
    ----------
    best_params_ : dict
        Dictionary of parameter values that produced the best CV score.
    best_score_ : float
        Mean cross-validation score achieved by best_params_.
    results_ : pd.DataFrame
        DataFrame containing all sampled combinations, their parameters,
        and mean CV scores. Columns: param_*, mean_score.
    best_model_ : OrchidRecommender
        Fitted OrchidRecommender using best_params_ on full training data.
    n_iter_ : int
        Actual number of parameter combinations sampled.

    Examples
    --------
    >>> param_dist = {
    ...     "factors": [10, 20, 30, 50, 100],
    ...     "regularization": [0.001, 0.01, 0.1, 1.0]
    ... }
    >>> random = RandomSearchCV(
    ...     strategy="als",
    ...     param_distributions=param_dist,
    ...     n_iter=20,
    ...     cv=5,
    ...     scoring="ndcg@10"
    ... )
    >>> random.fit(interactions_df)
    >>> print(random.best_params_)
    >>> best_model = random.best_model()
    """

    def __init__(
        self,
        strategy: str,
        param_distributions: Dict[str, List[Any]],
        n_iter: int = 10,
        cv: int = 3,
        scoring: str = "ndcg@10",
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.strategy = strategy
        self.param_distributions = param_distributions
        self.n_iter = max(1, int(n_iter))
        self.cv = max(2, int(cv))
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose

        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = -np.inf
        self.results_: Optional[pd.DataFrame] = None
        self.best_model_: Optional[OrchidRecommender] = None
        self.n_iter_: int = 0

        self._validate_scoring(scoring)
        self._rng = np.random.RandomState(random_state)

    @staticmethod
    def _validate_scoring(scoring: str) -> None:
        """Validate scoring metric format."""
        valid_metrics = {"precision", "recall", "ndcg", "map"}
        if "@" not in scoring:
            raise ValueError(
                f"Scoring format must be 'metric@k' (e.g., 'ndcg@10'), got '{scoring}'"
            )
        metric, k_str = scoring.split("@")
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported: {', '.join(valid_metrics)}"
            )
        try:
            int(k_str)
        except ValueError:
            raise ValueError(f"Invalid k value in scoring '{scoring}'")

    def _parse_scoring(self) -> tuple[str, int]:
        """Parse scoring string into (metric, k)."""
        metric, k_str = self.scoring.split("@")
        return metric, int(k_str)

    def _sample_params(self) -> List[Dict[str, Any]]:
        """Sample n_iter random parameter combinations."""
        param_names = sorted(self.param_distributions.keys())
        sampled = []

        for _ in range(self.n_iter):
            params = {}
            for name in param_names:
                choices = self.param_distributions[name]
                params[name] = self._rng.choice(choices)
            sampled.append(params)

        return sampled

    def _compute_fold_score(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        params: Dict[str, Any],
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None,
    ) -> float:
        """Fit model on train fold and evaluate on test fold."""
        from .recommender import OrchidRecommender
        metric, k = self._parse_scoring()

        try:
            model = OrchidRecommender(strategy=self.strategy, **params)
            model.fit(train_data, user_col=user_col, item_col=item_col, rating_col=rating_col)

            # Compute metric on test set
            scores = []
            test_users = test_data[user_col].unique()

            for user_id in test_users:
                user_test = test_data[test_data[user_col] == user_id]
                relevant = set(user_test[item_col].astype(int).unique())

                if not relevant:
                    continue

                try:
                    recs = model.recommend(int(user_id), top_k=k, filter_seen=False)
                    recommended = [rec.item_id for rec in recs]

                    if metric == "precision":
                        from .evaluation import precision_at_k
                        score = precision_at_k(recommended, relevant, k)
                    elif metric == "recall":
                        from .evaluation import recall_at_k
                        score = recall_at_k(recommended, relevant, k)
                    elif metric == "ndcg":
                        from .evaluation import ndcg_at_k
                        rel_dict = {item: 1.0 for item in relevant}
                        score = ndcg_at_k(recommended, rel_dict, k)
                    elif metric == "map":
                        from .evaluation import average_precision
                        score = average_precision(recommended, relevant, k)
                    else:
                        score = 0.0

                    scores.append(score)
                except KeyError:
                    # User not in training data
                    continue

            return float(np.mean(scores)) if scores else 0.0

        except (ValueError, RuntimeError, KeyError) as e:
            if self.verbose > 0:
                logger.warning(f"Error evaluating params {params}: {e}")
            return 0.0

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: Optional[str] = None,
    ) -> RandomSearchCV:
        """Run random search with k-fold cross-validation.

        Parameters
        ----------
        interactions : pd.DataFrame
            Interactions DataFrame with user_col and item_col columns.
        user_col : str, default="user_id"
            Name of user ID column.
        item_col : str, default="item_id"
            Name of item ID column.
        rating_col : str, optional
            Name of rating/label column. If None, treats as implicit feedback.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If interactions is empty, param_distributions is empty, cv < 2, or no combinations generated.
        """
        if interactions.empty:
            raise ValueError("interactions DataFrame is empty")

        if not self.param_distributions:
            raise ValueError("param_distributions must not be empty")

        if self.cv < 2:
            raise ValueError("cv must be >= 2")

        # Sample parameter combinations
        param_combinations = self._sample_params()
        self.n_iter_ = len(param_combinations)

        if self.n_iter_ == 0:
            raise ValueError("No parameter combinations generated from param_distributions")

        if self.verbose > 0:
            logger.info(f"Running random search with {self.n_iter_} parameter combinations")

        # Setup k-fold splits
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        results = []

        for combo_idx, params in enumerate(param_combinations):
            fold_scores = []

            # Cross-validation loop
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(interactions)):
                train_data = interactions.iloc[train_idx].reset_index(drop=True)
                test_data = interactions.iloc[test_idx].reset_index(drop=True)

                score = self._compute_fold_score(
                    train_data, test_data, params,
                    user_col=user_col,
                    item_col=item_col,
                    rating_col=rating_col,
                )
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))

            if self.verbose > 0:
                logger.info(
                    f"[{combo_idx + 1}/{self.n_iter_}] "
                    f"params={params} mean_score={mean_score:.6f}"
                )

            result_row = {**params, "mean_score": mean_score}
            results.append(result_row)

            # Track best score
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params.copy()

        # Store results as DataFrame
        self.results_ = pd.DataFrame(results)

        if self.verbose > 0:
            logger.info(
                f"Best score: {self.best_score_:.6f} "
                f"with params: {self.best_params_}"
            )

        # Fit final model on full dataset with best params
        from .recommender import OrchidRecommender
        self.best_model_ = OrchidRecommender(strategy=self.strategy, **self.best_params_)
        self.best_model_.fit(
            interactions,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
        )

        return self

    def best_model(self) -> OrchidRecommender:
        """Return the fitted OrchidRecommender with best hyperparameters.

        Returns
        -------
        OrchidRecommender
            Recommender fitted with best_params_ on the full training set.

        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        """
        if self.best_model_ is None:
            raise RuntimeError("RandomSearchCV has not been fit yet. Call fit() first.")
        return self.best_model_


__all__ = [
    "GridSearchCV",
    "RandomSearchCV",
]
