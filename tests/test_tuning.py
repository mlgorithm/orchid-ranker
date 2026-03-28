"""Comprehensive tests for GridSearchCV and RandomSearchCV tuning utilities."""

import pytest
import pandas as pd
import numpy as np

from orchid_ranker.tuning import GridSearchCV, RandomSearchCV


# ============================================================================
# Fixtures for test data
# ============================================================================

@pytest.fixture
def small_interactions():
    """Create a small synthetic interactions DataFrame for testing."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "item_id": [10, 20, 30, 10, 20, 40, 20, 30, 50, 10, 40, 50, 30, 40, 50],
    })


# ============================================================================
# GridSearchCV Tests
# ============================================================================

class TestGridSearchCVConstruction:
    """Tests for GridSearchCV __init__ and parameter validation."""

    def test_gridsearchcv_construction_valid_params(self):
        """Test GridSearchCV constructs with valid parameters."""
        param_grid = {"dummy": [1, 2]}
        grid = GridSearchCV(
            strategy="popularity",
            param_grid=param_grid,
            cv=3,
            scoring="ndcg@10",
        )
        assert grid.strategy == "popularity"
        assert grid.param_grid == param_grid
        assert grid.cv == 3
        assert grid.scoring == "ndcg@10"

    def test_gridsearchcv_construction_defaults(self):
        """Test GridSearchCV uses correct defaults."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1]},
        )
        assert grid.cv == 3
        assert grid.scoring == "ndcg@10"
        assert grid.random_state == 42
        assert grid.verbose == 0

    def test_gridsearchcv_construction_custom_scoring(self):
        """Test GridSearchCV with different scoring metrics."""
        for metric in ["precision@5", "recall@10", "map@10"]:
            grid = GridSearchCV(
                strategy="popularity",
                param_grid={"dummy": [1]},
                scoring=metric,
            )
            assert grid.scoring == metric

    def test_gridsearchcv_cv_minimum_enforced(self):
        """Test that cv is forced to at least 2."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1]},
            cv=1,  # Should be raised to 2
        )
        assert grid.cv == 2


# ============================================================================
# GridSearchCV Validation Tests
# ============================================================================

class TestGridSearchCVValidation:
    """Tests for parameter validation in GridSearchCV."""

    def test_gridsearchcv_invalid_scoring_no_at_sign(self):
        """Test that invalid scoring format raises ValueError."""
        with pytest.raises(ValueError, match="Scoring format must be"):
            GridSearchCV(
                strategy="popularity",
                param_grid={"dummy": [1]},
                scoring="ndcg",  # Missing @k
            )

    def test_gridsearchcv_invalid_scoring_unknown_metric(self):
        """Test that unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            GridSearchCV(
                strategy="popularity",
                param_grid={"dummy": [1]},
                scoring="invalid@10",
            )

    def test_gridsearchcv_invalid_scoring_non_numeric_k(self):
        """Test that non-numeric k raises ValueError."""
        with pytest.raises(ValueError, match="Invalid k value"):
            GridSearchCV(
                strategy="popularity",
                param_grid={"dummy": [1]},
                scoring="ndcg@abc",
            )


# ============================================================================
# GridSearchCV Fit Tests
# ============================================================================

class TestGridSearchCVFit:
    """Tests for GridSearchCV.fit() behavior."""

    def test_gridsearchcv_fit_basic(self, small_interactions):
        """Test GridSearchCV.fit() with a simple param grid."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1, 2]},
            cv=2,
        )
        result = grid.fit(small_interactions)
        assert result is grid  # Should return self

    def test_gridsearchcv_fit_empty_dataframe_raises(self):
        """Test that fitting on empty DataFrame raises ValueError."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1]},
        )
        with pytest.raises(ValueError, match="interactions DataFrame is empty"):
            grid.fit(pd.DataFrame())

    def test_gridsearchcv_best_params_set_after_fit(self, small_interactions):
        """Test that best_params_ is populated after fit."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1, 2, 3]},
            cv=2,
        )
        grid.fit(small_interactions)
        assert isinstance(grid.best_params_, dict)
        assert "dummy" in grid.best_params_
        assert grid.best_params_["dummy"] in [1, 2, 3]

    def test_gridsearchcv_best_score_set_after_fit(self, small_interactions):
        """Test that best_score_ is set after fit."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1, 2]},
            cv=2,
        )
        grid.fit(small_interactions)
        assert isinstance(grid.best_score_, float)
        assert grid.best_score_ >= -np.inf

    def test_gridsearchcv_results_is_dataframe_after_fit(self, small_interactions):
        """Test that results_ is a DataFrame after fit."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1, 2]},
            cv=2,
        )
        grid.fit(small_interactions)
        assert isinstance(grid.results_, pd.DataFrame)
        assert len(grid.results_) == 2  # One row per param combination
        assert "mean_score" in grid.results_.columns
        assert "dummy" in grid.results_.columns

    def test_gridsearchcv_n_iter_set_correctly(self, small_interactions):
        """Test that n_iter_ reflects number of evaluated combinations."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1, 2, 3, 4]},
            cv=2,
        )
        grid.fit(small_interactions)
        assert grid.n_iter_ == 4

    def test_gridsearchcv_best_model_fitted(self, small_interactions):
        """Test that best_model_ is fitted after fit()."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1, 2]},
            cv=2,
        )
        grid.fit(small_interactions)
        assert grid.best_model_ is not None
        assert grid.best_model_._baseline is not None

    def test_gridsearchcv_multi_param_grid(self, small_interactions):
        """Test GridSearchCV with multiple parameters."""
        param_grid = {"dummy": [1, 2], "other": [10, 20]}
        grid = GridSearchCV(
            strategy="popularity",
            param_grid=param_grid,
            cv=2,
        )
        grid.fit(small_interactions)
        assert grid.n_iter_ == 4  # 2 * 2 combinations
        assert "dummy" in grid.best_params_
        assert "other" in grid.best_params_
        assert len(grid.results_) == 4

    def test_gridsearchcv_single_param_value(self, small_interactions):
        """Test edge case with single parameter value."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1]},
            cv=2,
        )
        grid.fit(small_interactions)
        assert grid.n_iter_ == 1
        assert grid.best_params_["dummy"] == 1


# ============================================================================
# GridSearchCV best_model() Tests
# ============================================================================

class TestGridSearchCVBestModel:
    """Tests for GridSearchCV.best_model() method."""

    def test_gridsearchcv_best_model_returns_recommender(self, small_interactions):
        """Test that best_model() returns a fitted OrchidRecommender."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1, 2]},
            cv=2,
        )
        grid.fit(small_interactions)
        model = grid.best_model()
        assert model is not None
        assert hasattr(model, "recommend")
        assert hasattr(model, "predict")

    def test_gridsearchcv_best_model_raises_before_fit(self):
        """Test that best_model() raises RuntimeError before fit."""
        grid = GridSearchCV(
            strategy="popularity",
            param_grid={"dummy": [1]},
        )
        with pytest.raises(RuntimeError, match="has not been fit yet"):
            grid.best_model()


# ============================================================================
# RandomSearchCV Tests
# ============================================================================

class TestRandomSearchCVConstruction:
    """Tests for RandomSearchCV __init__."""

    def test_randomsearchcv_construction_valid_params(self):
        """Test RandomSearchCV constructs with valid parameters."""
        param_dist = {"dummy": [1, 2, 3]}
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions=param_dist,
            n_iter=5,
            cv=3,
        )
        assert rsearch.strategy == "popularity"
        assert rsearch.param_distributions == param_dist
        assert rsearch.n_iter == 5
        assert rsearch.cv == 3

    def test_randomsearchcv_construction_defaults(self):
        """Test RandomSearchCV uses correct defaults."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1]},
        )
        assert rsearch.n_iter == 10
        assert rsearch.cv == 3
        assert rsearch.scoring == "ndcg@10"
        assert rsearch.random_state == 42

    def test_randomsearchcv_n_iter_minimum_enforced(self):
        """Test that n_iter is forced to at least 1."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1]},
            n_iter=0,  # Should be raised to 1
        )
        assert rsearch.n_iter == 1

    def test_randomsearchcv_cv_minimum_enforced(self):
        """Test that cv is forced to at least 2."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1]},
            cv=1,  # Should be raised to 2
        )
        assert rsearch.cv == 2


# ============================================================================
# RandomSearchCV Validation Tests
# ============================================================================

class TestRandomSearchCVValidation:
    """Tests for parameter validation in RandomSearchCV."""

    def test_randomsearchcv_invalid_scoring_no_at_sign(self):
        """Test that invalid scoring format raises ValueError."""
        with pytest.raises(ValueError, match="Scoring format must be"):
            RandomSearchCV(
                strategy="popularity",
                param_distributions={"dummy": [1]},
                scoring="ndcg",  # Missing @k
            )

    def test_randomsearchcv_invalid_scoring_unknown_metric(self):
        """Test that unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            RandomSearchCV(
                strategy="popularity",
                param_distributions={"dummy": [1]},
                scoring="invalid@10",
            )


# ============================================================================
# RandomSearchCV Fit Tests
# ============================================================================

class TestRandomSearchCVFit:
    """Tests for RandomSearchCV.fit() behavior."""

    def test_randomsearchcv_fit_basic(self, small_interactions):
        """Test RandomSearchCV.fit() with a simple param distribution."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1, 2]},
            n_iter=2,
            cv=2,
        )
        result = rsearch.fit(small_interactions)
        assert result is rsearch  # Should return self

    def test_randomsearchcv_fit_empty_dataframe_raises(self):
        """Test that fitting on empty DataFrame raises ValueError."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1]},
        )
        with pytest.raises(ValueError, match="interactions DataFrame is empty"):
            rsearch.fit(pd.DataFrame())

    def test_randomsearchcv_n_iter_respected(self, small_interactions):
        """Test that RandomSearchCV respects n_iter."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1, 2, 3, 4, 5]},
            n_iter=3,
            cv=2,
        )
        rsearch.fit(small_interactions)
        assert rsearch.n_iter_ == 3
        assert len(rsearch.results_) == 3

    def test_randomsearchcv_best_params_set_after_fit(self, small_interactions):
        """Test that best_params_ is populated after fit."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1, 2, 3]},
            n_iter=3,
            cv=2,
        )
        rsearch.fit(small_interactions)
        assert isinstance(rsearch.best_params_, dict)
        assert "dummy" in rsearch.best_params_
        assert rsearch.best_params_["dummy"] in [1, 2, 3]

    def test_randomsearchcv_best_score_set_after_fit(self, small_interactions):
        """Test that best_score_ is set after fit."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1, 2]},
            n_iter=2,
            cv=2,
        )
        rsearch.fit(small_interactions)
        assert isinstance(rsearch.best_score_, float)
        assert rsearch.best_score_ >= -np.inf

    def test_randomsearchcv_results_is_dataframe_after_fit(self, small_interactions):
        """Test that results_ is a DataFrame after fit."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1, 2]},
            n_iter=2,
            cv=2,
        )
        rsearch.fit(small_interactions)
        assert isinstance(rsearch.results_, pd.DataFrame)
        assert len(rsearch.results_) == 2  # n_iter rows
        assert "mean_score" in rsearch.results_.columns

    def test_randomsearchcv_best_model_fitted(self, small_interactions):
        """Test that best_model_ is fitted after fit()."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1, 2]},
            n_iter=2,
            cv=2,
        )
        rsearch.fit(small_interactions)
        assert rsearch.best_model_ is not None
        assert rsearch.best_model_._baseline is not None

    def test_randomsearchcv_multi_param_distribution(self, small_interactions):
        """Test RandomSearchCV with multiple parameters."""
        param_dist = {
            "dummy": [1, 2, 3],
            "other": [10, 20, 30],
        }
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions=param_dist,
            n_iter=5,
            cv=2,
        )
        rsearch.fit(small_interactions)
        assert rsearch.n_iter_ == 5
        assert "dummy" in rsearch.best_params_
        assert "other" in rsearch.best_params_
        assert len(rsearch.results_) == 5


# ============================================================================
# RandomSearchCV best_model() Tests
# ============================================================================

class TestRandomSearchCVBestModel:
    """Tests for RandomSearchCV.best_model() method."""

    def test_randomsearchcv_best_model_returns_recommender(self, small_interactions):
        """Test that best_model() returns a fitted OrchidRecommender."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1, 2]},
            n_iter=2,
            cv=2,
        )
        rsearch.fit(small_interactions)
        model = rsearch.best_model()
        assert model is not None
        assert hasattr(model, "recommend")
        assert hasattr(model, "predict")

    def test_randomsearchcv_best_model_raises_before_fit(self):
        """Test that best_model() raises RuntimeError before fit."""
        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions={"dummy": [1]},
        )
        with pytest.raises(RuntimeError, match="has not been fit yet"):
            rsearch.best_model()


# ============================================================================
# Integration and Edge Case Tests
# ============================================================================

class TestTuningIntegration:
    """Integration tests for tuning utilities."""

    def test_gridsearchcv_and_randomsearchcv_same_data(self, small_interactions):
        """Test that both search methods can run on the same data."""
        param_grid = {"dummy": [1, 2]}

        grid = GridSearchCV(
            strategy="popularity",
            param_grid=param_grid,
            cv=2,
        )
        grid.fit(small_interactions)

        rsearch = RandomSearchCV(
            strategy="popularity",
            param_distributions=param_grid,
            n_iter=2,
            cv=2,
        )
        rsearch.fit(small_interactions)

        assert grid.best_params_ is not None
        assert rsearch.best_params_ is not None
        assert grid.best_model_ is not None
        assert rsearch.best_model_ is not None

    def test_gridsearchcv_results_consistency(self, small_interactions):
        """Test that GridSearchCV results are consistent across runs with same seed."""
        param_grid = {"dummy": [1, 2, 3]}

        grid1 = GridSearchCV(
            strategy="popularity",
            param_grid=param_grid,
            cv=2,
            random_state=42,
        )
        grid1.fit(small_interactions)

        grid2 = GridSearchCV(
            strategy="popularity",
            param_grid=param_grid,
            cv=2,
            random_state=42,
        )
        grid2.fit(small_interactions)

        assert grid1.best_params_ == grid2.best_params_
        assert grid1.best_score_ == grid2.best_score_
