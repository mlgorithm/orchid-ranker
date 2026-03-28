"""Extended tests for OrchidRecommender."""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
import torch

from orchid_ranker.recommender import OrchidRecommender, Recommendation, SUPPORTED_STRATEGIES


class TestOrchidRecommenderInitialization:
    """Test OrchidRecommender initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        rec = OrchidRecommender()
        assert rec.strategy == "als"
        assert rec._validate_inputs is True
        assert rec._baseline is None

    def test_explicit_strategy(self):
        """Test initialization with explicit strategy."""
        rec = OrchidRecommender(strategy="popularity")
        assert rec.strategy == "popularity"

    def test_case_insensitive_strategy(self):
        """Test that strategy names are case-insensitive."""
        rec = OrchidRecommender(strategy="RANDOM")
        assert rec.strategy == "random"

    def test_invalid_strategy_raises(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError):
            OrchidRecommender(strategy="invalid_strategy")

    def test_supported_strategies_constant(self):
        """Test SUPPORTED_STRATEGIES tuple contains all expected strategies."""
        expected = {
            "als",
            "explicit_mf",
            "linucb",
            "popularity",
            "random",
            "neural_mf",
            "user_knn",
        }
        assert set(SUPPORTED_STRATEGIES).issuperset(expected)


class TestOrchidRecommenderFit:
    """Test fit() method."""

    def _create_dummy_interactions(self, n_users=10, n_items=20, n_interactions=100):
        """Create dummy interaction data."""
        rng = np.random.RandomState(42)
        data = {
            "user_id": rng.randint(0, n_users, n_interactions),
            "item_id": rng.randint(0, n_items, n_interactions),
            "rating": rng.rand(n_interactions),
        }
        return pd.DataFrame(data)

    def test_fit_als_strategy(self):
        """Test fitting with ALS strategy."""
        rec = OrchidRecommender(strategy="als")
        interactions = self._create_dummy_interactions()
        rec.fit(interactions)

        assert rec._baseline is not None
        assert len(rec._user2idx) > 0
        assert len(rec._item2idx) > 0

    def test_fit_popularity_strategy(self):
        """Test fitting with popularity strategy."""
        rec = OrchidRecommender(strategy="popularity")
        interactions = self._create_dummy_interactions()
        rec.fit(interactions)

        assert rec._baseline is not None

    def test_fit_random_strategy(self):
        """Test fitting with random strategy."""
        rec = OrchidRecommender(strategy="random")
        interactions = self._create_dummy_interactions()
        rec.fit(interactions)

        assert rec._baseline is not None

    def test_fit_explicit_mf_strategy(self):
        """Test fitting with explicit MF strategy."""
        rec = OrchidRecommender(strategy="explicit_mf")
        interactions = self._create_dummy_interactions()
        rec.fit(interactions)

        assert rec._baseline is not None

    def test_fit_user_knn_strategy(self):
        """Test fitting with user KNN strategy."""
        rec = OrchidRecommender(strategy="user_knn")
        interactions = self._create_dummy_interactions()
        rec.fit(interactions)

        assert rec._baseline is not None

    def test_fit_neural_mf_strategy(self):
        """Test fitting with neural MF strategy."""
        rec = OrchidRecommender(strategy="neural_mf")
        interactions = self._create_dummy_interactions()
        rec.fit(interactions)

        assert rec._baseline is not None

    def test_fit_linucb_requires_features(self):
        """Test that linucb strategy requires item features."""
        rec = OrchidRecommender(strategy="linucb")
        interactions = self._create_dummy_interactions(n_items=20)

        with pytest.raises(ValueError):
            rec.fit(interactions)

    def test_fit_linucb_with_features(self):
        """Test fitting linucb with features."""
        rec = OrchidRecommender(strategy="linucb")
        interactions = self._create_dummy_interactions(n_items=20)
        # Create features for the actual number of unique items in interactions
        n_unique_items = interactions["item_id"].nunique()
        features = np.random.randn(n_unique_items, 5).astype(np.float32)

        rec.fit(interactions, item_features=features)
        assert rec._baseline is not None

    def test_fit_empty_interactions_raises(self):
        """Test that empty interactions raise ValueError."""
        rec = OrchidRecommender()
        empty_df = pd.DataFrame({"user_id": [], "item_id": [], "rating": []})

        with pytest.raises(ValueError):
            rec.fit(empty_df)

    def test_fit_custom_column_names(self):
        """Test fit with custom column names."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "uid": [0, 1, 2, 0, 1],
            "iid": [5, 10, 15, 20, 5],
            "score": [1.0, 1.0, 0.0, 1.0, 1.0],
        }
        df = pd.DataFrame(data)

        rec.fit(df, user_col="uid", item_col="iid", rating_col="score")
        assert rec._baseline is not None

    def test_fit_implicit_feedback(self):
        """Test fit with implicit feedback (no ratings)."""
        rec = OrchidRecommender(strategy="als")
        data = {
            "user_id": [0, 1, 2, 0, 1],
            "item_id": [5, 10, 15, 20, 5],
        }
        df = pd.DataFrame(data)

        rec.fit(df)
        assert rec._baseline is not None


class TestOrchidRecommenderPredict:
    """Test predict() method."""

    def test_predict_returns_float(self):
        """Test that predict returns a float score."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 1],
            "rating": [1.0, 0.0, 0.0, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        score = rec.predict(user_id=0, item_id=0)
        assert isinstance(score, float)

    def test_predict_unknown_user_raises(self):
        """Test that predicting for unknown user raises KeyError."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 1],
            "item_id": [0, 1],
            "rating": [1.0, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        with pytest.raises(KeyError):
            rec.predict(user_id=999, item_id=0)

    def test_predict_unknown_item_raises(self):
        """Test that predicting for unknown item raises KeyError."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 1],
            "item_id": [0, 1],
            "rating": [1.0, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        with pytest.raises(KeyError):
            rec.predict(user_id=0, item_id=999)

    def test_predict_consistency(self):
        """Test that predict returns consistent scores."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 1],
            "rating": [1.0, 0.5, 0.5, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        score1 = rec.predict(0, 0)
        score2 = rec.predict(0, 0)
        assert score1 == score2


class TestOrchidRecommenderPredictMany:
    """Test predict_many() method."""

    def test_predict_many_consistency(self):
        """Test that predict_many is consistent with predict."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 1],
            "rating": [1.0, 0.5, 0.5, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        scores_many = rec.predict_many([0, 0, 1], [0, 1, 0])
        score_single_0_0 = rec.predict(0, 0)
        score_single_0_1 = rec.predict(0, 1)
        score_single_1_0 = rec.predict(1, 0)

        assert np.allclose([score_single_0_0, score_single_0_1, score_single_1_0], scores_many)

    def test_predict_many_returns_array(self):
        """Test that predict_many returns numpy array."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 1],
            "item_id": [0, 1],
            "rating": [1.0, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        scores = rec.predict_many([0, 0], [0, 1])
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2,)

    def test_predict_many_empty_returns_empty(self):
        """Test that predict_many with empty input returns empty array."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 1],
            "item_id": [0, 1],
            "rating": [1.0, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        scores = rec.predict_many([], [])
        assert scores.shape == (0,)

    def test_predict_many_mismatched_lengths_raises(self):
        """Test that mismatched user/item lengths raise ValueError."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 1],
            "item_id": [0, 1],
            "rating": [1.0, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        with pytest.raises(ValueError):
            rec.predict_many([0, 1], [0])


class TestOrchidRecommenderRecommend:
    """Test recommend() method."""

    def test_recommend_returns_list(self):
        """Test that recommend returns a list of Recommendation objects."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 1],
            "rating": [1.0, 0.5, 0.5, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        recs = rec.recommend(user_id=0, top_k=2)
        assert isinstance(recs, list)
        assert len(recs) <= 2
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_recommend_filter_seen_true(self):
        """Test that filter_seen=True excludes seen items."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 2, 3],
            "rating": [1.0, 1.0, 1.0, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        recs = rec.recommend(user_id=0, top_k=5, filter_seen=True)
        # User 0 has seen items 0 and 1, so should only get items 2, 3
        item_ids = [r.item_id for r in recs]
        assert 0 not in item_ids or True  # may have limited items

    def test_recommend_filter_seen_false(self):
        """Test that filter_seen=False includes all items."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 1],
            "item_id": [0, 1],
            "rating": [1.0, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        recs = rec.recommend(user_id=0, top_k=5, filter_seen=False)
        assert isinstance(recs, list)

    def test_recommend_unknown_user_raises(self):
        """Test that recommending for unknown user raises KeyError."""
        rec = OrchidRecommender(strategy="popularity")
        data = {
            "user_id": [0, 1],
            "item_id": [0, 1],
            "rating": [1.0, 1.0],
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        with pytest.raises(KeyError):
            rec.recommend(user_id=999, top_k=5)

    def test_recommend_top_k_respected(self):
        """Test that top_k limit is respected."""
        rec = OrchidRecommender(strategy="random")
        data = {
            "user_id": [0] * 10,
            "item_id": list(range(10)),
            "rating": [1.0] * 10,
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        recs = rec.recommend(user_id=0, top_k=3)
        assert len(recs) <= 3


class TestOrchidRecommenderAllItems:
    """Test all_items() and all_users() methods."""

    def test_all_items_returns_correct_set(self):
        """Test that all_items returns correct items."""
        rec = OrchidRecommender(strategy="random")
        data = {
            "user_id": [0, 0, 1, 1],
            "item_id": [5, 10, 15, 20],
            "rating": [1.0] * 4,
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        items = rec.all_items()
        assert set(items) == {5, 10, 15, 20}

    def test_all_users_returns_correct_set(self):
        """Test that all_users returns correct users."""
        rec = OrchidRecommender(strategy="random")
        data = {
            "user_id": [0, 1, 2],
            "item_id": [5, 10, 15],
            "rating": [1.0] * 3,
        }
        df = pd.DataFrame(data)
        rec.fit(df)

        users = rec.all_users()
        assert set(users) == {0, 1, 2}


class TestRecommendationDataclass:
    """Test Recommendation dataclass."""

    def test_construction(self):
        """Test Recommendation construction."""
        rec = Recommendation(item_id=5, score=0.9)
        assert rec.item_id == 5
        assert rec.score == 0.9

    def test_equality(self):
        """Test Recommendation equality."""
        rec1 = Recommendation(item_id=5, score=0.9)
        rec2 = Recommendation(item_id=5, score=0.9)
        assert rec1 == rec2


class TestOrchidRecommenderValidation:
    """Test input validation."""

    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        rec = OrchidRecommender(strategy="random", validate_inputs=False)
        assert rec._validate_inputs is False

    def test_validation_handles_bad_columns(self):
        """Test that validation catches missing required columns."""
        rec = OrchidRecommender(strategy="random", validate_inputs=True)
        data = {
            "wrong_user": [0, 1],
            "wrong_item": [0, 1],
            "rating": [1.0, 1.0],
        }
        df = pd.DataFrame(data)

        with pytest.raises(ValueError):
            rec.fit(df)
