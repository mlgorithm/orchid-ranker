"""Comprehensive round-trip correctness tests for all OrchidRecommender strategies.

Tests behavioral assertions for each of the 9 supported strategies:
- als, explicit_mf, linucb, popularity, random, neural_mf, user_knn, implicit_als, implicit_bpr

For each strategy:
- Verifies fit() succeeds on small known dataset
- Verifies recommend() returns non-empty list of Recommendation objects
- Verifies all scores are finite (no NaN, Inf)
- Verifies filter_seen=True excludes seen items
- Verifies predict() returns finite float
- Verifies predict_many() returns correct shape and dtype
- Verifies predict() and predict_many() consistency
- Verifies top-k ranking: highest score is first
"""

import logging
import numpy as np
import pandas as pd
import pytest

from orchid_ranker import OrchidRecommender, Recommendation, SUPPORTED_STRATEGIES


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_known_data():
    """Small known dataset: 20 users, 30 items, 500 interactions.

    Used for exhaustive strategy testing with reproducible results.
    """
    rng = np.random.RandomState(42)
    user_ids = rng.randint(0, 20, size=500)
    item_ids = rng.randint(0, 30, size=500)
    ratings = rng.uniform(1, 5, size=500).round(1)

    return pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings,
    })


@pytest.fixture
def linucb_item_features():
    """Item feature matrix for LinUCB strategy.

    30 items x 8-dimensional feature vectors, dtype float32.
    """
    rng = np.random.RandomState(42)
    return rng.randn(30, 8).astype(np.float32)


@pytest.fixture
def strategie_params():
    """Strategy-specific initialization parameters for reproducibility."""
    return {
        "als": {"epochs": 3},
        "explicit_mf": {"epochs": 3, "emb_dim": 16},
        "linucb": {"alpha": 1.5},
        "popularity": {},
        "random": {},
        "neural_mf": {"epochs": 2, "emb_dim": 8, "hidden": (16,)},
        "user_knn": {"k": 5},
        "implicit_als": {"factors": 8, "iterations": 3},
        "implicit_bpr": {"factors": 8, "iterations": 3},
    }


# ────────────────────────────────────────────────────────────────────────────
# Parametrized tests: all strategies
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
class TestStrategyCorrectness:
    """Test correctness of each strategy across 9 expected behaviors."""

    def test_fit_succeeds_on_known_data(self, strategy, small_known_data, strategie_params):
        """Verify fit() completes without error."""
        kwargs = strategie_params.get(strategy, {})
        rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)

        # Fit should not raise
        rec.fit(small_known_data, rating_col="rating")
        assert rec._baseline is not None, f"{strategy} baseline not initialized"
        assert len(rec.all_users()) > 0
        assert len(rec.all_items()) > 0

    def test_recommend_returns_recommendation_objects(
        self, strategy, small_known_data, strategie_params
    ):
        """Verify recommend() returns list of Recommendation objects."""
        if strategy == "implicit_als" or strategy == "implicit_bpr":
            pytest.importorskip("implicit")

        kwargs = strategie_params.get(strategy, {})
        rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)
        rec.fit(small_known_data, rating_col="rating")

        # Pick a known user
        user_id = small_known_data["user_id"].iloc[0]
        recs = rec.recommend(user_id, top_k=5)

        assert isinstance(recs, list)
        assert len(recs) > 0, f"{strategy} should return at least one recommendation"
        assert all(isinstance(r, Recommendation) for r in recs)
        assert all(isinstance(r.item_id, int) for r in recs)
        assert all(isinstance(r.score, float) for r in recs)

    def test_all_scores_finite(self, strategy, small_known_data, strategie_params):
        """Verify all returned scores are finite (no NaN, Inf)."""
        if strategy == "implicit_als" or strategy == "implicit_bpr":
            pytest.importorskip("implicit")

        kwargs = strategie_params.get(strategy, {})
        rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)
        rec.fit(small_known_data, rating_col="rating")

        user_id = small_known_data["user_id"].iloc[0]
        recs = rec.recommend(user_id, top_k=10)

        for r in recs:
            assert np.isfinite(r.score), f"{strategy}: score is not finite: {r.score}"
            assert not np.isnan(r.score), f"{strategy}: score is NaN"
            assert not np.isinf(r.score), f"{strategy}: score is Inf"

    def test_filter_seen_excludes_seen_items(
        self, strategy, small_known_data, strategie_params
    ):
        """Verify filter_seen=True excludes items user has interacted with."""
        if strategy == "implicit_als" or strategy == "implicit_bpr":
            pytest.importorskip("implicit")

        kwargs = strategie_params.get(strategy, {})
        rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)
        rec.fit(small_known_data, rating_col="rating")

        # Get a user who has interacted with items
        user_id = small_known_data.iloc[0]["user_id"]
        seen_items = set(
            small_known_data[small_known_data["user_id"] == user_id]["item_id"].unique()
        )

        if not seen_items:
            pytest.skip(f"User {user_id} has no interactions")

        recs_with_filter = rec.recommend(user_id, top_k=100, filter_seen=True)
        rec_item_ids = {r.item_id for r in recs_with_filter}

        overlap = rec_item_ids & seen_items
        assert len(overlap) == 0, (
            f"{strategy}: filter_seen=True returned {len(overlap)} seen items. "
            f"Seen: {seen_items}, Got: {rec_item_ids}"
        )

    def test_predict_returns_finite_float(
        self, strategy, small_known_data, strategie_params
    ):
        """Verify predict(user, item) returns finite float."""
        if strategy == "implicit_als" or strategy == "implicit_bpr":
            pytest.importorskip("implicit")

        kwargs = strategie_params.get(strategy, {})
        rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)
        rec.fit(small_known_data, rating_col="rating")

        # Get a known user-item pair
        user_id = small_known_data.iloc[0]["user_id"]
        item_id = small_known_data.iloc[0]["item_id"]

        score = rec.predict(user_id, item_id)

        assert isinstance(score, float)
        assert np.isfinite(score), f"{strategy}: predict() returned non-finite score"
        assert not np.isnan(score)
        assert not np.isinf(score)

    def test_predict_many_shape_and_dtype(
        self, strategy, small_known_data, strategie_params
    ):
        """Verify predict_many() returns correct shape and dtype."""
        if strategy == "implicit_als" or strategy == "implicit_bpr":
            pytest.importorskip("implicit")

        kwargs = strategie_params.get(strategy, {})
        rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)
        rec.fit(small_known_data, rating_col="rating")

        # Get multiple known pairs
        user_ids = small_known_data["user_id"].iloc[:10].values
        item_ids = small_known_data["item_id"].iloc[:10].values

        scores = rec.predict_many(user_ids, item_ids)

        assert isinstance(scores, np.ndarray)
        assert scores.dtype in (np.float32, np.float64)
        assert scores.shape == (len(user_ids),), (
            f"{strategy}: predict_many shape {scores.shape} != ({len(user_ids)},)"
        )

    def test_predict_and_predict_many_consistency(
        self, strategy, small_known_data, strategie_params
    ):
        """Verify predict() and predict_many() return same score for same pair."""
        if strategy == "implicit_als" or strategy == "implicit_bpr":
            pytest.importorskip("implicit")

        kwargs = strategie_params.get(strategy, {})
        rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)
        rec.fit(small_known_data, rating_col="rating")

        user_id = small_known_data.iloc[0]["user_id"]
        item_id = small_known_data.iloc[0]["item_id"]

        score_single = rec.predict(user_id, item_id)
        scores_many = rec.predict_many([user_id], [item_id])

        assert len(scores_many) == 1
        assert np.isclose(
            score_single, scores_many[0], rtol=1e-5, atol=1e-7
        ), (
            f"{strategy}: predict() and predict_many() disagree. "
            f"Single: {score_single}, Many: {scores_many[0]}"
        )

    def test_top_k_ranking_highest_score_first(
        self, strategy, small_known_data, strategie_params
    ):
        """Verify recommendations are ordered by score (highest first)."""
        if strategy == "implicit_als" or strategy == "implicit_bpr":
            pytest.importorskip("implicit")

        kwargs = strategie_params.get(strategy, {})
        rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)
        rec.fit(small_known_data, rating_col="rating")

        user_id = small_known_data.iloc[0]["user_id"]
        recs = rec.recommend(user_id, top_k=5, filter_seen=False)

        scores = [r.score for r in recs]
        sorted_scores = sorted(scores, reverse=True)

        assert scores == sorted_scores, (
            f"{strategy}: recommendations not sorted by score. "
            f"Got {scores}, expected {sorted_scores}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Strategy-specific tests
# ────────────────────────────────────────────────────────────────────────────

def test_linucb_requires_item_features(small_known_data):
    """Verify LinUCB strategy requires item_features parameter."""
    rec = OrchidRecommender(strategy="linucb")

    with pytest.raises(ValueError, match="item_features"):
        rec.fit(small_known_data, rating_col="rating")


def test_linucb_with_features_works(small_known_data, linucb_item_features):
    """Verify LinUCB works when item_features is provided."""
    rec = OrchidRecommender(strategy="linucb", alpha=1.5)

    # Should not raise
    rec.fit(small_known_data, rating_col="rating", item_features=linucb_item_features)

    user_id = small_known_data.iloc[0]["user_id"]
    recs = rec.recommend(user_id, top_k=5)

    assert len(recs) > 0


def test_implicit_als_skipped_without_library(small_known_data):
    """Verify implicit_als is skipped gracefully if 'implicit' not installed."""
    try:
        import implicit  # noqa: F401
        pytest.skip("Test only runs when 'implicit' is not available")
    except ImportError:
        rec = OrchidRecommender(strategy="implicit_als")
        with pytest.raises(ImportError):
            rec.fit(small_known_data, rating_col="rating")


def test_implicit_bpr_skipped_without_library(small_known_data):
    """Verify implicit_bpr is skipped gracefully if 'implicit' not installed."""
    try:
        import implicit  # noqa: F401
        pytest.skip("Test only runs when 'implicit' is not available")
    except ImportError:
        rec = OrchidRecommender(strategy="implicit_bpr")
        with pytest.raises(ImportError):
            rec.fit(small_known_data, rating_col="rating")


# ────────────────────────────────────────────────────────────────────────────
# Cross-strategy consistency tests
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy1,strategy2", [
    ("als", "explicit_mf"),
    ("popularity", "random"),
    ("user_knn", "als"),
])
def test_different_strategies_produce_different_rankings(
    strategy1, strategy2, small_known_data, strategie_params
):
    """Verify different strategies produce different recommendations (with high probability)."""
    kwargs1 = strategie_params.get(strategy1, {})
    kwargs2 = strategie_params.get(strategy2, {})

    rec1 = OrchidRecommender(strategy=strategy1, epochs=1, **kwargs1)
    rec2 = OrchidRecommender(strategy=strategy2, epochs=1, **kwargs2)

    rec1.fit(small_known_data, rating_col="rating")
    rec2.fit(small_known_data, rating_col="rating")

    user_id = small_known_data.iloc[0]["user_id"]
    recs1 = rec1.recommend(user_id, top_k=5)
    recs2 = rec2.recommend(user_id, top_k=5)

    items1 = {r.item_id for r in recs1}
    items2 = {r.item_id for r in recs2}

    # They should be different (unless by chance they're identical)
    # This is a probabilistic test, but with high probability they differ
    assert items1 != items2 or (strategy1 == strategy2), (
        f"{strategy1} and {strategy2} produced identical top-5 recommendations. "
        f"Items: {items1}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Numerical stability tests
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy", [
    "als", "explicit_mf", "popularity", "random", "neural_mf", "user_knn"
])
def test_scores_remain_finite_across_many_recommendations(
    strategy, small_known_data, strategie_params
):
    """Verify scores don't drift to NaN/Inf with repeated recommendations."""
    kwargs = strategie_params.get(strategy, {})
    rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)
    rec.fit(small_known_data, rating_col="rating")

    user_id = small_known_data.iloc[0]["user_id"]

    # Generate many recommendations
    for _ in range(10):
        recs = rec.recommend(user_id, top_k=10, filter_seen=False)
        for r in recs:
            assert np.isfinite(r.score), f"{strategy}: score became non-finite"


@pytest.mark.parametrize("strategy", [
    "als", "explicit_mf", "neural_mf", "user_knn"
])
def test_predict_many_batch_vs_individual(
    strategy, small_known_data, strategie_params
):
    """Verify predict_many() batch processing matches individual predictions."""
    kwargs = strategie_params.get(strategy, {})
    rec = OrchidRecommender(strategy=strategy, epochs=1, **kwargs)
    rec.fit(small_known_data, rating_col="rating")

    # Get multiple pairs
    user_ids = small_known_data["user_id"].iloc[:5].values
    item_ids = small_known_data["item_id"].iloc[:5].values

    # Batch prediction
    batch_scores = rec.predict_many(user_ids, item_ids)

    # Individual predictions
    individual_scores = np.array([
        rec.predict(int(u), int(i))
        for u, i in zip(user_ids, item_ids)
    ])

    # Should be close (allowing for floating point precision)
    assert np.allclose(
        batch_scores, individual_scores, rtol=1e-5, atol=1e-7
    ), (
        f"{strategy}: batch vs individual mismatch. "
        f"Batch: {batch_scores}, Individual: {individual_scores}"
    )
