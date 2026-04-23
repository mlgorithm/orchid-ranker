"""Tests for orchid_ranker.cold_start -- content similarity, popularity, bridge."""
from __future__ import annotations

import numpy as np
import pytest

from orchid_ranker.cold_start import (
    ColdStartBridge,
    ColdStartConfig,
    ItemFeatureIndex,
    PopularityPrior,
)


# =========================================================================
# ItemFeatureIndex
# =========================================================================
class TestItemFeatureIndex:
    def test_self_similarity_excluded(self) -> None:
        feats = np.eye(5, dtype=np.float32)
        idx = ItemFeatureIndex(feats)
        results = idx.similar_items(0, top_k=3)
        assert all(iid != 0 for iid, _ in results)

    def test_identical_items_have_high_similarity(self) -> None:
        feats = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        idx = ItemFeatureIndex(feats)
        results = idx.similar_items(0, top_k=1)
        assert results[0][0] == 1  # item 1 is identical to item 0
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_items_have_zero_similarity(self) -> None:
        feats = np.eye(3, dtype=np.float32)
        idx = ItemFeatureIndex(feats)
        sim = idx.similarity(0, 1)
        assert sim == pytest.approx(0.0, abs=1e-5)

    def test_exclude_works(self) -> None:
        feats = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        idx = ItemFeatureIndex(feats)
        results = idx.similar_items(0, top_k=2, exclude={1})
        assert all(iid != 1 for iid, _ in results)

    def test_user_profile_scores_from_seeds(self) -> None:
        feats = np.array([
            [1, 0, 0],
            [1, 0.1, 0],  # similar to 0
            [0, 0, 1],    # different from 0
        ], dtype=np.float32)
        idx = ItemFeatureIndex(feats)
        scores = idx.user_profile_scores([0])
        # Item 1 should score higher than item 2
        assert scores[1] > scores[2]

    def test_empty_seeds_return_zeros(self) -> None:
        feats = np.eye(3, dtype=np.float32)
        idx = ItemFeatureIndex(feats)
        scores = idx.user_profile_scores([])
        assert np.all(scores == 0.0)

    def test_num_items(self) -> None:
        feats = np.random.randn(10, 4).astype(np.float32)
        idx = ItemFeatureIndex(feats)
        assert idx.num_items == 10

    def test_rejects_1d_features(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            ItemFeatureIndex(np.array([1, 2, 3], dtype=np.float32))

    def test_rejects_unsupported_metric(self) -> None:
        with pytest.raises(ValueError, match="Unsupported metric"):
            ItemFeatureIndex(np.eye(3), metric="euclidean")


# =========================================================================
# PopularityPrior
# =========================================================================
class TestPopularityPrior:
    def test_fit_and_score(self) -> None:
        pop = PopularityPrior(smoothing=0.0)
        # Item 1 appears 3x, item 2 appears 1x
        pop.fit([1, 1, 1, 2])
        scores = pop.scores([1, 2, 3])
        assert scores[0] > scores[1]  # item 1 more popular
        assert scores[2] == 0.0       # item 3 never seen

    def test_smoothing_prevents_zeros(self) -> None:
        pop = PopularityPrior(smoothing=1.0)
        pop.fit([1, 1, 2])
        scores = pop.scores([1, 2, 99])
        assert scores[2] > 0.0  # smoothing gives unseen items a non-zero score

    def test_scores_normalised_to_unit(self) -> None:
        pop = PopularityPrior(smoothing=0.0)
        pop.fit([1, 1, 1, 2, 3])
        scores = pop.scores([1, 2, 3])
        assert scores.max() == pytest.approx(1.0)
        assert np.all(scores >= 0.0)

    def test_segment_scores(self) -> None:
        pop = PopularityPrior(smoothing=0.0)
        pop.fit(
            [1, 1, 2, 2, 2],
            segments=["A", "A", "B", "B", "B"],
        )
        # In segment A, item 1 is popular; in segment B, item 2 is popular
        scores_a = pop.scores([1, 2], segment="A")
        scores_b = pop.scores([1, 2], segment="B")
        assert scores_a[0] > scores_a[1]  # item 1 popular in A
        assert scores_b[1] > scores_b[0]  # item 2 popular in B

    def test_is_fitted(self) -> None:
        pop = PopularityPrior()
        assert not pop.is_fitted
        pop.fit([1, 2, 3])
        assert pop.is_fitted


# =========================================================================
# ColdStartConfig
# =========================================================================
class TestColdStartConfig:
    def test_defaults(self) -> None:
        cfg = ColdStartConfig()
        assert cfg.min_interactions == 3
        assert cfg.blend_until == 20
        assert 0 < cfg.popularity_weight < 1

    def test_custom_values(self) -> None:
        cfg = ColdStartConfig(min_interactions=10, blend_until=50)
        assert cfg.min_interactions == 10
        assert cfg.blend_until == 50


# =========================================================================
# ColdStartBridge
# =========================================================================
class _MockRecommender:
    """Minimal mock that returns items sorted by ID (score = 1/id)."""
    def recommend(self, user_id, top_k, candidate_item_ids=None):
        items = candidate_item_ids or list(range(10))
        items = sorted(items)[:top_k]
        return [type("Rec", (), {"item_id": i, "score": 1.0 / max(i, 1)})() for i in items]


class _FailingRecommender:
    """Recommender that raises KeyError (simulates unknown user)."""
    def recommend(self, user_id, top_k, candidate_item_ids=None):
        raise KeyError(f"Unknown user {user_id}")


class TestColdStartBridge:
    @pytest.fixture
    def feats(self):
        """10 items with 4-dim features."""
        rng = np.random.default_rng(42)
        return rng.normal(size=(10, 4)).astype(np.float32)

    @pytest.fixture
    def bridge(self, feats):
        pop = PopularityPrior(smoothing=1.0)
        pop.fit([0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        return ColdStartBridge(
            recommender=_MockRecommender(),
            item_features=feats,
            popularity_prior=pop,
            config=ColdStartConfig(min_interactions=3, blend_until=10),
        )

    def test_new_user_gets_recommendations(self, bridge) -> None:
        """A brand-new user with 0 interactions should still get recs."""
        recs = bridge.recommend(user_id=999, top_k=5)
        assert len(recs) == 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)

    def test_warmth_zero_for_new_user(self, bridge) -> None:
        assert bridge.warmth(999) == 0.0

    def test_warmth_increases_with_observations(self, bridge) -> None:
        for i in range(5):
            bridge.observe(user_id=42, item_id=i, outcome=1.0)
        w = bridge.warmth(42)
        assert 0.0 < w < 1.0  # in the blend zone

    def test_warmth_reaches_one(self, bridge) -> None:
        for i in range(15):
            bridge.observe(user_id=42, item_id=i % 10, outcome=1.0)
        assert bridge.warmth(42) == 1.0

    def test_interaction_count(self, bridge) -> None:
        assert bridge.interaction_count(42) == 0
        bridge.observe(42, 1, 1.0)
        bridge.observe(42, 2, 1.0)
        assert bridge.interaction_count(42) == 2

    def test_cold_user_ignores_orchid(self, feats) -> None:
        """With 0 interactions, even a failing recommender should work."""
        bridge = ColdStartBridge(
            recommender=_FailingRecommender(),
            item_features=feats,
            config=ColdStartConfig(min_interactions=3, blend_until=10),
        )
        recs = bridge.recommend(user_id=999, top_k=5)
        assert len(recs) == 5  # pure cold-start, no Orchid needed

    def test_warm_user_uses_orchid(self, bridge) -> None:
        """After enough interactions, Orchid scores dominate."""
        for i in range(15):
            bridge.observe(user_id=1, item_id=i % 10, outcome=1.0)
        recs = bridge.recommend(user_id=1, top_k=5, candidate_item_ids=list(range(10)))
        assert len(recs) == 5

    def test_seed_items_influence_content_scores(self, feats) -> None:
        bridge = ColdStartBridge(
            recommender=_MockRecommender(),
            item_features=feats,
            config=ColdStartConfig(min_interactions=3, blend_until=10),
        )
        # Get recs with explicit seed items
        recs_seed0 = bridge.recommend(user_id=999, top_k=5, seed_item_ids=[0])
        recs_seed5 = bridge.recommend(user_id=999, top_k=5, seed_item_ids=[5])
        # Different seeds should produce different rankings
        ids_0 = [r[0] for r in recs_seed0]
        ids_5 = [r[0] for r in recs_seed5]
        assert ids_0 != ids_5

    def test_rejects_invalid_config(self, feats) -> None:
        with pytest.raises(ValueError, match="min_interactions"):
            ColdStartBridge(
                recommender=_MockRecommender(),
                item_features=feats,
                config=ColdStartConfig(min_interactions=50, blend_until=10),
            )

    def test_repr(self, bridge) -> None:
        r = repr(bridge)
        assert "ColdStartBridge" in r
        assert "min_interactions" in r

    def test_no_popularity_prior_uses_uniform(self, feats) -> None:
        """Without a popularity prior, cold-start still works (uniform pop)."""
        bridge = ColdStartBridge(
            recommender=_MockRecommender(),
            item_features=feats,
        )
        recs = bridge.recommend(user_id=999, top_k=5)
        assert len(recs) == 5
