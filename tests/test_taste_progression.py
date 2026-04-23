"""Tests for orchid_ranker.taste_progression -- taste evolution, sophistication, ranking."""
from __future__ import annotations

import pytest

from orchid_ranker.taste_progression import (
    SophisticationMapper,
    TasteConfig,
    TasteProfile,
    TasteProgressionRanker,
)


# =========================================================================
# SophisticationMapper
# =========================================================================
class TestSophisticationMapper:
    def test_explicit_scores(self) -> None:
        sm = SophisticationMapper({1: 0.2, 2: 0.8})
        assert sm[1] == pytest.approx(0.2)
        assert sm[2] == pytest.approx(0.8)

    def test_default_score_for_unknown(self) -> None:
        sm = SophisticationMapper({1: 0.5}, default_score=0.3)
        assert sm[99] == pytest.approx(0.3)

    def test_clamps_to_unit_interval(self) -> None:
        sm = SophisticationMapper({1: -0.5, 2: 1.5})
        assert sm[1] == pytest.approx(0.0)
        assert sm[2] == pytest.approx(1.0)

    def test_contains(self) -> None:
        sm = SophisticationMapper({1: 0.5})
        assert 1 in sm
        assert 99 not in sm

    def test_score_batch(self) -> None:
        sm = SophisticationMapper({1: 0.2, 2: 0.8}, default_score=0.5)
        scores = sm.score_batch([1, 2, 3])
        assert scores == [pytest.approx(0.2), pytest.approx(0.8), pytest.approx(0.5)]

    def test_from_prices(self) -> None:
        sm = SophisticationMapper.from_prices({1: 10.0, 2: 50.0, 3: 100.0})
        # Cheapest should be ~0, most expensive should be ~1
        assert sm[1] == pytest.approx(0.0)
        assert sm[3] == pytest.approx(1.0)
        assert 0.0 < sm[2] < 1.0

    def test_from_prices_single_item(self) -> None:
        sm = SophisticationMapper.from_prices({1: 25.0})
        assert sm[1] == pytest.approx(0.0)  # single item = rank 0 / max(0, 1)

    def test_from_prices_empty(self) -> None:
        sm = SophisticationMapper.from_prices({})
        assert sm[1] == pytest.approx(0.5)  # falls back to default


# =========================================================================
# TasteProfile
# =========================================================================
class TestTasteProfile:
    def test_initial_taste_level(self) -> None:
        cfg = TasteConfig(bkt_p_init=0.2)
        tp = TasteProfile(cfg)
        level = tp.taste_level("wine")
        # Should start near p_init (global, since no category observations)
        assert 0.0 <= level <= 1.0

    def test_taste_rises_on_positive_outcomes(self) -> None:
        tp = TasteProfile(TasteConfig(bkt_p_init=0.1))
        level_before = tp.taste_level("wine")
        for _ in range(10):
            tp.observe("wine", positive=True)
        level_after = tp.taste_level("wine")
        assert level_after > level_before

    def test_taste_drops_on_negative_outcomes(self) -> None:
        tp = TasteProfile(TasteConfig(bkt_p_init=0.5))
        # First warm up
        for _ in range(5):
            tp.observe("wine", positive=True)
        level_high = tp.taste_level("wine")
        # Then fail repeatedly
        for _ in range(15):
            tp.observe("wine", positive=False)
        level_low = tp.taste_level("wine")
        assert level_low < level_high

    def test_categories_tracked_independently(self) -> None:
        tp = TasteProfile()
        for _ in range(10):
            tp.observe("wine", positive=True)
            tp.observe("coffee", positive=False)
        wine_level = tp.taste_level("wine")
        coffee_level = tp.taste_level("coffee")
        assert wine_level > coffee_level

    def test_stretch_zone(self) -> None:
        tp = TasteProfile(TasteConfig(stretch_width=0.15))
        for _ in range(5):
            tp.observe("wine", positive=True)
        low, high = tp.stretch_zone("wine")
        assert low < high
        assert 0.0 <= low
        assert high <= 1.0
        level = tp.taste_level("wine")
        assert low <= level <= high

    def test_categories_list(self) -> None:
        tp = TasteProfile()
        tp.observe("wine", positive=True)
        tp.observe("coffee", positive=True)
        assert set(tp.categories) == {"wine", "coffee"}

    def test_interaction_count(self) -> None:
        tp = TasteProfile()
        assert tp.interaction_count == 0
        tp.observe("wine", positive=True)
        tp.observe("wine", positive=False)
        tp.observe("coffee", positive=True)
        assert tp.interaction_count == 3

    def test_global_taste_level(self) -> None:
        tp = TasteProfile()
        for _ in range(10):
            tp.observe("wine", positive=True)
        global_level = tp.taste_level()  # no category = global
        assert global_level > 0.0


# =========================================================================
# TasteConfig
# =========================================================================
class TestTasteConfig:
    def test_defaults(self) -> None:
        cfg = TasteConfig()
        assert cfg.stretch_width == 0.15
        assert cfg.keep_threshold == 4.0
        assert 0 < cfg.bkt_p_transit < 1

    def test_custom(self) -> None:
        cfg = TasteConfig(stretch_width=0.25, bkt_p_init=0.3)
        assert cfg.stretch_width == 0.25
        assert cfg.bkt_p_init == 0.3


# =========================================================================
# TasteProgressionRanker
# =========================================================================
class TestTasteProgressionRanker:
    @pytest.fixture
    def ranker(self):
        soph = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.9}
        r = TasteProgressionRanker(
            sophistication_scores=soph,
            config=TasteConfig(stretch_width=0.15),
        )
        r.set_item_categories({
            0: "wine", 1: "wine", 2: "wine",
            3: "coffee", 4: "coffee", 5: "coffee",
        })
        return r

    def test_recommend_returns_correct_count(self, ranker) -> None:
        recs = ranker.recommend(user_id=1, top_k=3, candidate_item_ids=[0, 1, 2, 3, 4, 5])
        assert len(recs) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)

    def test_observe_updates_taste(self, ranker) -> None:
        level_before = ranker.taste_level(user_id=1, category="wine")
        for _ in range(10):
            ranker.observe(user_id=1, item_id=0, category="wine", purchased=True)
        level_after = ranker.taste_level(user_id=1, category="wine")
        assert level_after > level_before

    def test_return_is_negative_signal(self, ranker) -> None:
        # First build some positive taste
        for _ in range(5):
            ranker.observe(user_id=1, item_id=0, category="wine", purchased=True)
        level_high = ranker.taste_level(1, "wine")
        # Then return items
        for _ in range(10):
            ranker.observe(user_id=1, item_id=0, category="wine",
                           purchased=True, returned=True)
        level_after_returns = ranker.taste_level(1, "wine")
        assert level_after_returns < level_high

    def test_low_rating_is_negative(self, ranker) -> None:
        result = ranker.observe(user_id=1, item_id=0, category="wine",
                                purchased=True, rating=2.0)
        assert result["positive"] is False

    def test_high_rating_is_positive(self, ranker) -> None:
        result = ranker.observe(user_id=1, item_id=0, category="wine",
                                purchased=True, rating=5.0)
        assert result["positive"] is True

    def test_not_purchased_is_negative(self, ranker) -> None:
        result = ranker.observe(user_id=1, item_id=0, category="wine",
                                purchased=False)
        assert result["positive"] is False

    def test_stretch_zone_favours_matching_sophistication(self, ranker) -> None:
        """A user with low taste should rank simple items higher."""
        # New user (low taste level)
        recs = ranker.recommend(user_id=999, top_k=6, candidate_item_ids=[0, 1, 2, 3, 4, 5])
        # First recommendation should NOT be the most sophisticated (5)
        top_item = recs[0][0]
        assert top_item != 5, "brand-new user should not get the most sophisticated item"

    def test_experienced_user_gets_sophisticated_items(self, ranker) -> None:
        """After many positive interactions, user should be recommended harder items."""
        # Train taste up by repeatedly succeeding on increasingly sophisticated items
        for _ in range(15):
            ranker.observe(user_id=7, item_id=2, category="wine", purchased=True, rating=5.0)
            ranker.observe(user_id=7, item_id=3, category="coffee", purchased=True, rating=5.0)
            ranker.observe(user_id=7, item_id=4, category="coffee", purchased=True, rating=5.0)

        recs = ranker.recommend(user_id=7, top_k=3, candidate_item_ids=[0, 1, 2, 3, 4, 5])
        top_ids = [r[0] for r in recs]
        # At least one of the top-3 should be a higher-sophistication item
        assert any(s >= 3 for s in top_ids), (
            f"experienced user should get sophisticated items, got {top_ids}"
        )

    def test_exploration_bonus_for_new_categories(self, ranker) -> None:
        """Items in unexplored categories should get an exploration bonus."""
        # User only interacts with wine
        for _ in range(5):
            ranker.observe(user_id=10, item_id=0, category="wine", purchased=True)

        # Recommend from both wine and coffee
        recs = ranker.recommend(user_id=10, top_k=6, candidate_item_ids=[0, 1, 2, 3, 4, 5])
        top_ids = [r[0] for r in recs]
        # Coffee items (3, 4, 5) should appear (exploration bonus)
        coffee_in_top = [i for i in top_ids if i in {3, 4, 5}]
        assert len(coffee_in_top) > 0, "exploration bonus should surface coffee items"

    def test_from_sophistication_mapper(self) -> None:
        sm = SophisticationMapper.from_prices({1: 10, 2: 50, 3: 100})
        ranker = TasteProgressionRanker(sophistication_scores=sm)
        recs = ranker.recommend(user_id=1, top_k=2, candidate_item_ids=[1, 2, 3])
        assert len(recs) == 2

    def test_requires_candidates_without_recommender(self) -> None:
        ranker = TasteProgressionRanker()
        with pytest.raises(ValueError, match="candidate_item_ids"):
            ranker.recommend(user_id=1, top_k=5)

    def test_repr(self, ranker) -> None:
        r = repr(ranker)
        assert "TasteProgressionRanker" in r
