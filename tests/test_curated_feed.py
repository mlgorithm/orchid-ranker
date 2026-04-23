"""Tests for orchid_ranker.curated_feed -- freshness, topics, reading level, feed ranking."""

from __future__ import annotations

import time

import numpy as np
import pytest

from orchid_ranker.curated_feed import (
    FeedItem,
    FeedRanker,
    FreshnessScorer,
    ReadingLevelEstimator,
    ScoredFeedItem,
    TopicTracker,
)

# =========================================================================
# FreshnessScorer
# =========================================================================


class TestFreshnessScorer:
    """Unit tests for FreshnessScorer."""

    def test_current_item_score_is_one(self) -> None:
        scorer = FreshnessScorer(halflife_hours=24.0)
        now = time.time()
        score = scorer.score(now, now=now)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_halflife_decay(self) -> None:
        scorer = FreshnessScorer(halflife_hours=24.0)
        now = time.time()
        one_halflife_ago = now - 24.0 * 3600.0
        score = scorer.score(one_halflife_ago, now=now)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_old_item_gets_min_score(self) -> None:
        scorer = FreshnessScorer(halflife_hours=1.0, min_score=0.05)
        now = time.time()
        very_old = now - 365 * 24 * 3600.0  # one year ago
        score = scorer.score(very_old, now=now)
        assert score == pytest.approx(0.05)

    def test_batch_scores(self) -> None:
        scorer = FreshnessScorer(halflife_hours=12.0)
        now = time.time()
        timestamps = np.array([
            now,
            now - 12 * 3600,
            now - 24 * 3600,
        ])
        batch = scorer.scores_batch(timestamps, now=now)
        assert batch.shape == (3,)
        # First should be ~1.0, second ~0.5, third ~0.25.
        assert batch[0] == pytest.approx(1.0, abs=1e-6)
        assert batch[1] == pytest.approx(0.5, abs=0.01)
        assert batch[2] == pytest.approx(0.25, abs=0.01)
        # Also verify consistency with single-item scoring.
        for i, ts in enumerate(timestamps):
            single = scorer.score(ts, now=now)
            assert batch[i] == pytest.approx(single, abs=1e-9)

    def test_negative_halflife_raises(self) -> None:
        with pytest.raises(ValueError, match="halflife_hours must be positive"):
            FreshnessScorer(halflife_hours=-1.0)
        with pytest.raises(ValueError, match="halflife_hours must be positive"):
            FreshnessScorer(halflife_hours=0.0)


# =========================================================================
# TopicTracker
# =========================================================================


class TestTopicTracker:
    """Unit tests for TopicTracker."""

    def test_observe_increases_competence(self) -> None:
        tracker = TopicTracker(p_init=0.3)
        initial = tracker.competence(user_id=1, topic="math")
        assert initial == pytest.approx(0.3)
        for _ in range(10):
            tracker.observe(user_id=1, topic="math", engaged=True)
        updated = tracker.competence(user_id=1, topic="math")
        assert updated > initial

    def test_user_profile(self) -> None:
        tracker = TopicTracker()
        tracker.observe(1, "math", engaged=True)
        tracker.observe(1, "science", engaged=True)
        profile = tracker.user_profile(1)
        assert isinstance(profile, dict)
        assert "math" in profile
        assert "science" in profile
        assert all(0.0 <= v <= 1.0 for v in profile.values())
        # Unknown user returns empty dict.
        assert tracker.user_profile(999) == {}

    def test_topic_coverage(self) -> None:
        tracker = TopicTracker(success_threshold=0.5, p_init=0.3, p_transit=0.3)
        all_topics = ["a", "b", "c", "d"]
        # Before any observations, p_init=0.3 < threshold=0.5.
        assert tracker.topic_coverage(1, all_topics) == 0.0
        # Drive topic "a" above threshold with many correct observations.
        for _ in range(20):
            tracker.observe(1, "a", engaged=True)
        coverage = tracker.topic_coverage(1, all_topics)
        # At least topic "a" should be above threshold (1/4 = 0.25).
        assert coverage >= 0.25

    def test_independent_users(self) -> None:
        tracker = TopicTracker()
        for _ in range(10):
            tracker.observe(1, "math", engaged=True)
        comp_user1 = tracker.competence(1, "math")
        comp_user2 = tracker.competence(2, "math")
        # User 2 was never observed, so should still be at p_init.
        assert comp_user1 > comp_user2

    def test_multiple_topics(self) -> None:
        tracker = TopicTracker(p_init=0.3, p_transit=0.1)
        topics = ["algebra", "geometry", "calculus"]
        for topic in topics:
            for _ in range(5):
                tracker.observe(1, topic, engaged=True)
        profile = tracker.user_profile(1)
        assert len(profile) == 3
        for topic in topics:
            assert topic in profile
            assert profile[topic] > 0.3  # should have increased from p_init


# =========================================================================
# ReadingLevelEstimator
# =========================================================================


class TestReadingLevelEstimator:
    """Unit tests for ReadingLevelEstimator."""

    def test_initial_level(self) -> None:
        estimator = ReadingLevelEstimator(initial_level=0.4)
        assert estimator.level(user_id=1) == pytest.approx(0.4)

    def test_level_increases_on_hard_success(self) -> None:
        estimator = ReadingLevelEstimator(alpha=0.3, initial_level=0.5)
        # Engage with a hard item (difficulty=0.9).
        estimator.observe(user_id=1, item_difficulty=0.9, engaged=True)
        new_level = estimator.level(1)
        assert new_level > 0.5, "Level should increase when engaging with harder items"

    def test_level_stable_on_failure(self) -> None:
        estimator = ReadingLevelEstimator(alpha=0.3, initial_level=0.5)
        # Disengage -- level should NOT update.
        estimator.observe(user_id=1, item_difficulty=0.9, engaged=False)
        assert estimator.level(1) == pytest.approx(0.5)

    def test_stretch_zone_around_level(self) -> None:
        estimator = ReadingLevelEstimator(initial_level=0.5)
        lower, upper = estimator.stretch_zone(user_id=1, width=0.15)
        assert lower == pytest.approx(0.35)
        assert upper == pytest.approx(0.65)

    def test_stretch_zone_clamps(self) -> None:
        # User near bottom.
        estimator = ReadingLevelEstimator(initial_level=0.05)
        lower, upper = estimator.stretch_zone(user_id=1, width=0.15)
        assert lower >= 0.0
        assert upper <= 1.0
        assert lower == pytest.approx(0.0)

        # User near top.
        est_high = ReadingLevelEstimator(alpha=1.0, initial_level=0.5)
        est_high.observe(user_id=2, item_difficulty=1.0, engaged=True)
        lower2, upper2 = est_high.stretch_zone(user_id=2, width=0.15)
        assert upper2 <= 1.0


# =========================================================================
# FeedRanker
# =========================================================================


class TestFeedRanker:
    """Unit tests for FeedRanker."""

    @staticmethod
    def _make_items(n: int, topic: str = "tech", base_time: float | None = None) -> list[FeedItem]:
        """Helper to create n synthetic FeedItems."""
        now = base_time if base_time is not None else time.time()
        return [
            FeedItem(
                item_id=i,
                topic=topic,
                difficulty=i / max(n - 1, 1),
                timestamp=now - i * 3600,  # each item 1 hour older
            )
            for i in range(n)
        ]

    def test_rank_returns_correct_count(self) -> None:
        ranker = FeedRanker()
        items = self._make_items(10)
        result = ranker.rank(user_id=1, candidates=items, top_k=5)
        assert len(result) == 5
        assert all(isinstance(r, ScoredFeedItem) for r in result)

    def test_rank_freshest_preferred(self) -> None:
        ranker = FeedRanker(
            w_freshness=1.0,
            w_relevance=0.0,
            w_stretch=0.0,
            w_diversity=0.0,
            w_competence=0.0,
        )
        now = time.time()
        items = [
            FeedItem(item_id=0, topic="a", difficulty=0.5, timestamp=now),
            FeedItem(item_id=1, topic="a", difficulty=0.5, timestamp=now - 48 * 3600),
            FeedItem(item_id=2, topic="a", difficulty=0.5, timestamp=now - 96 * 3600),
        ]
        result = ranker.rank(user_id=1, candidates=items, top_k=3)
        # Freshest item should be ranked first.
        assert result[0].item.item_id == 0
        # Scores should be in descending order.
        scores = [r.freshness_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_diversity(self) -> None:
        ranker = FeedRanker(
            w_diversity=1.0,
            w_relevance=0.0,
            w_freshness=0.0,
            w_stretch=0.0,
            w_competence=0.0,
        )
        now = time.time()
        items = [
            FeedItem(item_id=0, topic="tech", difficulty=0.5, timestamp=now),
            FeedItem(item_id=1, topic="tech", difficulty=0.5, timestamp=now),
            FeedItem(item_id=2, topic="sports", difficulty=0.5, timestamp=now),
            FeedItem(item_id=3, topic="music", difficulty=0.5, timestamp=now),
        ]
        result = ranker.rank(user_id=1, candidates=items, top_k=3)
        topics = [r.item.topic for r in result]
        # With high diversity weight, all 3 results should have different topics.
        assert len(set(topics)) == 3

    def test_observe_updates_state(self) -> None:
        ranker = FeedRanker()
        now = time.time()
        item = FeedItem(item_id=1, topic="biology", difficulty=0.7, timestamp=now)
        # Competence before observation.
        comp_before = ranker.topic_tracker.competence(1, "biology")
        ranker.observe(user_id=1, item=item, engaged=True)
        comp_after = ranker.topic_tracker.competence(1, "biology")
        assert comp_after != comp_before

    def test_rank_with_base_scores(self) -> None:
        ranker = FeedRanker(
            w_relevance=1.0,
            w_freshness=0.0,
            w_stretch=0.0,
            w_diversity=0.0,
            w_competence=0.0,
        )
        now = time.time()
        items = [
            FeedItem(item_id=0, topic="a", difficulty=0.5, timestamp=now),
            FeedItem(item_id=1, topic="b", difficulty=0.5, timestamp=now),
            FeedItem(item_id=2, topic="c", difficulty=0.5, timestamp=now),
        ]
        base_scores = np.array([0.1, 0.9, 0.5])
        result = ranker.rank(user_id=1, candidates=items, base_scores=base_scores, top_k=3)
        # Highest base score (item 1) should rank first.
        assert result[0].item.item_id == 1
        assert result[0].relevance_score == pytest.approx(0.9)

    def test_all_components_combined(self) -> None:
        """Smoke test with all scorers active (default weights)."""
        ranker = FeedRanker()
        now = time.time()
        items = [
            FeedItem(item_id=i, topic=f"topic_{i % 3}", difficulty=i * 0.1, timestamp=now - i * 1800)
            for i in range(10)
        ]
        result = ranker.rank(user_id=42, candidates=items, top_k=5)
        assert len(result) == 5
        # All total_scores should be finite.
        for scored in result:
            assert np.isfinite(scored.total_score)

    def test_empty_candidates(self) -> None:
        ranker = FeedRanker()
        result = ranker.rank(user_id=1, candidates=[], top_k=10)
        assert result == []

    def test_fewer_candidates_than_top_k(self) -> None:
        ranker = FeedRanker()
        items = self._make_items(3)
        result = ranker.rank(user_id=1, candidates=items, top_k=10)
        assert len(result) == 3


# =========================================================================
# Data classes
# =========================================================================


class TestFeedDataClasses:
    """Tests for FeedItem and ScoredFeedItem data classes."""

    def test_feed_item_frozen(self) -> None:
        item = FeedItem(item_id=1, topic="tech", difficulty=0.5, timestamp=1000.0)
        with pytest.raises(AttributeError):
            item.topic = "sports"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            item.item_id = 99  # type: ignore[misc]

    def test_scored_feed_item_has_breakdown(self) -> None:
        item = FeedItem(item_id=1, topic="tech", difficulty=0.5, timestamp=1000.0)
        scored = ScoredFeedItem(
            item=item,
            total_score=0.85,
            relevance_score=0.3,
            freshness_score=0.2,
            stretch_score=0.15,
            diversity_score=0.1,
            competence_score=0.1,
        )
        assert scored.total_score == pytest.approx(0.85)
        assert scored.relevance_score == pytest.approx(0.3)
        assert scored.freshness_score == pytest.approx(0.2)
        assert scored.stretch_score == pytest.approx(0.15)
        assert scored.diversity_score == pytest.approx(0.1)
        assert scored.competence_score == pytest.approx(0.1)
        assert scored.item is item
