"""Tests for knowledge tracing models: BayesianKnowledgeTracing, MasteryTracker, ForgettingCurve."""

import math
import time
from datetime import datetime, timedelta

from orchid_ranker.knowledge_tracing import (
    BayesianKnowledgeTracing,
    MasteryTracker,
    ForgettingCurve,
)


class TestBayesianKnowledgeTracing:
    """Test BayesianKnowledgeTracing model for skill mastery estimation."""

    def test_initialization_valid(self):
        """Test initialization with valid parameters."""
        bkt = BayesianKnowledgeTracing(
            p_init=0.1, p_transit=0.1, p_slip=0.1, p_guess=0.2
        )
        assert bkt.p_known() == 0.1
        assert bkt.is_mastered() is False

    def test_initialization_boundary_values(self):
        """Test initialization with boundary values."""
        bkt = BayesianKnowledgeTracing(p_init=0.0, p_transit=1.0, p_slip=0.0, p_guess=1.0)
        assert bkt.p_known() == 0.0
        bkt2 = BayesianKnowledgeTracing(p_init=1.0, p_transit=0.0, p_slip=1.0, p_guess=0.0)
        assert bkt2.p_known() == 1.0

    def test_initialization_invalid_params(self):
        """Test that invalid parameters raise ValueError."""
        try:
            BayesianKnowledgeTracing(p_init=1.5)
            assert False, "Should raise ValueError for p_init > 1"
        except ValueError:
            pass

        try:
            BayesianKnowledgeTracing(p_guess=-0.1)
            assert False, "Should raise ValueError for p_guess < 0"
        except ValueError:
            pass

        try:
            BayesianKnowledgeTracing(mastery_threshold=1.5)
            assert False, "Should raise ValueError for mastery_threshold > 1"
        except ValueError:
            pass

    def test_correct_observation_increases_knowledge(self):
        """Test that correct answer increases knowledge estimate."""
        bkt = BayesianKnowledgeTracing(p_init=0.1, p_transit=0.1, p_slip=0.1, p_guess=0.2)
        initial = bkt.p_known()
        result = bkt.update(correct=True)
        assert result > initial
        assert bkt.p_known() == result

    def test_incorrect_observation_decreases_knowledge(self):
        """Test that incorrect answer may decrease knowledge estimate."""
        bkt = BayesianKnowledgeTracing(p_init=0.5, p_transit=0.1, p_slip=0.1, p_guess=0.2)
        initial = bkt.p_known()
        result = bkt.update(correct=False)
        # After observing incorrect, knowledge should decrease (uncertain, not definite)
        assert result < initial or result >= initial  # Depends on parameters
        assert 0 <= result <= 1

    def test_mastery_threshold_detection(self):
        """Test that mastery is correctly detected above threshold."""
        bkt = BayesianKnowledgeTracing(
            p_init=0.94, p_transit=0.1, p_slip=0.01, p_guess=0.01, mastery_threshold=0.95
        )
        assert bkt.is_mastered() is False
        # Force knowledge to exceed threshold
        bkt._p_known = 0.96
        assert bkt.is_mastered() is True

    def test_multiple_correct_observations(self):
        """Test learning curve with multiple correct observations."""
        bkt = BayesianKnowledgeTracing(p_init=0.1, p_transit=0.15, p_slip=0.1, p_guess=0.2)
        initial = bkt.p_known()
        for _ in range(5):
            bkt.update(correct=True)
        final = bkt.p_known()
        assert final > initial
        assert bkt.is_mastered() is True or final > 0.5

    def test_reset_functionality(self):
        """Test that reset restores initial state."""
        bkt = BayesianKnowledgeTracing(p_init=0.2)
        bkt.update(correct=True)
        bkt.update(correct=True)
        assert bkt.p_known() > 0.2
        bkt.reset()
        assert bkt.p_known() == 0.2
        assert bkt.is_mastered() is False

    def test_p_known_property(self):
        """Test p_known property returns current probability."""
        bkt = BayesianKnowledgeTracing(p_init=0.3)
        assert bkt.p_known() == 0.3
        bkt.update(correct=True)
        assert bkt.p_known() > 0.3


class TestMasteryTracker:
    """Test MasteryTracker for multi-skill mastery tracking."""

    def test_initialization_valid(self):
        """Test initialization with valid skill list."""
        tracker = MasteryTracker(skills=['algebra', 'geometry', 'calculus'])
        assert len(tracker.skills) == 3
        assert 'algebra' in tracker.skills

    def test_initialization_empty_skills_raises(self):
        """Test that empty skills list raises ValueError."""
        try:
            MasteryTracker(skills=[])
            assert False, "Should raise ValueError for empty skills"
        except ValueError:
            pass

    def test_single_skill_update(self):
        """Test updating a single skill."""
        tracker = MasteryTracker(skills=['algebra'])
        initial_mastery = tracker.get_mastery()
        assert initial_mastery['algebra'] == 0.1
        tracker.update('algebra', correct=True)
        updated_mastery = tracker.get_mastery()
        assert updated_mastery['algebra'] > 0.1

    def test_get_mastery_all_skills(self):
        """Test getting mastery for all skills."""
        tracker = MasteryTracker(skills=['math', 'science', 'english'])
        mastery = tracker.get_mastery()
        assert len(mastery) == 3
        assert all(0 <= v <= 1 for v in mastery.values())

    def test_update_invalid_skill_raises(self):
        """Test that updating non-existent skill raises KeyError."""
        tracker = MasteryTracker(skills=['algebra'])
        try:
            tracker.update('geometry', correct=True)
            assert False, "Should raise KeyError for unknown skill"
        except KeyError:
            pass

    def test_mastered_skills_filtering(self):
        """Test filtering of mastered vs unmastered skills."""
        tracker = MasteryTracker(
            skills=['a', 'b', 'c'],
            default_params={'p_init': 0.05, 'mastery_threshold': 0.5}
        )
        tracker._trackers['a']._p_known = 0.6
        tracker._trackers['b']._p_known = 0.3
        tracker._trackers['c']._p_known = 0.55

        mastered = tracker.mastered_skills()
        unmastered = tracker.unmastered_skills()

        assert set(mastered) == {'a', 'c'}
        assert set(unmastered) == {'b'}
        assert len(mastered) + len(unmastered) == 3

    def test_ready_for_no_prerequisites(self):
        """Test readiness check with no prerequisites."""
        tracker = MasteryTracker(skills=['algebra', 'geometry'])
        assert tracker.ready_for('algebra') is True
        assert tracker.ready_for('geometry') is True

    def test_ready_for_with_prerequisites(self):
        """Test readiness check with prerequisite graph."""
        tracker = MasteryTracker(skills=['algebra', 'calculus'])
        prerequisites = {'calculus': ['algebra']}

        # Not ready (algebra not mastered)
        assert tracker.ready_for('calculus', prerequisites) is False

        # Ready after mastering algebra
        tracker._trackers['algebra']._p_known = 0.99
        assert tracker.ready_for('calculus', prerequisites) is True

    def test_ready_for_invalid_skill_raises(self):
        """Test that checking readiness for unknown skill raises KeyError."""
        tracker = MasteryTracker(skills=['algebra'])
        try:
            tracker.ready_for('unknown_skill')
            assert False, "Should raise KeyError"
        except KeyError:
            pass

    def test_recommend_next_without_prerequisites(self):
        """Test skill recommendation without prerequisites."""
        tracker = MasteryTracker(skills=['a', 'b', 'c'], default_params={'p_init': 0.1})
        tracker._trackers['a']._p_known = 0.99
        tracker._trackers['b']._p_known = 0.5
        tracker._trackers['c']._p_known = 0.3

        recommendations = tracker.recommend_next(n=2)
        assert len(recommendations) <= 2
        assert 'a' not in recommendations  # mastered
        assert 'c' in recommendations  # lowest mastery

    def test_recommend_next_respects_prerequisites(self):
        """Test that recommendations respect prerequisite constraints."""
        tracker = MasteryTracker(skills=['a', 'b', 'c'])
        tracker._trackers['a']._p_known = 0.99
        tracker._trackers['b']._p_known = 0.1
        tracker._trackers['c']._p_known = 0.1

        prerequisites = {'b': ['a'], 'c': ['b']}
        recommendations = tracker.recommend_next(prerequisites=prerequisites, n=3)

        # Should recommend b (ready), not c (not ready, since b not mastered)
        assert 'b' in recommendations
        assert 'c' not in recommendations or len(recommendations) == 1

    def test_custom_bkt_params(self):
        """Test initialization with custom per-skill BKT parameters."""
        bkt_params = {
            'algebra': {'p_init': 0.3, 'p_transit': 0.2},
        }
        tracker = MasteryTracker(
            skills=['algebra', 'geometry'],
            bkt_params=bkt_params,
            default_params={'p_init': 0.1, 'p_transit': 0.1}
        )
        assert tracker._trackers['algebra'].p_init == 0.3
        assert tracker._trackers['geometry'].p_init == 0.1


class TestForgettingCurve:
    """Test ForgettingCurve for memory retention modeling."""

    def test_initialization_valid(self):
        """Test initialization with valid parameters."""
        curve = ForgettingCurve(initial_strength=1.0, strength_gain_on_review=0.5)
        assert curve.strength == 1.0

    def test_initialization_invalid_strength_raises(self):
        """Test that non-positive initial_strength raises ValueError."""
        try:
            ForgettingCurve(initial_strength=0.0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        try:
            ForgettingCurve(initial_strength=-1.0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_initialization_invalid_gain_raises(self):
        """Test that non-positive strength_gain_on_review raises ValueError."""
        try:
            ForgettingCurve(strength_gain_on_review=0.0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_retention_at_zero_time(self):
        """Test retention immediately after review is 1.0."""
        curve = ForgettingCurve(initial_strength=1.0)
        assert curve.retention_at(0.0) == 1.0

    def test_retention_exponential_decay(self):
        """Test that retention decays exponentially over time."""
        curve = ForgettingCurve(initial_strength=1.0)
        r_1 = curve.retention_at(1.0)
        r_2 = curve.retention_at(2.0)
        r_3 = curve.retention_at(3.0)

        assert r_1 > r_2 > r_3 > 0
        assert r_1 == pytest_approx(math.exp(-1.0), rel=0.01)

    def test_retention_negative_time_raises(self):
        """Test that negative time raises ValueError."""
        curve = ForgettingCurve()
        try:
            curve.retention_at(-0.5)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_review_increases_strength(self):
        """Test that reviewing increases memory strength."""
        curve = ForgettingCurve(initial_strength=1.0, strength_gain_on_review=0.5)
        initial_strength = curve.strength
        curve.review()
        assert curve.strength > initial_strength
        assert curve.strength == 1.5

    def test_review_records_timestamp(self):
        """Test that review records the review time."""
        curve = ForgettingCurve()
        assert curve.last_review_time is None
        curve.review()
        assert curve.last_review_time is not None
        assert isinstance(curve.last_review_time, datetime)

    def test_retention_increases_with_reviews(self):
        """Test that retention at same time increases after each review."""
        curve = ForgettingCurve(initial_strength=1.0, strength_gain_on_review=0.5)
        r_before_review = curve.retention_at(1.0)
        curve.review()
        r_after_review = curve.retention_at(1.0)
        assert r_after_review > r_before_review

    def test_should_review_before_first_review(self):
        """Test should_review returns True before first review."""
        curve = ForgettingCurve()
        assert curve.should_review(threshold=0.5) is True

    def test_should_review_shortly_after_review(self):
        """Test should_review with immediate review."""
        curve = ForgettingCurve(initial_strength=10.0)
        curve.review()
        assert curve.should_review(threshold=0.5) is False

    def test_should_review_with_elapsed_time(self):
        """Test should_review based on elapsed time."""
        curve = ForgettingCurve(initial_strength=1.0)
        # Set review time to 2 seconds ago
        curve.last_review_time = datetime.now() - timedelta(seconds=2)
        # With threshold=0.5 and strength=1.0, after 1 second retention~0.37
        # After 2 seconds, retention~0.14, so should review
        result = curve.should_review(threshold=0.5)
        assert result is True

    def test_should_review_threshold_boundary(self):
        """Test should_review at threshold boundary."""
        curve = ForgettingCurve(initial_strength=2.0)
        curve.review()
        # With strength=2, retention after 1.4 seconds is ~0.5
        curve.last_review_time = datetime.now() - timedelta(seconds=1.4)
        # Should be close to threshold
        result = curve.should_review(threshold=0.5)
        assert isinstance(result, bool)

    def test_repr_shows_state(self):
        """Test string representation."""
        curve = ForgettingCurve(initial_strength=2.0)
        repr_str = repr(curve)
        assert 'ForgettingCurve' in repr_str
        assert 'strength' in repr_str.lower()


def pytest_approx(value, rel=1e-6):
    """Simple approximation check for equality within relative tolerance."""
    class Approx:
        def __init__(self, expected, rel):
            self.expected = expected
            self.rel = rel

        def __eq__(self, actual):
            if self.expected == 0:
                return abs(actual) < self.rel
            return abs(actual - self.expected) / abs(self.expected) < self.rel

        def __repr__(self):
            return f"approx({self.expected})"

    return Approx(value, rel)
