"""Tests for educational evaluation metrics: learning_gain, knowledge_coverage, etc."""

from orchid_ranker.evaluation import (
    learning_gain,
    knowledge_coverage,
    curriculum_adherence,
    difficulty_appropriateness,
    engagement_score,
)


class TestLearningGain:
    """Test learning_gain metric."""

    def test_learning_gain_zero_initial(self):
        """Test learning gain when starting from zero."""
        gain = learning_gain(pre_score=0.0, post_score=0.5)
        assert gain == 0.5

    def test_learning_gain_no_improvement(self):
        """Test learning gain with no improvement."""
        gain = learning_gain(pre_score=0.5, post_score=0.5)
        assert gain == 0.0

    def test_learning_gain_perfect_improvement(self):
        """Test learning gain with perfect improvement."""
        gain = learning_gain(pre_score=0.5, post_score=1.0)
        assert gain == 1.0

    def test_learning_gain_already_perfect(self):
        """Test learning gain when already perfect."""
        gain = learning_gain(pre_score=1.0, post_score=1.0)
        assert gain == 0.0

    def test_learning_gain_decline(self):
        """Test learning gain with negative change (decline)."""
        gain = learning_gain(pre_score=0.8, post_score=0.5)
        assert gain < 0

    def test_learning_gain_midpoint(self):
        """Test learning gain at midpoint."""
        gain = learning_gain(pre_score=0.25, post_score=0.75)
        assert abs(gain - 2.0 / 3.0) < 0.01

    def test_learning_gain_bounds(self):
        """Test that learning gain can reach 1.0 (perfect improvement)."""
        gain = learning_gain(pre_score=0.1, post_score=1.0)
        assert gain == 1.0


class TestKnowledgeCoverage:
    """Test knowledge_coverage metric."""

    def test_coverage_empty_mastered(self):
        """Test coverage with no mastered skills."""
        coverage = knowledge_coverage(mastered_skills=set(), total_skills={'a', 'b', 'c'})
        assert coverage == 0.0

    def test_coverage_all_mastered(self):
        """Test coverage with all skills mastered."""
        coverage = knowledge_coverage(mastered_skills={'a', 'b', 'c'}, total_skills={'a', 'b', 'c'})
        assert coverage == 1.0

    def test_coverage_partial(self):
        """Test coverage with some skills mastered."""
        coverage = knowledge_coverage(mastered_skills={'a', 'b'}, total_skills={'a', 'b', 'c', 'd'})
        assert coverage == 0.5

    def test_coverage_empty_total_skills(self):
        """Test coverage when total skills is empty."""
        coverage = knowledge_coverage(mastered_skills={'a'}, total_skills=set())
        assert coverage == 0.0

    def test_coverage_mastered_outside_total(self):
        """Test coverage when mastered includes skills not in total."""
        coverage = knowledge_coverage(mastered_skills={'a', 'b', 'x'}, total_skills={'a', 'b', 'c'})
        # Only count intersection
        assert coverage == 2.0 / 3.0

    def test_coverage_single_skill(self):
        """Test coverage with single skill."""
        coverage = knowledge_coverage(mastered_skills={'a'}, total_skills={'a'})
        assert coverage == 1.0

    def test_coverage_from_lists(self):
        """Test coverage accepts lists as well as sets."""
        coverage = knowledge_coverage(mastered_skills=['a', 'b'], total_skills=['a', 'b', 'c'])
        assert coverage == 2.0 / 3.0


class TestCurriculumAdherence:
    """Test curriculum_adherence metric."""

    def test_adherence_empty_recommendations(self):
        """Test adherence with no recommendations."""
        adherence = curriculum_adherence([], {}, set())
        assert adherence == 1.0

    def test_adherence_all_satisfied(self):
        """Test adherence when all prerequisites are met."""
        recs = [1, 2, 3]
        prereqs = {1: set(), 2: {1}, 3: {1, 2}}
        mastered = {1, 2}
        adherence = curriculum_adherence(recs, prereqs, mastered)
        assert adherence == 1.0

    def test_adherence_none_satisfied(self):
        """Test adherence when no prerequisites are met."""
        recs = [2, 3]
        prereqs = {2: {1}, 3: {1, 2}}
        mastered = set()
        adherence = curriculum_adherence(recs, prereqs, mastered)
        assert adherence == 0.0

    def test_adherence_partial(self):
        """Test adherence with partial satisfaction."""
        recs = [2, 3]
        prereqs = {2: {1}, 3: {1, 2}}
        mastered = {1}
        adherence = curriculum_adherence(recs, prereqs, mastered)
        # Item 2 is OK (prereq 1 mastered), item 3 is not (needs 2 which is not mastered)
        assert adherence == 0.5

    def test_adherence_no_prerequisites(self):
        """Test adherence when items have no prerequisites."""
        recs = [1, 2, 3]
        prereqs = {}
        mastered = set()
        adherence = curriculum_adherence(recs, prereqs, mastered)
        assert adherence == 1.0

    def test_adherence_with_unmastered_prereqs(self):
        """Test adherence with some unmastered prerequisites."""
        recs = [3]
        prereqs = {3: {1, 2}}
        mastered = {1}
        # One of two prerequisites not met
        adherence = curriculum_adherence(recs, prereqs, mastered)
        assert adherence == 0.0

    def test_adherence_order_independent(self):
        """Test that adherence order doesn't matter."""
        recs1 = [1, 2, 3]
        recs2 = [3, 1, 2]
        prereqs = {1: set(), 2: {1}, 3: {1, 2}}
        mastered = {1, 2}

        adh1 = curriculum_adherence(recs1, prereqs, mastered)
        adh2 = curriculum_adherence(recs2, prereqs, mastered)
        assert adh1 == adh2


class TestDifficultyAppropriateness:
    """Test difficulty_appropriateness metric (Zone of Proximal Development)."""

    def test_zpd_empty_recommendations(self):
        """Test ZPD with no recommendations."""
        zpd = difficulty_appropriateness([], student_ability=0.5)
        assert zpd == 1.0

    def test_zpd_all_in_zone(self):
        """Test ZPD when all items are in zone."""
        difficulties = [0.5, 0.6, 0.7]
        zpd = difficulty_appropriateness(difficulties, student_ability=0.5, zpd_width=0.25)
        # All in [0.5, 0.75]
        assert zpd == 1.0

    def test_zpd_none_in_zone(self):
        """Test ZPD when no items are in zone."""
        difficulties = [0.0, 0.1, 0.2]
        zpd = difficulty_appropriateness(difficulties, student_ability=0.8, zpd_width=0.25)
        # All below [0.8, 1.05]
        assert zpd == 0.0

    def test_zpd_partial_in_zone(self):
        """Test ZPD with partial coverage."""
        difficulties = [0.5, 0.6, 0.8]
        zpd = difficulty_appropriateness(difficulties, student_ability=0.5, zpd_width=0.25)
        # 0.5 and 0.6 in [0.5, 0.75], 0.8 is outside
        assert abs(zpd - 2.0 / 3.0) < 0.01

    def test_zpd_narrow_zone(self):
        """Test ZPD with narrow zone."""
        difficulties = [0.5, 0.55, 0.6, 0.65]
        zpd = difficulty_appropriateness(difficulties, student_ability=0.5, zpd_width=0.1)
        # [0.5, 0.6] zone contains 0.5, 0.55, and 0.6 (inclusive)
        assert abs(zpd - 0.75) < 0.01

    def test_zpd_wide_zone(self):
        """Test ZPD with wide zone."""
        difficulties = [0.3, 0.5, 0.7, 0.9]
        zpd = difficulty_appropriateness(difficulties, student_ability=0.5, zpd_width=0.5)
        # [0.5, 1.0] contains 0.5, 0.7, 0.9
        assert zpd == 0.75

    def test_zpd_boundary_cases(self):
        """Test ZPD at zone boundaries."""
        difficulties = [0.5, 0.75]
        zpd = difficulty_appropriateness(difficulties, student_ability=0.5, zpd_width=0.25)
        # [0.5, 0.75] includes both boundaries
        assert zpd == 1.0

    def test_zpd_default_width(self):
        """Test ZPD with default width."""
        difficulties = [0.5, 0.6, 0.7]
        zpd = difficulty_appropriateness(difficulties, student_ability=0.5)
        # Default width is 0.25, so zone is [0.5, 0.75]
        # All three in zone: 0.5, 0.6, 0.7 all <= 0.75
        assert abs(zpd - 1.0) < 0.01


class TestEngagementScore:
    """Test engagement_score metric."""

    def test_engagement_no_interactions(self):
        """Test engagement with no interactions."""
        engagement = engagement_score(interactions=set(), total_available=10)
        assert engagement == 0.0

    def test_engagement_all_interact(self):
        """Test engagement when all items interact."""
        engagement = engagement_score(interactions={1, 2, 3, 4, 5}, total_available=5)
        assert engagement == 1.0

    def test_engagement_partial(self):
        """Test engagement with partial interactions."""
        engagement = engagement_score(interactions={1, 2}, total_available=5)
        assert engagement == 0.4

    def test_engagement_zero_available(self):
        """Test engagement with zero available items."""
        engagement = engagement_score(interactions={1, 2}, total_available=0)
        assert engagement == 0.0

    def test_engagement_negative_available(self):
        """Test engagement with negative available (should be treated as 0)."""
        engagement = engagement_score(interactions={1, 2}, total_available=-1)
        assert engagement == 0.0

    def test_engagement_from_list(self):
        """Test engagement accepts lists."""
        engagement = engagement_score(interactions=[1, 2, 3], total_available=10)
        assert engagement == 0.3

    def test_engagement_from_set(self):
        """Test engagement accepts sets."""
        engagement = engagement_score(interactions={1, 2, 3}, total_available=10)
        assert engagement == 0.3

    def test_engagement_duplicate_interactions(self):
        """Test that engagement deduplicates interactions."""
        # When passed as list with duplicates, should be converted to set
        engagement = engagement_score(interactions=[1, 1, 2, 2], total_available=5)
        # Should count {1, 2} = 2 unique items
        assert engagement == 0.4


class TestMetricsIntegration:
    """Integration tests combining multiple metrics."""

    def test_comprehensive_educational_profile(self):
        """Test a comprehensive set of educational metrics together."""
        # Scenario: student improved from 0.3 to 0.7
        gain = learning_gain(0.3, 0.7)

        # Student mastered 7 out of 10 skills
        coverage = knowledge_coverage({'s1', 's2', 's3', 's4', 's5', 's6', 's7'},
                                      {'s' + str(i) for i in range(1, 11)})

        # Recommended 5 items, 4 had prerequisites met
        recs = [1, 2, 3, 4, 5]
        prereqs = {1: set(), 2: {1}, 3: {1, 2}, 4: {1, 2, 3}, 5: {1, 2, 3, 4}}
        mastered = {1, 2, 3}
        adherence = curriculum_adherence(recs, prereqs, mastered)

        # Student ability 0.5, items at [0.4, 0.6, 0.7, 0.8, 0.9]
        zpd = difficulty_appropriateness([0.4, 0.6, 0.7, 0.8, 0.9], 0.5, zpd_width=0.25)

        # Student engaged with 3 out of 5 recommended items
        engagement = engagement_score({1, 2, 3}, 5)

        # All should be valid scores
        assert 0 <= gain <= 2.0
        assert 0 <= coverage <= 1.0
        assert 0 <= adherence <= 1.0
        assert 0 <= zpd <= 1.0
        assert 0 <= engagement <= 1.0

    def test_metric_edge_cases(self):
        """Test metrics with edge cases and boundary values."""
        # All metrics should handle extreme inputs gracefully
        assert learning_gain(0.0, 1.0) > 0
        assert knowledge_coverage(set(), {'a'}) == 0.0
        assert curriculum_adherence([], {}, set()) == 1.0
        assert difficulty_appropriateness([], 0.5) == 1.0
        assert engagement_score(set(), 0) == 0.0

    def test_metric_consistency(self):
        """Test that metrics are consistent across similar inputs."""
        # Knowledge coverage should be symmetric in set operations
        cov1 = knowledge_coverage({'a', 'b'}, {'a', 'b', 'c'})
        cov2 = knowledge_coverage({'a', 'b'}, {'b', 'c', 'a'})
        assert cov1 == cov2

        # Engagement should be the same whether passed as list or set
        eng1 = engagement_score([1, 2, 3], 10)
        eng2 = engagement_score({1, 2, 3}, 10)
        assert eng1 == eng2

        # Learning gain should be the same regardless of score representation
        gain1 = learning_gain(0.5, 0.7)
        gain2 = learning_gain(float(0.5), float(0.7))
        assert gain1 == gain2
