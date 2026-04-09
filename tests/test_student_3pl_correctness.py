"""Correctness tests for StudentAgent 3PL model and learning dynamics.

Tests with known expected values for:
- 3PL formula: p = c + (1-c-s)*sigmoid(a*(theta-difficulty) - 0.5*fatigue)
- Learning dynamics and convergence
- Fatigue, trust, and engagement update formulas
- Reward calculation
- ZPD match score bell curve
- Knowledge decay via forgetting
"""
import sys
sys.path.insert(0, "src")

import math
import numpy as np
import pytest

from orchid_ranker.agents.student_agent import (
    StudentAgent,
    ItemMeta,
)


class Test3PLFormula:
    """Test correctness of the 3PL probability formula."""

    def test_3pl_bounds(self):
        """Test that 3PL probability is always in [c, 1-s]."""
        agent = StudentAgent(user_id=1, seed=42)
        # With known c and s
        c, s = agent.c, agent.s
        expected_min = c
        expected_max = 1.0 - s

        # Test various ability/difficulty combinations
        for theta in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for diff in [0.0, 0.3, 0.5, 0.7, 1.0]:
                p = agent._prob_correct_3pl(theta, diff)
                assert expected_min <= p <= expected_max, \
                    f"3PL out of bounds: p={p}, expected [{expected_min}, {expected_max}]"

    def test_3pl_no_fatigue_vs_with_fatigue(self):
        """Test that higher fatigue decreases P(correct)."""
        agent = StudentAgent(user_id=2, seed=42)
        theta, diff = 0.6, 0.4

        # Compute with no fatigue
        agent.fatigue = 0.0
        p_no_fatigue = agent._prob_correct_3pl(theta, diff)

        # Compute with fatigue
        agent.fatigue = 0.8
        p_with_fatigue = agent._prob_correct_3pl(theta, diff)

        assert p_with_fatigue < p_no_fatigue, \
            "Higher fatigue should reduce P(correct)"

    def test_3pl_higher_ability_increases_probability(self):
        """Test that higher ability theta increases P(correct)."""
        agent = StudentAgent(user_id=3, seed=42)
        agent.fatigue = 0.3  # moderate fatigue
        diff = 0.5

        p_low = agent._prob_correct_3pl(0.2, diff)
        p_mid = agent._prob_correct_3pl(0.5, diff)
        p_high = agent._prob_correct_3pl(0.8, diff)

        assert p_low < p_mid < p_high, \
            "Higher ability should increase P(correct)"

    def test_3pl_higher_difficulty_decreases_probability(self):
        """Test that higher difficulty decreases P(correct)."""
        agent = StudentAgent(user_id=4, seed=42)
        agent.fatigue = 0.2
        theta = 0.5

        p_easy = agent._prob_correct_3pl(theta, 0.1)
        p_mid = agent._prob_correct_3pl(theta, 0.5)
        p_hard = agent._prob_correct_3pl(theta, 0.9)

        assert p_easy > p_mid > p_hard, \
            "Higher difficulty should decrease P(correct)"

    def test_3pl_formula_explicit(self):
        """Test 3PL formula matches explicit computation."""
        agent = StudentAgent(user_id=5, seed=42)
        theta, diff = 0.6, 0.4
        agent.fatigue = 0.3

        # Get actual value
        p_actual = agent._prob_correct_3pl(theta, diff)

        # Compute expected value explicitly
        c, s, a = agent.c, agent.s, agent.a
        beta_fatigue = 0.5
        logit = a * (theta - diff) - beta_fatigue * agent.fatigue
        sig = 1.0 / (1.0 + math.exp(-logit))
        p_expected = np.clip(c + (1.0 - c - s) * sig, 0.0, 1.0)

        assert p_actual == pytest.approx(p_expected, abs=1e-10)

    def test_3pl_guessing_floor(self):
        """Test that P(correct) >= c even with very low ability."""
        agent = StudentAgent(user_id=6, seed=42)
        c = agent.c

        # Even with very low ability and high difficulty
        p = agent._prob_correct_3pl(0.0, 1.0)
        assert p >= c - 1e-10, \
            "P(correct) should never fall below guessing floor c"

    def test_3pl_slip_ceiling(self):
        """Test that P(correct) <= 1-s even with very high ability."""
        agent = StudentAgent(user_id=7, seed=42)
        s = agent.s

        # Even with very high ability and low difficulty
        p = agent._prob_correct_3pl(1.0, 0.0)
        assert p <= (1.0 - s) + 1e-10, \
            "P(correct) should never exceed ceiling 1-s"


class TestMonotonicityProperties:
    """Test monotonicity properties of ability, fatigue, and difficulty."""

    def test_monotonicity_ability_increases_p_correct(self):
        """Test that increasing ability monotonically increases P(correct)."""
        agent = StudentAgent(user_id=10, seed=42)
        agent.fatigue = 0.2
        diff = 0.5

        thetas = np.linspace(0.0, 1.0, 11)
        probs = [agent._prob_correct_3pl(float(t), diff) for t in thetas]

        # Check monotonicity
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i+1] + 1e-10, \
                f"P(correct) not monotonic in ability at theta={thetas[i]}"

    def test_monotonicity_fatigue_decreases_p_correct(self):
        """Test that increasing fatigue monotonically decreases P(correct)."""
        agent = StudentAgent(user_id=11, seed=42)
        theta, diff = 0.6, 0.4

        fatigues = np.linspace(0.0, 1.0, 11)
        probs = []
        for f in fatigues:
            agent.fatigue = float(f)
            probs.append(agent._prob_correct_3pl(theta, diff))

        # Check monotonicity
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i+1] - 1e-10, \
                f"P(correct) not monotonically decreasing with fatigue={fatigues[i]}"

    def test_monotonicity_difficulty_decreases_p_correct(self):
        """Test that increasing difficulty monotonically decreases P(correct)."""
        agent = StudentAgent(user_id=12, seed=42)
        agent.fatigue = 0.3
        theta = 0.5

        diffs = np.linspace(0.0, 1.0, 11)
        probs = [agent._prob_correct_3pl(theta, float(d)) for d in diffs]

        # Check monotonicity
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i+1] - 1e-10, \
                f"P(correct) not monotonically decreasing with difficulty={diffs[i]}"


class TestLearningDynamicsConvergence:
    """Test learning dynamics and convergence to high knowledge."""

    def test_repeated_correct_increases_knowledge(self):
        """Test that repeated correct answers increase knowledge toward 1.0."""
        agent = StudentAgent(
            user_id=20,
            knowledge_mode="scalar",
            lr=0.3,
            decay=0.0,  # disable decay for this test
            seed=42,
        )
        initial_knowledge = agent.knowledge

        # Simulate multiple correct answers (manually)
        for _ in range(5):
            p_correct = 0.8  # mock probability
            correct = 1  # all correct
            agent._apply_learning_update(
                item_id=0,
                correct=correct,
                p_correct=p_correct,
                items_meta=None,
            )

        final_knowledge = agent.knowledge
        assert final_knowledge > initial_knowledge, \
            "Knowledge should increase after correct answers"

    def test_learning_formula_scalar(self):
        """Test scalar learning update formula: knowledge = (1-decay)*k + lr*(correct - p_correct)."""
        agent = StudentAgent(
            user_id=21,
            knowledge_mode="scalar",
            lr=0.2,
            decay=0.1,
            seed=42,
        )
        initial_k = agent.knowledge

        # Manually apply learning with known values
        correct = 1
        p_correct = 0.5
        lr = agent.lr
        decay = agent.decay

        # Set RNG seed to control noise
        agent.rng.seed(100)
        agent._apply_learning_update(0, correct, p_correct, None)

        # Expected: (1 - 0.1) * initial_k + 0.2 * (1 - 0.5) + noise
        # noise is small, so we check approximately
        expected_delta = lr * (correct - p_correct)  # 0.2 * 0.5 = 0.1
        expected_k = (1 - decay) * initial_k + expected_delta

        # Allow for noise term
        assert abs(agent.knowledge - expected_k) < 0.02, \
            f"Knowledge update doesn't match formula. got {agent.knowledge}, expected ≈{expected_k}"

    def test_learning_vector_mode(self):
        """Test vector mode learning applies per-skill updates."""
        agent = StudentAgent(
            user_id=22,
            knowledge_dim=3,
            knowledge_mode="vector",
            lr=0.3,
            decay=0.05,
            seed=42,
        )
        initial_k = np.array(agent.knowledge)

        # Apply learning with specific skills
        skills = [True, False, True]  # affects dimensions 0 and 2
        meta = ItemMeta(difficulty=0.5, skills=skills)
        items_meta = {0: meta}

        agent.rng.seed(100)
        agent._apply_learning_update(0, 1, 0.4, items_meta)

        final_k = np.array(agent.knowledge)

        # Dimensions 0 and 2 should change, dimension 1 should stay similar
        assert final_k[0] != pytest.approx(initial_k[0]), \
            "Skills[0] should be updated"
        assert final_k[1] == pytest.approx(initial_k[1] * (1 - 0.05), abs=1e-10), \
            "Non-skill dimension should only decay"
        assert final_k[2] != pytest.approx(initial_k[2]), \
            "Skills[2] should be updated"


class TestFatigueGrowthFormula:
    """Test fatigue growth: fatigue += fatigue_growth * num_items * avg_difficulty * factor."""

    def test_fatigue_growth_with_single_item(self):
        """Test fatigue growth with one item."""
        agent = StudentAgent(
            user_id=30,
            fatigue_growth=0.05,
            fatigue_recovery=0.0,  # disable recovery
            trust_influence=False,
            seed=42,
        )
        agent.fatigue = 0.1
        initial_fatigue = agent.fatigue

        # One item, difficulty 0.6
        feedback = {0: 1}
        items_meta = {0: ItemMeta(difficulty=0.6)}

        agent._update_latents_after_round(feedback, items_meta)

        # Expected: 0.1 + 0.05 * 1 * 0.6 * 1.0 = 0.1 + 0.03 = 0.13
        expected = initial_fatigue + 0.05 * 1 * 0.6 * 1.0
        assert agent.fatigue == pytest.approx(expected, abs=1e-10)

    def test_fatigue_growth_multiple_items(self):
        """Test fatigue growth with multiple items."""
        agent = StudentAgent(
            user_id=31,
            fatigue_growth=0.05,
            fatigue_recovery=0.0,
            trust_influence=False,
            seed=42,
        )
        agent.fatigue = 0.1

        # Three items: difficulties 0.4, 0.5, 0.6
        feedback = {0: 1, 1: 0, 2: 1}
        items_meta = {
            0: ItemMeta(difficulty=0.4),
            1: ItemMeta(difficulty=0.5),
            2: ItemMeta(difficulty=0.6),
        }

        agent._update_latents_after_round(feedback, items_meta)

        avg_diff = (0.4 + 0.5 + 0.6) / 3
        expected = 0.1 + 0.05 * 3 * avg_diff * 1.0
        assert agent.fatigue == pytest.approx(expected, abs=1e-10)

    def test_fatigue_growth_with_trust_influence(self):
        """Test fatigue growth factor depends on trust when trust_influence=True."""
        agent1 = StudentAgent(
            user_id=32,
            fatigue_growth=0.05,
            fatigue_recovery=0.0,
            trust_influence=True,
            seed=42,
        )
        agent2 = StudentAgent(
            user_id=33,
            fatigue_growth=0.05,
            fatigue_recovery=0.0,
            trust_influence=True,
            seed=42,
        )

        agent1.fatigue = 0.1
        agent1.trust = 0.2  # low trust
        agent2.fatigue = 0.1
        agent2.trust = 0.9  # high trust

        feedback = {0: 1}
        items_meta = {0: ItemMeta(difficulty=0.5)}

        agent1._update_latents_after_round(feedback, items_meta)
        agent2._update_latents_after_round(feedback, items_meta)

        # factor = (1 - 0.3 * trust)
        # agent1: factor = 1 - 0.3*0.2 = 0.94
        # agent2: factor = 1 - 0.3*0.9 = 0.73
        expected1 = 0.1 + 0.05 * 1 * 0.5 * 0.94
        expected2 = 0.1 + 0.05 * 1 * 0.5 * 0.73

        assert agent1.fatigue == pytest.approx(expected1, abs=1e-10)
        assert agent2.fatigue == pytest.approx(expected2, abs=1e-10)


class TestTrustUpdateFormula:
    """Test trust update: trust += 0.05 * (accuracy - 0.5)."""

    def test_trust_increases_with_high_accuracy(self):
        """Test trust increases when accuracy > 0.5."""
        agent = StudentAgent(user_id=40, trust_influence=False, seed=42)
        agent.trust = 0.5

        # All correct
        feedback = {0: 1, 1: 1}
        agent._update_latents_after_round(feedback, None)

        # accuracy = 1.0, delta = 0.05 * (1.0 - 0.5) = 0.025
        expected = 0.5 + 0.05 * (1.0 - 0.5)
        assert agent.trust == pytest.approx(expected, abs=1e-10)

    def test_trust_decreases_with_low_accuracy(self):
        """Test trust decreases when accuracy < 0.5."""
        agent = StudentAgent(user_id=41, trust_influence=False, seed=42)
        agent.trust = 0.5

        # All incorrect
        feedback = {0: 0, 1: 0}
        agent._update_latents_after_round(feedback, None)

        # accuracy = 0.0, delta = 0.05 * (0.0 - 0.5) = -0.025
        expected = 0.5 + 0.05 * (0.0 - 0.5)
        assert agent.trust == pytest.approx(expected, abs=1e-10)

    def test_trust_stays_same_with_50_percent_accuracy(self):
        """Test trust unchanged when accuracy = 0.5."""
        agent = StudentAgent(user_id=42, trust_influence=False, seed=42)
        agent.trust = 0.5

        # One correct, one incorrect
        feedback = {0: 1, 1: 0}
        agent._update_latents_after_round(feedback, None)

        assert agent.trust == pytest.approx(0.5, abs=1e-10)

    def test_trust_clamped_to_bounds(self):
        """Test trust is clamped to [0, 1]."""
        agent = StudentAgent(user_id=43, trust_influence=False, seed=42)

        # Start near 1, try to increase
        agent.trust = 0.99
        feedback = {0: 1, 1: 1}
        agent._update_latents_after_round(feedback, None)
        assert agent.trust <= 1.0

        # Start near 0, try to decrease
        agent.trust = 0.01
        feedback = {0: 0, 1: 0}
        agent._update_latents_after_round(feedback, None)
        assert agent.trust >= 0.0


class TestEngagementUpdateFormula:
    """Test engagement update with and without trust_influence."""

    def test_engagement_without_trust_influence(self):
        """Test engagement update without trust_influence.

        Formula: engagement += 0.1*(acc-0.5) - 0.05*fatigue, clamped to [0.2, 1.2]
        """
        agent = StudentAgent(user_id=50, trust_influence=False, seed=42)
        agent.engagement = 0.8
        agent.fatigue = 0.2

        feedback = {0: 1, 1: 1}  # acc = 1.0
        agent._update_latents_after_round(feedback, None)

        # delta = 0.1*(1.0-0.5) - 0.05*0.2 = 0.05 - 0.01 = 0.04
        expected = 0.8 + 0.04
        # Note: there's an additional +0.04 coupling term
        expected += 0.04 * (1.0 - 0.5)
        assert agent.engagement == pytest.approx(expected, abs=1e-10)

    def test_engagement_with_trust_influence(self):
        """Test engagement update with trust_influence=True.

        Formula: engagement += 0.1*(acc-0.5) - 0.05*fatigue + 0.05*(trust-0.5)
        """
        agent = StudentAgent(user_id=51, trust_influence=True, seed=42)
        agent.engagement = 0.8
        agent.fatigue = 0.2
        agent.trust = 0.7

        feedback = {0: 1, 1: 0}  # acc = 0.5
        agent._update_latents_after_round(feedback, None)

        # First update: 0.8 + 0.1*0.0 - 0.05*0.2 + 0.05*(0.7-0.5)
        #            = 0.8 + 0 - 0.01 + 0.01 = 0.8
        # Then trust update affects engagement
        # acc=0.5, so trust update is 0.05*(0.5-0.5)=0 (no change to trust)
        # Second coupling: + 0.04*(acc-0.5) = 0

        expected = 0.8 + 0.1*(0.5-0.5) - 0.05*0.2 + 0.05*(0.7-0.5)
        expected += 0.04 * (0.5 - 0.5)
        assert agent.engagement == pytest.approx(expected, abs=1e-10)

    def test_engagement_clamped_to_bounds(self):
        """Test engagement is clamped to [0.2, 1.2]."""
        agent = StudentAgent(user_id=52, trust_influence=False, seed=42)

        # Try to go below 0.2
        agent.engagement = 0.3
        agent.fatigue = 0.9
        feedback = {0: 0}  # all incorrect
        agent._update_latents_after_round(feedback, None)
        assert agent.engagement >= 0.2

        # Try to go above 1.2
        agent.engagement = 1.1
        agent.fatigue = 0.0
        feedback = {0: 1}  # all correct
        agent._update_latents_after_round(feedback, None)
        assert agent.engagement <= 1.2


class TestRewardFormula:
    """Test reward = 0.60*accuracy + 0.25*(1-fatigue) + 0.15*engagement."""

    def test_reward_formula_explicit(self):
        """Test reward formula matches explicit calculation."""
        agent = StudentAgent(user_id=60, seed=42)
        agent.fatigue = 0.3
        agent.engagement = 0.9

        feedback = {0: 1, 1: 1, 2: 0}  # acc = 2/3

        actual_reward = agent.reward(feedback)

        acc = 2.0 / 3.0
        expected = np.clip(
            0.60 * acc + 0.25 * (1.0 - 0.3) + 0.15 * 0.9,
            0.0,
            1.2
        )

        assert actual_reward == pytest.approx(expected, abs=1e-10)

    def test_reward_empty_feedback(self):
        """Test reward with empty feedback (acc=0)."""
        agent = StudentAgent(user_id=61, seed=42)
        agent.fatigue = 0.4
        agent.engagement = 1.0

        reward = agent.reward({})

        expected = np.clip(
            0.60 * 0.0 + 0.25 * (1.0 - 0.4) + 0.15 * 1.0,
            0.0,
            1.2
        )
        assert reward == pytest.approx(expected, abs=1e-10)

    def test_reward_perfect_conditions(self):
        """Test reward with perfect accuracy, zero fatigue, high engagement."""
        agent = StudentAgent(user_id=62, seed=42)
        agent.fatigue = 0.0
        agent.engagement = 1.2

        feedback = {0: 1, 1: 1}  # acc = 1.0
        reward = agent.reward(feedback)

        expected = 0.60 * 1.0 + 0.25 * 1.0 + 0.15 * 1.2
        assert reward == pytest.approx(expected, abs=1e-10)

    def test_reward_poor_conditions(self):
        """Test reward with poor accuracy, high fatigue, low engagement."""
        agent = StudentAgent(user_id=63, seed=42)
        agent.fatigue = 1.0
        agent.engagement = 0.2

        feedback = {0: 0, 1: 0}  # acc = 0.0
        reward = agent.reward(feedback)

        expected = np.clip(
            0.60 * 0.0 + 0.25 * 0.0 + 0.15 * 0.2,
            0.0,
            1.2
        )
        assert reward == pytest.approx(expected, abs=1e-10)


class TestZPDMatchScore:
    """Test ZPD match score bell curve shape."""

    def test_zpd_peak_at_target(self):
        """Test ZPD score peaks at target difficulty (theta + delta)."""
        agent = StudentAgent(user_id=70, zpd_delta=0.1, zpd_width=0.25, seed=42)
        theta = 0.5
        delta = agent.zpd_delta
        width = agent.zpd_width

        target = theta + delta  # 0.6

        # Score at target should be maximum
        score_at_target = agent._zpd_match_score(theta, target)

        # Scores slightly off target should be lower
        score_below = agent._zpd_match_score(theta, target - 0.05)
        score_above = agent._zpd_match_score(theta, target + 0.05)

        assert score_at_target >= score_below, \
            "ZPD score should be higher at target"
        assert score_at_target >= score_above, \
            "ZPD score should be higher at target"

    def test_zpd_symmetric_falloff(self):
        """Test ZPD score falls off symmetrically around target."""
        agent = StudentAgent(user_id=71, zpd_delta=0.1, zpd_width=0.25, seed=42)
        theta = 0.5
        target = theta + agent.zpd_delta

        # Scores at equal distances from target should be approximately equal
        score_minus = agent._zpd_match_score(theta, target - 0.1)
        score_plus = agent._zpd_match_score(theta, target + 0.1)

        assert score_minus == pytest.approx(score_plus, abs=1e-10), \
            "ZPD should be symmetric around target"

    def test_zpd_far_from_target_low_score(self):
        """Test ZPD score is low far from target."""
        agent = StudentAgent(user_id=72, zpd_delta=0.1, zpd_width=0.25, seed=42)
        theta = 0.5
        target = theta + agent.zpd_delta

        # Very far from target
        score_at_target = agent._zpd_match_score(theta, target)
        score_far = agent._zpd_match_score(theta, target + 1.0)

        assert score_far < 0.1 * score_at_target, \
            "Score far from target should be much lower"

    def test_zpd_custom_delta_and_width(self):
        """Test ZPD with custom delta and width parameters."""
        agent = StudentAgent(user_id=73, zpd_delta=0.2, zpd_width=0.3, seed=42)
        theta = 0.4

        # With delta=0.2, width=0.3
        score_at_optimal = agent._zpd_match_score(theta, theta + 0.2)
        assert score_at_optimal > 0.9, \
            "Score at optimal should be high"


class TestKnowledgeDecay:
    """Test knowledge decay via forgetting_rate."""

    def test_scalar_knowledge_decays(self):
        """Test scalar knowledge decays by forgetting_rate each round."""
        agent = StudentAgent(
            user_id=80,
            knowledge_mode="scalar",
            forgetting_rate=0.1,
            seed=42,
        )
        agent.knowledge = 0.8
        initial_k = agent.knowledge

        agent._apply_forgetting()

        expected_k = 0.8 * (1.0 - 0.1)
        assert agent.knowledge == pytest.approx(expected_k, abs=1e-10)

    def test_vector_knowledge_decays(self):
        """Test vector knowledge decays element-wise."""
        agent = StudentAgent(
            user_id=81,
            knowledge_dim=3,
            knowledge_mode="vector",
            forgetting_rate=0.05,
            seed=42,
        )
        agent.knowledge = [0.8, 0.6, 0.4]

        agent._apply_forgetting()

        expected = [0.8*0.95, 0.6*0.95, 0.4*0.95]
        for i in range(3):
            assert agent.knowledge[i] == pytest.approx(expected[i], abs=1e-10)

    def test_forgetting_bounded_at_zero(self):
        """Test that knowledge doesn't go below 0."""
        agent = StudentAgent(
            user_id=82,
            knowledge_mode="scalar",
            forgetting_rate=0.5,
            seed=42,
        )
        agent.knowledge = 0.0

        agent._apply_forgetting()

        assert agent.knowledge == pytest.approx(0.0, abs=1e-10)

    def test_forgetting_over_multiple_rounds(self):
        """Test cumulative effect of forgetting over multiple rounds."""
        agent = StudentAgent(
            user_id=83,
            knowledge_mode="scalar",
            forgetting_rate=0.1,
            seed=42,
        )
        agent.knowledge = 1.0

        for _ in range(5):
            agent._apply_forgetting()

        # After 5 rounds: k = 1.0 * (0.9)^5
        expected = (0.9) ** 5
        assert agent.knowledge == pytest.approx(expected, abs=1e-10)


class TestFatigueRecovery:
    """Test fatigue recovery when no feedback."""

    def test_fatigue_recovers_with_no_feedback(self):
        """Test fatigue decreases when no items are engaged."""
        agent = StudentAgent(
            user_id=90,
            fatigue_recovery=0.02,
            seed=42,
        )
        agent.fatigue = 0.5

        agent._update_latents_after_round({}, None)

        expected = max(0.0, 0.5 - 0.02)
        assert agent.fatigue == pytest.approx(expected, abs=1e-10)

    def test_fatigue_bounded_at_zero(self):
        """Test fatigue doesn't go below 0 during recovery."""
        agent = StudentAgent(
            user_id=91,
            fatigue_recovery=0.1,
            seed=42,
        )
        agent.fatigue = 0.05

        agent._update_latents_after_round({}, None)

        assert agent.fatigue == pytest.approx(0.0, abs=1e-10)


class TestAbilityScalarComputation:
    """Test _ability_scalar helper method."""

    def test_scalar_knowledge_returns_knowledge(self):
        """Test that scalar mode returns the knowledge value."""
        agent = StudentAgent(
            user_id=100,
            knowledge_mode="scalar",
            seed=42,
        )
        agent.knowledge = 0.7

        theta = agent._ability_scalar(0, None)
        assert theta == pytest.approx(0.7, abs=1e-10)

    def test_vector_knowledge_mean(self):
        """Test that vector mode returns mean knowledge."""
        agent = StudentAgent(
            user_id=101,
            knowledge_dim=3,
            knowledge_mode="vector",
            seed=42,
        )
        agent.knowledge = [0.6, 0.8, 0.4]

        theta = agent._ability_scalar(0, None)
        expected = (0.6 + 0.8 + 0.4) / 3.0
        assert theta == pytest.approx(expected, abs=1e-10)

    def test_vector_knowledge_with_skills(self):
        """Test vector mode uses skill subset when available."""
        agent = StudentAgent(
            user_id=102,
            knowledge_dim=4,
            knowledge_mode="vector",
            seed=42,
        )
        agent.knowledge = [0.5, 0.7, 0.6, 0.8]

        # Skills are indices 1 and 3
        meta = ItemMeta(difficulty=0.5, skills=[False, True, False, True])
        items_meta = {10: meta}

        theta = agent._ability_scalar(10, items_meta)
        expected = (0.7 + 0.8) / 2.0
        assert theta == pytest.approx(expected, abs=1e-10)


class TestPositionBias:
    """Test position bias formula: eta^rank."""

    def test_position_bias_rank_zero(self):
        """Test position bias at rank 0."""
        agent = StudentAgent(user_id=110, pos_eta=0.85, seed=42)

        bias = agent._position_bias(0)
        assert bias == pytest.approx(1.0, abs=1e-10)

    def test_position_bias_decreases_with_rank(self):
        """Test position bias decreases with increasing rank."""
        agent = StudentAgent(user_id=111, pos_eta=0.85, seed=42)

        biases = [agent._position_bias(r) for r in range(5)]

        for i in range(len(biases) - 1):
            assert biases[i] > biases[i+1], \
                "Position bias should decrease with rank"

    def test_position_bias_formula(self):
        """Test position bias follows eta^rank formula."""
        agent = StudentAgent(user_id=112, pos_eta=0.8, seed=42)

        for rank in [0, 1, 2, 3]:
            bias = agent._position_bias(rank)
            expected = 0.8 ** rank
            assert bias == pytest.approx(expected, abs=1e-10)


class TestBaseRelevance:
    """Test base relevance formula (logistic without guess/slip)."""

    def test_base_relevance_zero_to_one(self):
        """Test base relevance returns value in (0, 1)."""
        agent = StudentAgent(user_id=120, seed=42)

        for theta in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for diff in [0.0, 0.3, 0.5, 0.7, 1.0]:
                rel = agent._base_relevance(theta, diff)
                assert 0.0 < rel < 1.0, \
                    f"Base relevance out of bounds: {rel}"

    def test_base_relevance_increases_with_ability(self):
        """Test base relevance increases with ability."""
        agent = StudentAgent(user_id=121, seed=42)
        diff = 0.5

        rel_low = agent._base_relevance(0.2, diff)
        rel_high = agent._base_relevance(0.8, diff)

        assert rel_high > rel_low, \
            "Base relevance should increase with ability"

    def test_base_relevance_decreases_with_difficulty(self):
        """Test base relevance decreases with difficulty."""
        agent = StudentAgent(user_id=122, seed=42)
        theta = 0.5

        rel_easy = agent._base_relevance(theta, 0.1)
        rel_hard = agent._base_relevance(theta, 0.9)

        assert rel_easy > rel_hard, \
            "Base relevance should decrease with difficulty"
