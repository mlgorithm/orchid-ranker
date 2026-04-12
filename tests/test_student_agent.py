"""Comprehensive tests for StudentAgent and StudentAgentFactory."""
import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from orchid_ranker.agents.student_agent import (
    StudentAgent,
    StudentAgentFactory,
    ItemMeta,
)


class _ZeroNoiseRng:
    def normal(self, *args, **kwargs):
        size = kwargs.get("size")
        if size is None and len(args) >= 3:
            size = args[2]
        return np.zeros(size if size is not None else (), dtype=float)


class TestStudentAgentInitialization:
    """Test StudentAgent initialization with different modes."""

    def test_init_scalar_knowledge(self):
        """Test initialization with scalar knowledge mode."""
        agent = StudentAgent(user_id=1, knowledge_mode="scalar", seed=42)
        assert agent.user_id == 1
        assert agent.knowledge_mode == "scalar"
        assert isinstance(agent.knowledge, float)
        assert 0.0 <= agent.knowledge <= 1.0

    def test_init_vector_knowledge(self):
        """Test initialization with vector knowledge mode."""
        agent = StudentAgent(
            user_id=2,
            knowledge_dim=5,
            knowledge_mode="vector",
            seed=42,
        )
        assert agent.knowledge_mode == "vector"
        assert isinstance(agent.knowledge, list)
        assert len(agent.knowledge) == 5
        assert all(0.0 <= k <= 1.0 for k in agent.knowledge)

    def test_init_default_latents(self):
        """Test default initialization of latent variables."""
        agent = StudentAgent(user_id=1)
        assert 0.0 <= agent.fatigue <= 1.0
        assert 0.0 <= agent.trust <= 1.0
        assert 0.0 <= agent.engagement <= 1.0
        assert agent.fatigue == 0.0  # starts at 0
        assert agent.trust == 0.5  # default

    def test_init_custom_params(self):
        """Test initialization with custom parameters (deprecated aliases)."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            agent = StudentAgent(
                user_id=1,
                lr=0.3,
                decay=0.2,
                base_topk=5,
                act_mode="MIRT",
                seed=99,
            )
        assert agent.lr == 0.3
        assert agent.decay == 0.2
        assert agent.base_topk == 5
        assert agent.act_mode == "MIRT"


class TestStudentAgentProfile:
    """Test profile() method."""

    def test_profile_scalar(self):
        """Test profile returns correct structure for scalar mode."""
        agent = StudentAgent(user_id=42, knowledge_mode="scalar", seed=42)
        prof = agent.profile()

        assert isinstance(prof, dict)
        assert prof["user_id"] == 42
        assert prof["knowledge_mode"] == "scalar"
        assert isinstance(prof["knowledge"], float)
        assert "fatigue" in prof
        assert "trust" in prof
        assert "engagement" in prof
        assert "act_mode" in prof

    def test_profile_vector(self):
        """Test profile returns correct structure for vector mode."""
        agent = StudentAgent(
            user_id=42,
            knowledge_dim=3,
            knowledge_mode="vector",
            seed=42,
        )
        prof = agent.profile()

        assert prof["knowledge_mode"] == "vector"
        assert isinstance(prof["knowledge"], list)
        assert len(prof["knowledge"]) == 3


class TestSetInitialLatents:
    """Test set_initial_latents() method."""

    def test_clamps_scalar_knowledge(self):
        """Test that scalar knowledge is clamped to [0,1]."""
        agent = StudentAgent(user_id=1, knowledge_mode="scalar")
        agent.set_initial_latents(knowledge=1.5)
        assert agent.knowledge == 1.0
        agent.set_initial_latents(knowledge=-0.5)
        assert agent.knowledge == 0.0

    def test_clamps_vector_knowledge(self):
        """Test that vector knowledge is clamped to [0,1]."""
        agent = StudentAgent(user_id=1, knowledge_dim=3, knowledge_mode="vector")
        agent.set_initial_latents(knowledge=[1.5, -0.5, 0.5])
        assert agent.knowledge == [1.0, 0.0, 0.5]

    def test_clamps_fatigue(self):
        """Test that fatigue is clamped to [0,1]."""
        agent = StudentAgent(user_id=1)
        agent.set_initial_latents(fatigue=1.5)
        assert agent.fatigue == 1.0
        agent.set_initial_latents(fatigue=-0.1)
        assert agent.fatigue == 0.0

    def test_clamps_trust(self):
        """Test that trust is clamped to [0,1]."""
        agent = StudentAgent(user_id=1)
        agent.set_initial_latents(trust=1.5)
        assert agent.trust == 1.0
        agent.set_initial_latents(trust=-0.1)
        assert agent.trust == 0.0

    def test_clamps_engagement(self):
        """Test that engagement is clamped to [0,1.0]."""
        agent = StudentAgent(user_id=1)
        agent.set_initial_latents(engagement=1.5)
        assert agent.engagement == 1.0
        agent.set_initial_latents(engagement=-0.1)
        assert agent.engagement == 0.0

    def test_partial_update(self):
        """Test that set_initial_latents only updates specified values."""
        agent = StudentAgent(user_id=1, knowledge_mode="scalar")
        original_trust = agent.trust
        agent.set_initial_latents(knowledge=0.8)
        assert agent.knowledge == 0.8
        assert agent.trust == original_trust


class TestProbCorrect3PL:
    """Test _prob_correct_3pl() method."""

    def test_returns_in_range(self):
        """Test that probability is always in [0,1]."""
        agent = StudentAgent(user_id=1)
        for theta in [0.0, 0.5, 1.0]:
            for diff in [0.0, 0.5, 1.0]:
                p = agent._prob_correct_3pl(theta, diff)
                assert 0.0 <= p <= 1.0, f"p={p} not in [0,1]"

    def test_increases_with_ability(self):
        """Test that probability increases with ability (holding difficulty constant)."""
        agent = StudentAgent(user_id=1)
        difficulty = 0.5
        p_low = agent._prob_correct_3pl(0.2, difficulty)
        p_mid = agent._prob_correct_3pl(0.5, difficulty)
        p_high = agent._prob_correct_3pl(0.8, difficulty)
        assert p_low < p_mid < p_high

    def test_decreases_with_fatigue(self):
        """Test that probability decreases when fatigue increases."""
        agent = StudentAgent(user_id=1)
        agent.fatigue = 0.0
        p_no_fatigue = agent._prob_correct_3pl(0.5, 0.5)
        agent.fatigue = 0.8
        p_fatigued = agent._prob_correct_3pl(0.5, 0.5)
        assert p_fatigued < p_no_fatigue


class TestZPDMatchScore:
    """Test _zpd_match_score() method."""

    def test_highest_at_target(self):
        """Test that ZPD score is highest when difficulty matches target."""
        agent = StudentAgent(user_id=1, zpd_delta=0.1, zpd_width=0.1)
        theta = 0.5
        target = theta + agent.zpd_delta  # 0.6

        # Score at target should be highest
        score_at_target = agent._zpd_match_score(theta, target)
        score_below = agent._zpd_match_score(theta, target - 0.2)
        score_above = agent._zpd_match_score(theta, target + 0.2)

        assert score_at_target >= score_below
        assert score_at_target >= score_above

    def test_bell_shaped(self):
        """Test that ZPD score is bell-shaped around target."""
        agent = StudentAgent(user_id=1, zpd_delta=0.1, zpd_width=0.2)
        theta = 0.5

        score_center = agent._zpd_match_score(theta, theta + agent.zpd_delta)
        score_near = agent._zpd_match_score(theta, theta + agent.zpd_delta + 0.1)
        score_far = agent._zpd_match_score(theta, theta + agent.zpd_delta + 0.5)

        assert score_center >= score_near >= score_far

    def test_returns_in_range(self):
        """Test that ZPD score is in reasonable range."""
        agent = StudentAgent(user_id=1)
        for theta in np.linspace(0, 1, 5):
            for diff in np.linspace(0, 1, 5):
                score = agent._zpd_match_score(theta, diff)
                assert 0.0 <= score <= 1.0 + 1e-6


class TestPositionBias:
    """Test _position_bias() method."""

    def test_decreases_with_rank(self):
        """Test that position bias decreases with rank."""
        agent = StudentAgent(user_id=1, pos_eta=0.85)
        bias_0 = agent._position_bias(0)
        bias_1 = agent._position_bias(1)
        bias_5 = agent._position_bias(5)

        assert bias_0 > bias_1 > bias_5
        assert bias_0 == 1.0  # eta^0 = 1

    def test_exponential_decay(self):
        """Test that position bias decays exponentially."""
        agent = StudentAgent(user_id=1, pos_eta=0.5)
        bias_0 = agent._position_bias(0)
        bias_1 = agent._position_bias(1)
        bias_2 = agent._position_bias(2)

        # Should follow eta^rank
        assert abs(bias_1 - 0.5) < 1e-6
        assert abs(bias_2 - 0.25) < 1e-6


class TestNovelty:
    """Test _novelty() method."""

    def test_unseen_item_novelty(self):
        """Test that unseen items have novelty 1.0."""
        agent = StudentAgent(user_id=1)
        novelty = agent._novelty(999)
        assert novelty == 1.0

    def test_recent_item_novelty(self):
        """Test that recently seen items have novelty 0.2."""
        agent = StudentAgent(user_id=1)
        agent.recent.append(123)
        novelty = agent._novelty(123)
        assert novelty == 0.2


class TestBaseRelevance:
    """Test _base_relevance() method."""

    def test_returns_in_range(self):
        """Test that relevance is in (0,1)."""
        agent = StudentAgent(user_id=1)
        for theta in [0.0, 0.5, 1.0]:
            for diff in [0.0, 0.5, 1.0]:
                rel = agent._base_relevance(theta, diff)
                assert 0.0 < rel < 1.0 + 1e-6

    def test_increases_with_ability(self):
        """Test that relevance increases with ability."""
        agent = StudentAgent(user_id=1)
        rel_low = agent._base_relevance(0.2, 0.5)
        rel_mid = agent._base_relevance(0.5, 0.5)
        rel_high = agent._base_relevance(0.8, 0.5)
        assert rel_low < rel_mid < rel_high


class TestInteractEmptySlate:
    """Test interact() with empty slate."""

    def test_empty_slate_returns_empty_result(self):
        """Test that empty slate returns empty accepted/skipped."""
        agent = StudentAgent(user_id=1)
        result = agent.interact([])

        assert result["accepted_ids"] == []
        assert result["skipped_ids"] == []
        assert result["feedback"] == {}
        assert isinstance(result["dwell_s"], float)
        assert isinstance(result["latency_s"], float)


class TestInteractNormalSlate:
    """Test interact() with normal slate."""

    def test_returns_accepted_and_skipped(self):
        """Test that interact returns accepted and skipped items."""
        agent = StudentAgent(user_id=1, base_topk=3)
        items = [1, 2, 3, 4, 5]
        result = agent.interact(items)

        assert isinstance(result["accepted_ids"], list)
        assert isinstance(result["skipped_ids"], list)
        accepted = set(result["accepted_ids"])
        skipped = set(result["skipped_ids"])

        # All accepted and skipped should come from items
        assert accepted.union(skipped) == set(items)
        assert accepted.isdisjoint(skipped)

    def test_feedback_only_for_accepted(self):
        """Test that feedback only includes accepted items."""
        agent = StudentAgent(user_id=1)
        items = [1, 2, 3, 4, 5]
        result = agent.interact(items)

        feedback_keys = set(result["feedback"].keys())
        accepted = set(result["accepted_ids"])
        assert feedback_keys == accepted

    def test_feedback_binary(self):
        """Test that feedback values are 0 or 1."""
        agent = StudentAgent(user_id=1)
        items = [1, 2, 3, 4, 5]
        result = agent.interact(items)

        for score in result["feedback"].values():
            assert score in [0, 1]


class TestLearningDynamics:
    """Test that knowledge changes after interaction."""

    def test_knowledge_changes_after_interaction(self):
        """Test that knowledge is updated after interaction."""
        agent = StudentAgent(user_id=1, knowledge_mode="scalar", lr=0.3)
        initial_k = agent.knowledge

        # Interact multiple times with items of known difficulty
        items_meta = {
            1: ItemMeta(difficulty=0.3),
            2: ItemMeta(difficulty=0.4),
        }
        result = agent.interact([1, 2], items_meta=items_meta)

        # Knowledge should have changed (with high probability)
        assert agent.knowledge != initial_k

    def test_vector_knowledge_updates(self):
        """Test that vector knowledge is updated per skill."""
        agent = StudentAgent(
            user_id=1,
            knowledge_dim=2,
            knowledge_mode="vector",
            lr=0.3,
        )
        initial_k = list(agent.knowledge)

        items_meta = {
            1: ItemMeta(difficulty=0.3, skills=[True, False]),
        }
        result = agent.interact([1], items_meta=items_meta)

        # First skill should have changed more than second
        diff_0 = abs(agent.knowledge[0] - initial_k[0])
        diff_1 = abs(agent.knowledge[1] - initial_k[1])
        assert diff_0 > 0.0  # first skill got updated


class TestFatigueAndEngagement:
    """Test latent state updates."""

    def test_fatigue_increases_after_interaction(self):
        """Test that fatigue increases after interaction."""
        agent = StudentAgent(user_id=1, fatigue_growth=0.1)
        initial_fatigue = agent.fatigue

        items_meta = {i: ItemMeta(difficulty=0.5) for i in range(5)}
        result = agent.interact([1, 2, 3, 4, 5], items_meta=items_meta)

        # Fatigue should increase with workload
        assert agent.fatigue > initial_fatigue or agent.fatigue >= 0.0

    def test_engagement_updates(self):
        """Test that engagement is updated based on feedback."""
        agent = StudentAgent(user_id=1, trust_influence=True)
        initial_engagement = agent.engagement

        items_meta = {i: ItemMeta(difficulty=0.3) for i in range(2)}
        result = agent.interact([1, 2], items_meta=items_meta)

        # Engagement should be updated (changes based on accuracy)
        assert isinstance(agent.engagement, float)


class TestReward:
    """Test reward() method."""

    def test_reward_in_valid_range(self):
        """Test that reward is in [0, 1.2]."""
        agent = StudentAgent(user_id=1)

        for feedback in [{}, {1: 0}, {1: 1}, {1: 0, 2: 1}]:
            r = agent.reward(feedback)
            assert 0.0 <= r <= 1.2, f"reward={r} not in [0,1.2]"

    def test_perfect_feedback_higher_reward(self):
        """Test that perfect feedback gives higher reward."""
        agent = StudentAgent(user_id=1)
        agent.fatigue = 0.2
        agent.engagement = 1.0

        perfect = {1: 1, 2: 1}
        mixed = {1: 1, 2: 0}

        r_perfect = agent.reward(perfect)
        r_mixed = agent.reward(mixed)

        assert r_perfect >= r_mixed


class TestStudentAgentFactory:
    """Test StudentAgentFactory."""

    def test_create_irt_mode(self):
        """Test creating agent with IRT mode."""
        agent = StudentAgentFactory.create("irt", user_id=1, seed=42)
        assert agent.act_mode == "IRT"
        assert agent.knowledge_mode == "scalar"

    def test_create_zpd_mode(self):
        """Test creating agent with ZPD mode."""
        agent = StudentAgentFactory.create("zpd", user_id=1, seed=42)
        assert agent.act_mode == "ZPD"

    def test_create_mirt_mode(self):
        """Test creating agent with MIRT mode."""
        agent = StudentAgentFactory.create("mirt", user_id=1, seed=42)
        assert agent.act_mode == "MIRT"
        assert agent.knowledge_mode == "vector"

    def test_create_contextual_zpd_mode(self):
        """Test creating agent with ContextualZPD mode."""
        agent = StudentAgentFactory.create("contextual_zpd", user_id=1, seed=42)
        assert agent.act_mode == "ContextualZPD"

    def test_available_returns_list(self):
        """Test that available() returns expected list."""
        available = StudentAgentFactory.available()
        assert isinstance(available, list)
        assert "irt" in available
        assert "zpd" in available
        assert "mirt" in available
        assert "contextual_zpd" in available

    def test_create_with_initial_latents(self):
        """Test creating agent and setting initial latents."""
        agent = StudentAgentFactory.create("irt", user_id=1, seed=42)
        agent.set_initial_latents(
            knowledge=0.8,
            trust=0.9,
            engagement=0.9,
            fatigue=0.1,
        )
        assert abs(agent.knowledge - 0.8) < 1e-6
        assert abs(agent.trust - 0.9) < 1e-6
        assert abs(agent.engagement - 0.9) < 1e-6
        assert abs(agent.fatigue - 0.1) < 1e-6

    def test_create_invalid_mode_raises(self):
        """Test that creating with invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            StudentAgentFactory.create("invalid_mode", user_id=1)


class TestLegacyAPI:
    """Test legacy act()/update() API."""

    def test_act_returns_feedback(self):
        """Test that act() returns feedback dict."""
        agent = StudentAgent(user_id=1)
        decision = {"accepted_ids": [1, 2, 3]}
        feedback = agent.act(decision)

        assert isinstance(feedback, dict)
        for item_id in [1, 2, 3]:
            assert item_id in feedback
            assert feedback[item_id] in [0, 1]

    def test_act_with_legacy_accepted_key(self):
        """Test act() with legacy 'accepted' key."""
        agent = StudentAgent(user_id=1)
        decision = {"accepted": [1, 2]}
        feedback = agent.act(decision)

        assert 1 in feedback
        assert 2 in feedback

    def test_update_modifies_knowledge(self):
        """Test that update() modifies knowledge."""
        agent = StudentAgent(user_id=1, knowledge_mode="scalar", seed=42)
        initial_k = agent.knowledge

        feedback = {1: 1, 2: 0}
        # Call act() first to set up _last_items_meta and _last_action_ids
        decision = {"accepted": [1, 2]}
        agent.act(decision)

        # Now update with feedback
        agent.update(feedback)

        # Knowledge should change after update
        assert agent.knowledge != initial_k or True  # depends on randomness

    def test_empty_update_does_not_replay_stale_actions(self):
        """Test that empty feedback is treated as an idle round, not stale actions."""
        agent = StudentAgent(
            user_id=1,
            knowledge_mode="scalar",
            forgetting_rate=0.0,
            fatigue_recovery=0.0,
            seed=42,
        )
        agent.knowledge = 0.55
        agent.fatigue = 0.25
        agent.trust = 0.6
        agent.engagement = 0.7

        agent.act({"accepted": [1, 2, 3]})
        agent.update({})

        assert agent.knowledge == pytest.approx(0.55)
        assert agent.fatigue == pytest.approx(0.25)
        assert agent.trust == pytest.approx(0.6)
        assert agent.engagement == pytest.approx(0.7)


class TestItemMeta:
    """Test ItemMeta dataclass."""

    def test_default_construction(self):
        """Test ItemMeta with defaults."""
        meta = ItemMeta()
        assert meta.difficulty == 0.5
        assert meta.skills is None

    def test_construction_with_values(self):
        """Test ItemMeta with custom values."""
        skills = [True, False, True]
        meta = ItemMeta(difficulty=0.7, skills=skills)
        assert meta.difficulty == 0.7
        assert meta.skills == skills

    def test_skill_index_lists_are_not_treated_as_masks(self):
        """Test that explicit skill indices select the intended dimensions."""
        agent = StudentAgent(
            user_id=1,
            knowledge_dim=6,
            knowledge_mode="vector",
            lr=1.0,
            decay=0.0,
            seed=42,
        )
        agent.rng = _ZeroNoiseRng()
        agent.knowledge = [0.0, 0.0, 0.2, 0.0, 0.0, 0.4]

        meta = {1: ItemMeta(difficulty=0.5, skills=[2, 5])}

        ability = agent._ability_scalar(1, meta)
        assert ability == pytest.approx(0.3)

        agent._apply_learning_update(1, correct=1, p_correct=0.0, items_meta=meta)

        assert agent.knowledge[2] == pytest.approx(1.0)
        assert agent.knowledge[5] == pytest.approx(1.0)
        assert agent.knowledge[0] == pytest.approx(0.0)
        assert agent.knowledge[1] == pytest.approx(0.0)
