from __future__ import annotations

from orchid_ranker.learning_policy import ProgressionValuePolicy
from orchid_ranker.progression_reward import (
    ProgressionRewardConfig,
    expected_progression_reward,
    observed_progression_reward,
)


class _FakeTracer:
    def __init__(self) -> None:
        self.observed = []

    def predict_many(self, user_id, item_ids):
        del user_id
        values = {10: 0.95, 20: 0.70, 30: 0.35}
        return {item_id: values[item_id] for item_id in item_ids}

    def observe(self, user_id, item_id, correct):
        self.observed.append((user_id, item_id, correct))
        return len(self.observed)


def test_expected_progression_reward_penalizes_too_easy_items():
    easy = expected_progression_reward(p_correct=0.97, difficulty=0.1, competence=0.5)
    stretch = expected_progression_reward(p_correct=0.70, difficulty=0.65, competence=0.5)

    assert easy.easy_penalty > 0.0
    assert stretch.stretch_fit > easy.stretch_fit
    assert stretch.expected_reward > easy.expected_reward


def test_observed_progression_reward_values_harder_correct_outcomes():
    hard_correct = observed_progression_reward(correct=1, p_correct=0.7, difficulty=0.8, competence=0.5)
    easy_correct = observed_progression_reward(correct=1, p_correct=0.95, difficulty=0.1, competence=0.5)

    assert hard_correct > easy_correct


def test_observed_progression_reward_thresholds_numeric_labels():
    partial = observed_progression_reward(correct=0.4, p_correct=0.4, difficulty=0.5, competence=0.5)
    incorrect = observed_progression_reward(correct=0, p_correct=0.4, difficulty=0.5, competence=0.5)
    correct = observed_progression_reward(correct=0.6, p_correct=0.4, difficulty=0.5, competence=0.5)

    assert partial == incorrect
    assert correct > incorrect


def test_progression_reward_uses_target_correct_band():
    config = ProgressionRewardConfig(
        target_correct=0.85,
        correctness_weight=0.0,
        mastery_gain_weight=0.0,
        difficulty_weight=0.0,
        stretch_weight=1.0,
    )
    better_band = expected_progression_reward(p_correct=0.85, difficulty=0.65, competence=0.5, config=config)
    lower_band = expected_progression_reward(p_correct=0.65, difficulty=0.65, competence=0.5, config=config)

    assert better_band.stretch_fit > lower_band.stretch_fit
    assert better_band.expected_reward > lower_band.expected_reward


def test_progression_value_policy_prefers_stretch_over_easy_correct():
    tracer = _FakeTracer()
    policy = ProgressionValuePolicy(
        tracer,
        difficulty_by_item={10: 0.1, 20: 0.65, 30: 0.85},
        concept_by_item={10: "fractions", 20: "fractions", 30: "ratios"},
    )

    recs = policy.rank("u1", [10, 20, 30], top_k=3)
    length = policy.observe("u1", 20, 1)
    after_observe = policy.rank("u1", [10, 20], top_k=2)

    assert recs[0].item_id == 20
    assert recs[0].expected_reward == recs[0].score
    assert length == 1
    assert after_observe[0].recent_repetition >= 1


def test_progression_value_policy_uses_configured_correct_threshold():
    policy = ProgressionValuePolicy(
        _FakeTracer(),
        concept_by_item={20: "fractions"},
        correct_threshold=0.8,
    )

    policy.record_outcome("u1", 20, 0.7)

    assert policy.competence_for("u1", "fractions") == 0.0
