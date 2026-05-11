from __future__ import annotations

import pandas as pd

from orchid_ranker import AdaptiveLearningRecommender


def _learning_events() -> pd.DataFrame:
    sequence = [
        (10, "basics", 0.20),
        (11, "basics", 0.25),
        (20, "fractions", 0.45),
        (21, "fractions", 0.50),
        (20, "fractions", 0.45),
        (30, "ratios", 0.65),
        (31, "ratios", 0.70),
        (30, "ratios", 0.65),
    ]
    labels = {
        "needs-basics": [0, 0, 0, 0, 0, 0, 0, 0],
        "ready": [1, 1, 1, 1, 1, 0, 0, 1],
        "advanced": [1, 1, 1, 1, 1, 1, 1, 1],
        "mixed": [1, 0, 1, 0, 1, 0, 0, 0],
    }
    rows = []
    for user_id, outcomes in labels.items():
        for step, ((item_id, concept, difficulty), correct) in enumerate(zip(sequence, outcomes)):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "skill_id": concept,
                    "difficulty": difficulty,
                    "correct": correct,
                    "timestamp": step,
                }
            )
    return pd.DataFrame(rows)


def _small_rec(**kwargs) -> AdaptiveLearningRecommender:
    return AdaptiveLearningRecommender(
        max_seq_len=4,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        device="cpu",
        random_state=17,
        reward_model_max_examples=20,
        reward_model_cross_fit_folds=1,
        **kwargs,
    )


def test_adaptive_learning_auto_uses_progression_policy_until_delayed_gain_is_explicit():
    rec = _small_rec().fit(
        _learning_events(),
        timestamp_col="timestamp",
        concept_col="skill_id",
        item_difficulty_col="difficulty",
    )

    ranked = rec.rank("ready", [10, 20, 30, 31], top_k=3)
    diagnostics = rec.diagnostics()

    assert rec.policy_name_ == "progression"
    assert len(ranked) == 3
    assert all(0.0 <= item.p_correct <= 1.0 for item in ranked)
    assert all(item.policy == "progression" for item in ranked)
    assert diagnostics["delayed_gain_priors"] is None
    assert diagnostics["delayed_gain_reward_model"] is None


def test_adaptive_learning_support_constrained_delayed_gain_is_explicit_policy():
    rec = _small_rec(policy="support_delayed_gain").fit(
        _learning_events(),
        timestamp_col="timestamp",
        concept_col="skill_id",
        item_difficulty_col="difficulty",
    )

    ranked = rec.rank("ready", [10, 20, 30, 31], top_k=3)
    diagnostics = rec.diagnostics()

    assert rec.policy_name_ == "support_delayed_gain"
    assert ranked
    assert diagnostics["delayed_gain_priors"]["concept_priors"] >= 0
    assert diagnostics["delayed_gain_reward_model"] is not None


def test_adaptive_learning_prerequisites_use_warm_started_mastery_state():
    rec = _small_rec(
        policy="progression",
        allow_prerequisite_fallback=False,
        mastery_threshold=0.8,
    ).fit(
        _learning_events(),
        timestamp_col="timestamp",
        concept_col="skill_id",
        item_difficulty_col="difficulty",
        prerequisite_by_concept={
            "fractions": ["basics"],
            "ratios": ["fractions"],
        },
    )

    blocked = rec.rank("needs-basics", [10, 20, 30], top_k=3)
    ready = rec.rank("ready", [20, 30], top_k=2)

    assert rec.competence_for("needs-basics", "basics") == 0.0
    assert blocked
    assert {item.concept_id for item in blocked} == {"basics"}
    assert "basics" in rec.mastered_concepts("ready")
    assert ready
    assert all(item.prerequisites_met for item in ready)


def test_adaptive_learning_observe_updates_live_progression_state():
    rec = _small_rec(policy="progression").fit(
        _learning_events(),
        timestamp_col="timestamp",
        concept_col="skill_id",
        item_difficulty_col="difficulty",
    )

    before = rec.rank("live-user", [20, 21], top_k=2, enforce_prerequisites=False)
    length = rec.observe("live-user", 20, correct=True)
    after = rec.rank("live-user", [20, 21], top_k=2, enforce_prerequisites=False)

    assert length == 1
    assert before[0].recent_repetition == 0
    assert max(item.recent_repetition for item in after) >= 1
    assert rec.competence_for("live-user", "fractions") == 1.0


def test_adaptive_learning_prerequisites_have_state_for_kt_value_policy():
    rec = _small_rec(policy="kt_value", mastery_threshold=0.8).fit(
        _learning_events(),
        timestamp_col="timestamp",
        concept_col="skill_id",
        item_difficulty_col="difficulty",
        prerequisite_by_concept={"fractions": ["basics"]},
    )

    ready = rec.rank("ready", [20], top_k=1)
    blocked = rec.rank("needs-basics", [20], top_k=1)

    assert ready and ready[0].prerequisites_met
    assert blocked == []
