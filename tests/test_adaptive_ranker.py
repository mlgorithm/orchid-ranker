from __future__ import annotations

import pandas as pd
import pytest

from orchid_ranker import AdaptiveRanker


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
            "item_id": [101, 102, 201, 202] * 3,
            "outcome": [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            "timestamp": list(range(4)) * 3,
            "category_id": ["start", "start", "advance", "advance"] * 3,
            "difficulty": [0.2, 0.3, 0.6, 0.7] * 3,
        }
    )


def _ranker() -> AdaptiveRanker:
    return AdaptiveRanker(epochs=1, d_model=8, n_heads=2, batch_size=4, device="cpu").fit(
        _history(),
        category_col="category_id",
        difficulty_col="difficulty",
    )


def test_supported_loop_uses_neutral_fields_only():
    ranker = _ranker()
    ranked = ranker.recommend(user_id="a", candidate_item_ids=[101, 102, 201, 202], top_k=2)

    assert len(ranked) == 2
    assert 0.0 <= ranked[0].outcome_probability <= 1.0
    assert not hasattr(ranked[0], "p_correct")

    ranker.observe(user_id="a", item_id=ranked[0].item_id, outcome=1, timestamp=5)


def test_reward_model_refit_preserves_live_observations_and_registered_items():
    rows = [
        {
            "user_id": f"u{user}",
            "item_id": step % 4,
            "outcome": int((user + step) % 3 != 0),
            "timestamp": step,
            "category_id": "skill",
        }
        for user in range(8)
        for step in range(12)
    ]
    ranker = AdaptiveRanker(kt_backbone="empirical", reward_model_cross_fit_folds=1).fit(
        pd.DataFrame(rows), category_col="category_id"
    )
    ranker.register_items(pd.DataFrame({"item_id": [99], "category_id": ["skill"]}))
    ranker.observe(user_id="u0", item_id=0, outcome=0, timestamp=12, update_global=False)
    ranker.observe(user_id="u0", item_id=99, outcome=1, timestamp=13)
    assert ranker.recommender_ is not None
    before = (
        ranker.recommender_.tracer_._user_count["u0"],
        ranker.recommender_.tracer_._global_count,
        ranker.recommender_.item_support_[99],
    )

    ranker.fit_reward_model()

    assert ranker.recommender_ is not None
    assert ranker.recommender_.policy_name_ == "support_delayed_gain"
    assert ranker.recommender_.tracer_._user_count["u0"] == before[0]
    assert ranker.recommender_.tracer_._global_count == before[1]
    assert ranker.recommender_.item_support_[99] == before[2]


def test_legacy_schema_and_aliases_are_rejected():
    ranker = AdaptiveRanker(epochs=1, d_model=8, n_heads=2, batch_size=4, device="cpu")
    legacy = _history().rename(columns={"user_id": "learner_id", "outcome": "correct", "timestamp": "ts"})

    with pytest.raises(ValueError, match="user"):
        ranker.fit(legacy)
    with pytest.raises(TypeError):
        ranker.fit(_history(), tracer_model="akt")


def test_decision_logging_is_neutral_and_links_one_outcome():
    ranker = _ranker()
    ranked, decision = ranker.recommend_and_log(
        user_id="a",
        candidate_item_ids=[101, 102, 201, 202],
        timestamp=5,
        top_k=2,
    )
    linked = ranker.observe_decision(decision.decision_id, outcome=1, timestamp=6)
    frame = ranker.decision_log_frame(completed_only=True)

    assert ranked[0].item_id == decision.chosen_item_id
    assert decision.user_id == "a"
    assert decision.timestamp == 5
    assert linked.user_id == "a"
    assert linked.outcome_timestamp == 6
    assert {"user_id", "timestamp", "outcome", "predicted_outcomes"}.issubset(frame.columns)
    assert "learner_id" not in frame.columns
    assert "correct" not in frame.columns

    with pytest.raises(ValueError, match="already has an outcome"):
        ranker.observe_decision(decision.decision_id, outcome=1, timestamp=7)


def test_custom_event_column_names_are_explicit():
    events = _history().rename(
        columns={"user_id": "account", "item_id": "task", "outcome": "success", "timestamp": "event_time"}
    )
    ranker = AdaptiveRanker(epochs=1, d_model=8, n_heads=2, batch_size=4, device="cpu").fit(
        events,
        user_col="account",
        item_col="task",
        outcome_col="success",
        timestamp_col="event_time",
    )

    assert ranker.recommend(user_id="a", candidate_item_ids=[101, 102], top_k=1)
