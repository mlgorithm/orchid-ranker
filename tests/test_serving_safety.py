"""Regression tests for serving-state safety boundaries."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from orchid_ranker import AdaptiveRanker
from orchid_ranker.adaptive_schema import LoggedDecision, validate_logged_decisions, validate_user_events
from orchid_ranker.offline_policy import CQLDiscretePolicy


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "item_id": [101, 102, 201, 202] * 2,
            "outcome": [1, 1, 0, 0, 1, 0, 1, 0],
            "timestamp": [1, 2, 3, 4] * 2,
            "category_id": ["start", "start", "advance", "advance"] * 2,
        }
    )


def _ranker(**overrides: object) -> AdaptiveRanker:
    return AdaptiveRanker(
        epochs=1,
        d_model=8,
        n_heads=2,
        batch_size=4,
        device="cpu",
        **overrides,
    ).fit(_history(), category_col="category_id")


def _logged_rows(prefix: str, start: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset in range(16):
        chosen = "a" if offset % 2 == 0 else "b"
        rows.append(
            {
                "decision_id": f"{prefix}-{offset}",
                "user_id": f"user-{offset % 4}",
                "timestamp": start + offset,
                "candidate_item_ids": json.dumps(["a", "b"]),
                "chosen_item_id": chosen,
                "propensity": 0.5,
                "action_probabilities": json.dumps([0.5, 0.5]),
                "policy_name": "logging",
                "policy_version": "logging-v1",
                "scores": json.dumps([0.0, 0.0]),
                "context_hash": "shared-context",
                "reward": float(chosen == "a"),
            }
        )
    return pd.DataFrame(rows)


def test_empty_eligible_set_never_falls_back_to_the_training_catalog() -> None:
    ranker = _ranker()

    assert ranker.recommend("a", []) == []
    assert ranker.recommend("a") == []


@pytest.mark.parametrize("outcome", [0.9, 1.9, -0.5, float("nan")])
def test_live_outcomes_are_exact_binary_values(outcome: float) -> None:
    ranker = _ranker()

    with pytest.raises(ValueError, match="exactly 0 or 1"):
        ranker.observe(user_id="a", item_id=101, outcome=outcome, timestamp=5)


def test_live_timestamp_is_validated_before_conversion() -> None:
    ranker = _ranker()

    with pytest.raises(ValueError, match="timestamp"):
        ranker.observe(user_id="a", item_id=101, outcome=1, timestamp=-0.5)


def test_logged_decision_is_deeply_immutable_and_stored_defensively() -> None:
    ranker = _ranker()
    _, decision = ranker.recommend_and_log(
        "a",
        [101, 102],
        timestamp=5,
        min_outcome_probability=0.0,
        max_outcome_probability=1.0,
    )

    with pytest.raises(AttributeError):
        decision.candidate_item_ids.append(999)  # type: ignore[attr-defined]
    assert decision.policy_metadata is not None
    with pytest.raises(TypeError):
        decision.policy_metadata["min_item_support"] = -1  # type: ignore[index]

    stored = ranker.decision_log_frame()
    assert stored.loc[0, "candidate_item_ids"] == list(decision.candidate_item_ids)
    assert stored.loc[0, "policy_metadata"]["min_item_support"] == 0.0


def test_logged_decision_freezes_nested_metadata() -> None:
    decision = LoggedDecision(
        user_id="a",
        timestamp=1.5,
        candidate_item_ids=("x",),
        chosen_item_id="x",
        propensity=1.0,
        policy_name="policy",
        policy_version="version",
        scores=(1.0,),
        context_hash="context",
        policy_metadata={"nested": {"values": [1, 2]}},
    )

    assert decision.policy_metadata is not None
    with pytest.raises(TypeError):
        decision.policy_metadata["nested"]["values"] = []  # type: ignore[index]
    assert decision.to_dict()["policy_metadata"] == {"nested": {"values": [1, 2]}}


def test_schema_normalizes_timestamps_before_chronological_use() -> None:
    events = pd.DataFrame(
        {"user_id": ["a", "a"], "item_id": [1, 2], "outcome": [1, 0], "timestamp": ["10", "2"]}
    )
    normalized = validate_user_events(events)
    assert normalized["timestamp"].tolist() == [10.0, 2.0]

    decisions = _logged_rows("schema", 0).iloc[:2].copy()
    decisions["timestamp"] = ["10", "2"]
    logged = validate_logged_decisions(decisions, reward_col="reward")
    assert logged.sort_values("timestamp")["timestamp"].tolist() == [2.0, 10.0]


def test_logged_policy_schema_rejects_duplicate_candidates() -> None:
    decisions = _logged_rows("duplicate", 0).iloc[:1].copy()
    decisions.loc[0, "candidate_item_ids"] = json.dumps(["a", "a"])
    decisions.loc[0, "scores"] = json.dumps([0.0, 0.0])
    decisions.loc[0, "action_probabilities"] = json.dumps([0.5, 0.5])

    with pytest.raises(ValueError, match="must not contain duplicates"):
        validate_logged_decisions(decisions, reward_col="reward")


def test_failed_policy_gate_does_not_install_the_candidate_policy() -> None:
    ranker = AdaptiveRanker(offline_policy_min_effect=1.0)
    training = _logged_rows("train", 0)
    evaluation = _logged_rows("evaluation", 100)

    report = ranker.fit_policy(training, evaluation_decisions=evaluation, epochs=2)

    assert report.n_events == len(training)
    assert ranker.last_policy_gate_ is not None
    assert ranker.last_policy_gate_.allowed is False
    assert ranker.offline_policy_ is None


def test_invalid_policy_evaluation_leaves_an_active_policy_unchanged() -> None:
    training = _logged_rows("existing", 0)
    ranker = AdaptiveRanker()
    existing = CQLDiscretePolicy(epochs=2).fit(training)
    ranker.offline_policy_ = existing

    with pytest.raises(ValueError, match="disjoint"):
        ranker.fit_policy(training, evaluation_decisions=training, epochs=2)

    assert ranker.offline_policy_ is existing


def test_registered_cold_item_can_complete_the_serve_log_observe_loop() -> None:
    ranker = _ranker()
    ranker.register_items(pd.DataFrame({"item_id": [999], "category_id": ["advance"], "difficulty": [0.5]}))

    ranked, decision = ranker.recommend_and_log(
        "a",
        [999],
        timestamp=5,
        min_outcome_probability=0.0,
        max_outcome_probability=1.0,
        require_prerequisites=False,
    )
    linked = ranker.observe_decision(decision.decision_id, outcome=1, timestamp=6)

    assert ranked[0].item_id == 999
    assert ranked[0].feedback_supported is True
    assert linked.item_id == 999


def test_log_uses_resolved_policy_and_deployment_fingerprint() -> None:
    ranker = _ranker()
    _, decision = ranker.recommend_and_log(
        "a", [101, 102], timestamp=5, min_outcome_probability=0.0, max_outcome_probability=1.0
    )

    assert decision.policy_name == "hybrid"
    assert decision.policy_version.startswith("orchid-hybrid-")
