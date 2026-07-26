"""Regression tests for serving-state safety boundaries."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from orchid_ranker import AdaptiveRanker
from orchid_ranker.adaptive_ranker import _require_disjoint_policy_logs
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
    ranker = _ranker(offline_policy_min_effect=1.0)
    training = _logged_rows("train", 0)
    evaluation = _logged_rows("evaluation", 100)

    report = ranker.fit_policy(
        training,
        evaluation_decisions=evaluation,
        epochs=2,
        cluster_bootstrap_samples=10,
        min_evaluation_events=1,
        min_evaluation_users=1,
    )

    assert report.n_events == len(training)
    assert ranker.last_policy_gate_ is not None
    assert ranker.last_policy_gate_.allowed is False
    assert ranker.offline_policy_ is None


def test_invalid_policy_evaluation_leaves_an_active_policy_unchanged() -> None:
    training = _logged_rows("existing", 0)
    evaluation = training.copy()
    evaluation["timestamp"] += 100
    ranker = _ranker()
    existing = CQLDiscretePolicy(epochs=2).fit(training)
    ranker.offline_policy_ = existing

    with pytest.raises(ValueError, match="disjoint"):
        ranker.fit_policy(
            training,
            evaluation_decisions=evaluation,
            epochs=2,
            cluster_bootstrap_samples=10,
            min_evaluation_events=1,
            min_evaluation_users=1,
        )

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


def test_offline_policy_evaluation_replays_the_deployed_hybrid_cql_action_rule() -> None:
    ranker = _ranker()
    training = _logged_rows("train", 0)
    training["chosen_item_id"] = "b"
    training["reward"] = 1.0
    evaluation = _logged_rows("evaluation", 100)
    evaluation["chosen_item_id"] = "a"
    evaluation["reward"] = 1.0
    evaluation["scores"] = json.dumps([1.0, 0.0])
    candidate = CQLDiscretePolicy(epochs=10, random_state=1).fit(training)

    assert candidate.recommend("shared-context", ["a", "b"], top_k=1) == ["b"]
    evidence = ranker._evaluate_candidate_policy(
        candidate,
        evaluation,
        reward_col="reward",
        cluster_bootstrap_samples=0,
        cluster_col="user_id",
    )

    # CQL alone chooses b, but the deployed blend retains a because its base
    # adaptive score exceeds the bounded CQL contribution.
    assert evidence.target.coverage == 1.0


def test_successful_policy_promotion_versions_the_hybrid_cql_action_rule() -> None:
    ranker = _ranker(offline_policy_min_effect=-0.01)
    training = _logged_rows("train", 0)
    evaluation = _logged_rows("evaluation", 100)
    before = ranker._deployment_version

    ranker.fit_policy(
        training,
        evaluation_decisions=evaluation,
        epochs=10,
        cluster_bootstrap_samples=20,
        min_evaluation_events=1,
        min_evaluation_users=1,
    )

    assert ranker.offline_policy_ is not None
    assert ranker._deployment_version != before
    _, decision = ranker.recommend_and_log(
        "a", [101, 102], timestamp=5, min_outcome_probability=0.0, max_outcome_probability=1.0
    )
    assert decision.policy_name == "hybrid+cql"
    assert decision.policy_version.startswith("orchid-hybrid+cql-")
    assert decision.policy_metadata is not None
    assert len(decision.policy_metadata["base_scores"]) == len(decision.candidate_item_ids)


def test_exploration_support_override_is_respected() -> None:
    ranker = _ranker()

    with pytest.raises(ValueError, match="no candidate"):
        ranker.recommend_and_log(
            "a",
            [101, 102],
            timestamp=5,
            exploration=0.05,
            min_item_support=100.0,
            min_outcome_probability=0.0,
            max_outcome_probability=1.0,
        )


def test_exploration_can_explicitly_include_a_zero_support_registered_item() -> None:
    ranker = _ranker()
    ranker.register_items(pd.DataFrame({"item_id": [999]}))

    _, decision = ranker.recommend_and_log(
        "a",
        [999],
        timestamp=5,
        exploration=0.05,
        min_item_support=0.0,
        min_outcome_probability=0.0,
        max_outcome_probability=1.0,
        require_prerequisites=False,
    )

    assert decision.policy_metadata is not None
    assert decision.policy_metadata["min_item_support"] == 0.0


def test_fit_policy_requires_a_strictly_future_evaluation_window() -> None:
    ranker = _ranker()
    training = _logged_rows("train", 100)
    evaluation = _logged_rows("evaluation", 0)

    with pytest.raises(ValueError, match="strictly later"):
        ranker.fit_policy(
            training,
            evaluation_decisions=evaluation,
            epochs=2,
            cluster_bootstrap_samples=10,
            min_evaluation_events=1,
            min_evaluation_users=1,
        )


def test_policy_holdout_rejects_a_reidentified_duplicate_event() -> None:
    training = _logged_rows("training", 0).iloc[:1]
    evaluation = training.copy()
    evaluation["decision_id"] = "different-id"

    with pytest.raises(ValueError, match="disjoint"):
        _require_disjoint_policy_logs(training, evaluation)


class _ExternalSemanticEncoder:
    is_fitted = True
    item_ids_ = ["external-item"]

    def similar_items(self, query_text: str, *, top_k: int) -> list[str]:
        del query_text, top_k
        return list(self.item_ids_)

    def scores(self, query_text: str, *, candidate_item_ids: list[str]) -> dict[str, float]:
        del query_text
        return {item_id: 0.9 for item_id in candidate_item_ids}

    def metadata(self, item_id: str) -> dict[str, object]:
        del item_id
        return {}


def test_semantic_items_without_local_feedback_are_flagged_and_not_logged_by_default() -> None:
    ranker = _ranker()
    ranker.attach_semantic_encoder(_ExternalSemanticEncoder())

    recs = ranker.recommend("a", ["external-item"], item_query_text="query")
    assert recs[0].feedback_supported is False
    with pytest.raises(ValueError, match="without local feedback support"):
        ranker.recommend_and_log(
            "a",
            ["external-item"],
            timestamp=5,
            min_outcome_probability=0.0,
            max_outcome_probability=1.0,
        )
    _, decision = ranker.recommend_and_log(
        "a",
        ["external-item"],
        timestamp=6,
        min_outcome_probability=0.0,
        max_outcome_probability=1.0,
        allow_unsupported_feedback=True,
    )
    assert decision.policy_metadata is not None
    assert decision.policy_metadata["feedback_supported"] is False
    assert decision.policy_metadata["external_feedback_required"] is True


def test_explicit_empty_numpy_candidate_array_is_safe() -> None:
    ranker = _ranker()

    assert ranker.recommend("a", np.asarray([], dtype=int)) == []
