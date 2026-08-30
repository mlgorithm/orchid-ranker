from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from orchid_ranker import AdaptiveRanker
from orchid_ranker.adaptive_schema import DecisionOutcome, LoggedDecision
from orchid_ranker.decision_store import InMemoryDecisionStore, SQLiteDecisionStore


def _decision(decision_id: str = "decision-1") -> LoggedDecision:
    return LoggedDecision(
        decision_id=decision_id,
        user_id="learner-1",
        timestamp=10.0,
        candidate_item_ids=(101, 102),
        chosen_item_id=101,
        propensity=1.0,
        policy_name="hybrid",
        policy_version="test-v1",
        scores=(0.9, 0.4),
        context_hash="context-1",
    )


def _outcome(*, decision_id: str = "decision-1", value: int = 1, event_id: str | None = None) -> DecisionOutcome:
    return DecisionOutcome(
        decision_id=decision_id,
        user_id="learner-1",
        item_id=101,
        outcome_timestamp=11.0,
        outcome=value,
        reward=float(value),
        outcome_event_id=event_id,
    )


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_store_idempotently_creates_decisions_and_attaches_outcomes(kind: str, tmp_path: Path) -> None:
    store = InMemoryDecisionStore() if kind == "memory" else SQLiteDecisionStore(tmp_path / "orchid.sqlite")
    try:
        stored, created = store.create_decision(_decision())
        retried, retried_created = store.create_decision(_decision())

        assert created is True
        assert retried_created is False
        assert stored == retried

        with pytest.raises(ValueError, match="different immutable content"):
            store.create_decision(
                LoggedDecision(
                    **{
                        **_decision().to_dict(),
                        "chosen_item_id": 102,
                        "propensity": 0.0,
                    }
                )
            )

        linked, linked_created = store.attach_outcome(_outcome())
        repeated, repeated_created = store.attach_outcome(_outcome())

        assert linked_created is True
        assert repeated_created is False
        assert linked == repeated

        with pytest.raises(ValueError, match="already has an outcome"):
            store.attach_outcome(_outcome(value=0))
    finally:
        if isinstance(store, SQLiteDecisionStore):
            store.close()


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_store_rejects_outcome_event_id_reuse_across_decisions(kind: str, tmp_path: Path) -> None:
    store = InMemoryDecisionStore() if kind == "memory" else SQLiteDecisionStore(tmp_path / "orchid.sqlite")
    try:
        store.create_decision(_decision("decision-1"))
        store.create_decision(_decision("decision-2"))
        stored, created = store.attach_outcome(_outcome(event_id="lms-score-1"))

        assert created is True
        assert store.get_outcome_by_event_id("lms-score-1") == stored
        with pytest.raises(ValueError, match="outcome_event_id already belongs"):
            store.attach_outcome(_outcome(decision_id="decision-2", event_id="lms-score-1"))
    finally:
        if isinstance(store, SQLiteDecisionStore):
            store.close()


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b", "b"],
            "item_id": [101, 102, 201, 101, 102, 201],
            "outcome": [1, 1, 0, 1, 0, 0],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "category_id": ["foundation", "foundation", "advanced"] * 2,
        }
    )


def test_ranker_uses_sqlite_store_for_idempotent_serving_and_outcomes(tmp_path: Path) -> None:
    database = tmp_path / "orchid.sqlite"
    store = SQLiteDecisionStore(database)
    ranker = AdaptiveRanker(
        decision_store=store,
        kt_backbone="empirical",
        random_state=7,
    ).fit(_events(), category_col="category_id")
    kwargs = {
        "user_id": "a",
        "candidate_item_ids": [101, 102, 201],
        "timestamp": 4,
        "decision_id": "client-request-7",
        "min_outcome_probability": 0.0,
        "max_outcome_probability": 1.0,
        "require_prerequisites": False,
    }

    first_recommendations, first = ranker.recommend_and_log(**kwargs)
    retried_recommendations, retried = ranker.recommend_and_log(**kwargs)
    first_outcome = ranker.observe_decision(first.decision_id, outcome=1, timestamp=5)
    retried_outcome = ranker.observe_decision(first.decision_id, outcome=1, timestamp=5)

    assert first.decision_id == retried.decision_id == "client-request-7"
    assert [item.item_id for item in first_recommendations] == [item.item_id for item in retried_recommendations]
    assert first_outcome == retried_outcome
    assert ranker.diagnostics()["adaptive_ranker"]["logged_decisions"] == 1
    assert ranker.diagnostics()["adaptive_ranker"]["linked_outcomes"] == 1
    store.close()

    reopened = SQLiteDecisionStore(database)
    try:
        persisted = AdaptiveRanker(decision_store=reopened).decision_log_frame(completed_only=True)
        assert persisted["decision_id"].tolist() == ["client-request-7"]
        assert persisted["outcome"].tolist() == [1]
    finally:
        reopened.close()


def test_pending_outcome_replays_after_apply_failure_and_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = tmp_path / "orchid.sqlite"
    store = SQLiteDecisionStore(database)
    ranker = AdaptiveRanker(decision_store=store, kt_backbone="empirical", random_state=7).fit(
        _events(), category_col="category_id"
    )
    _, decision = ranker.recommend_and_log(
        "a",
        [101, 102, 201],
        timestamp=4,
        decision_id="pending-apply",
        min_outcome_probability=0.0,
        max_outcome_probability=1.0,
        require_prerequisites=False,
    )

    def fail_after_durable_attach(**_: object) -> None:
        raise RuntimeError("simulated live-state failure")

    monkeypatch.setattr(ranker, "observe", fail_after_durable_attach)
    with pytest.raises(RuntimeError, match="simulated live-state failure"):
        ranker.observe_decision(
            decision.decision_id,
            outcome=1,
            timestamp=5,
            outcome_event_id="score-pending",
        )

    assert store.get_outcome(decision.decision_id) is not None
    assert store.is_outcome_applied(decision.decision_id) is False
    assert [outcome.decision_id for outcome in store.pending_outcomes()] == [decision.decision_id]
    store.close()

    reopened = SQLiteDecisionStore(database)
    recovered = AdaptiveRanker(decision_store=reopened, kt_backbone="empirical", random_state=7).fit(
        _events(), category_col="category_id"
    )
    try:
        replayed = recovered.replay_pending_outcomes()

        assert [outcome.decision_id for outcome in replayed] == [decision.decision_id]
        assert reopened.is_outcome_applied(decision.decision_id) is True
        assert recovered.recommender_.tracer_._global_count == 7.0  # type: ignore[union-attr]
        assert recovered.replay_pending_outcomes() == []
    finally:
        reopened.close()


def test_fresh_baseline_can_rebuild_already_applied_outcomes_once(tmp_path: Path) -> None:
    database = tmp_path / "baseline-replay.sqlite"
    store = SQLiteDecisionStore(database)
    ranker = AdaptiveRanker(decision_store=store, kt_backbone="empirical", random_state=7).fit(
        _events(), category_col="category_id"
    )
    _, decision = ranker.recommend_and_log(
        "a",
        [101, 102, 201],
        timestamp=4,
        decision_id="baseline-replay",
        min_outcome_probability=0.0,
        max_outcome_probability=1.0,
        require_prerequisites=False,
    )
    ranker.observe_decision(
        decision.decision_id,
        outcome=1,
        timestamp=5,
        outcome_event_id="baseline-replay-score",
        update_global=False,
    )
    assert store.is_outcome_applied(decision.decision_id) is True
    store.close()

    reopened = SQLiteDecisionStore(database)
    recovered = AdaptiveRanker(decision_store=reopened, kt_backbone="empirical", random_state=7).fit(
        _events(), category_col="category_id"
    )
    try:
        assert recovered.replay_pending_outcomes() == []
        assert recovered.recommender_ is not None
        tracer = recovered.recommender_.tracer_
        before = tracer._user_count["a"]

        replayed = recovered.replay_all_outcomes_from_baseline()

        assert [outcome.decision_id for outcome in replayed] == [decision.decision_id]
        assert tracer._user_count["a"] == before + 1.0
    finally:
        reopened.close()


def test_frozen_outcome_updates_learner_state_without_global_empirical_counts() -> None:
    ranker = AdaptiveRanker(kt_backbone="empirical", random_state=7).fit(_events(), category_col="category_id")
    _, decision = ranker.recommend_and_log(
        "a",
        [101, 102, 201],
        timestamp=4,
        decision_id="frozen-outcome",
        min_outcome_probability=0.0,
        max_outcome_probability=1.0,
        require_prerequisites=False,
    )
    assert ranker.recommender_ is not None
    tracer = ranker.recommender_.tracer_
    global_count_before = tracer._global_count
    item_support_before = ranker.recommender_.item_support_[decision.chosen_item_id]
    user_count_before = tracer._user_count["a"]

    stored = ranker.observe_decision(
        decision.decision_id,
        outcome=1,
        timestamp=5,
        outcome_event_id="frozen-score",
        update_global=False,
    )

    assert stored.update_global is False
    assert tracer._global_count == global_count_before
    assert ranker.recommender_.item_support_[decision.chosen_item_id] == item_support_before
    assert tracer._user_count["a"] == user_count_before + 1.0


def test_audit_only_outcome_is_persisted_without_changing_live_state() -> None:
    ranker = AdaptiveRanker(kt_backbone="empirical", random_state=7).fit(_events(), category_col="category_id")
    _, decision = ranker.recommend_and_log(
        "a",
        [101, 102, 201],
        timestamp=4,
        decision_id="audit-only-outcome",
        min_outcome_probability=0.0,
        max_outcome_probability=1.0,
        require_prerequisites=False,
    )
    assert ranker.recommender_ is not None
    tracer = ranker.recommender_.tracer_
    global_count_before = tracer._global_count
    user_count_before = tracer._user_count["a"]

    stored = ranker.observe_decision(
        decision.decision_id,
        outcome=1,
        timestamp=5,
        outcome_event_id="audit-only-score",
        apply_state=False,
        update_global=False,
    )

    assert stored.apply_state is False
    assert ranker.decision_store.is_outcome_applied(decision.decision_id) is True
    assert tracer._global_count == global_count_before
    assert tracer._user_count["a"] == user_count_before


def test_decision_metadata_is_immutable_and_part_of_idempotency() -> None:
    ranker = AdaptiveRanker(kt_backbone="empirical", random_state=7).fit(_events(), category_col="category_id")
    kwargs = {
        "user_id": "a",
        "candidate_item_ids": [101, 102, 201],
        "timestamp": 4,
        "decision_id": "metadata-request",
        "min_outcome_probability": 0.0,
        "max_outcome_probability": 1.0,
        "require_prerequisites": False,
    }

    _, decision = ranker.recommend_and_log(
        **kwargs,
        decision_metadata={"pilot": {"catalog_version": "v1", "arm": "treatment"}},
    )
    _, retried = ranker.recommend_and_log(
        **kwargs,
        decision_metadata={"pilot": {"catalog_version": "v1", "arm": "treatment"}},
    )

    assert decision == retried
    assert decision.policy_metadata is not None
    assert decision.policy_metadata["decision_metadata"]["pilot"]["catalog_version"] == "v1"  # type: ignore[index]

    with pytest.raises(ValueError, match="different serving request"):
        ranker.recommend_and_log(
            **kwargs,
            decision_metadata={"pilot": {"catalog_version": "v2", "arm": "treatment"}},
        )
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        ranker.recommend_and_log(**{**kwargs, "decision_id": "bad-metadata"}, decision_metadata={1: "bad"})
