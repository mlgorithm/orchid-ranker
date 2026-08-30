"""End-to-end checks for the reference adaptive-practice pilot adapter."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orchid_ranker import AdaptiveRanker
from orchid_ranker.decision_store import SQLiteDecisionStore
from orchid_ranker.pilot import (
    AdaptivePracticePilot,
    PilotCatalog,
    PilotRequest,
    SQLiteExperimentAssignmentStore,
    SQLitePilotLifecycleStore,
    _analysis_grouping_key,
)


def _catalog(*, required_first: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["net-1", "net-2", "net-3", "net-assessment"],
            "content_version": ["2026.1"] * 4,
            "course_id": ["networking"] * 4,
            "module_id": ["routing"] * 4,
            "category_id": ["routing", "routing", "routing", "routing"],
            "difficulty": [0.2, 0.4, 0.6, 0.7],
            "assessment_only": [False, False, False, True],
            "prerequisites": [[], ["net-1"], ["net-2"], []],
            "available": [True, True, True, True],
            "required": [required_first, False, False, False],
            "authored_sequence_position": [10, 20, 30, 40],
        }
    )


def _events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for learner_index, outcomes in enumerate(((1, 0, 0), (1, 1, 0), (1, 1, 1), (0, 1, 0))):
        for step, (item_id, outcome) in enumerate(zip(("net-1", "net-2", "net-3"), outcomes)):
            rows.append(
                {
                    "user_id": f"learner-{learner_index}",
                    "item_id": item_id,
                    "outcome": outcome,
                    "timestamp": learner_index * 10 + step,
                }
            )
    return pd.DataFrame(rows)


def _ranker(catalog: pd.DataFrame, **kwargs: object) -> AdaptiveRanker:
    return AdaptiveRanker(kt_backbone="empirical", random_state=7, **kwargs).fit(
        _events(),
        catalog=catalog,
    )


def test_control_enforces_authored_requirements_and_assessment_holdouts() -> None:
    catalog = _catalog(required_first=True)
    pilot = AdaptivePracticePilot(
        _ranker(catalog),
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="routing-pilot",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=0.0,
    )
    request = PilotRequest(
        request_id="request-1",
        user_id="new-learner",
        course_id="networking",
        module_id="routing",
        timestamp=100,
        candidate_item_ids=("net-1",),
    )

    served = pilot.serve(request)
    retried = pilot.serve(request)

    assert served.arm == "control"
    assert served.decision.policy_name == "authored-static"
    assert served.decision.chosen_item_id == "net-1"
    assert served.reason_code == "AUTHORED_REQUIRED"
    assert retried.decision == served.decision
    assert len(pilot.ranker.decision_store.decisions()) == 1

    pilot.record_rendered(
        served.decision.decision_id,
        event_id="render-event-1",
        item_id="net-1",
        content_version="2026.1",
        timestamp=100.5,
    )
    pilot.record_submitted(
        served.decision.decision_id,
        event_id="submission-event-1",
        item_id="net-1",
        content_version="2026.1",
        timestamp=100.75,
    )
    linked = pilot.record_scored(
        served.decision.decision_id,
        outcome=1,
        timestamp=101,
        outcome_event_id="score-event-1",
        item_id="net-1",
        content_version="2026.1",
    )
    assert linked.item_id == "net-1"
    assert linked.outcome_event_id == "score-event-1"
    assert (
        pilot.record_scored(
            served.decision.decision_id,
            outcome=1,
            timestamp=101,
            outcome_event_id="score-event-1",
            item_id="net-1",
            content_version="2026.1",
        )
        == linked
    )

    next_request = PilotRequest(
        request_id="request-2",
        user_id="new-learner",
        course_id="networking",
        module_id="routing",
        timestamp=102,
        completed_item_ids=("net-1",),
        candidate_item_ids=("net-2",),
    )
    assert pilot.serve(next_request).decision.chosen_item_id == "net-2"

    blocked_assessment = PilotRequest(
        request_id="request-3",
        user_id="another-learner",
        course_id="networking",
        module_id="routing",
        timestamp=103,
        candidate_item_ids=("net-1", "net-assessment"),
    )
    with pytest.raises(ValueError, match="assessment-only"):
        pilot.serve(blocked_assessment)


def test_treatment_records_immutable_pilot_context_and_rejects_changed_retries() -> None:
    catalog = _catalog(required_first=False)
    pilot = AdaptivePracticePilot(
        _ranker(catalog),
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="routing-pilot",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=1.0,
    )
    request = PilotRequest(
        request_id="request-treatment",
        user_id="learner-treatment",
        course_id="networking",
        module_id="routing",
        timestamp=100,
        candidate_item_ids=("net-1",),
        stratum="baseline-low",
    )

    served = pilot.serve(request)
    metadata = served.decision.policy_metadata["decision_metadata"]["pilot"]  # type: ignore[index]

    assert served.arm == "treatment"
    assert served.decision.policy_version == "orchid-empirical-2026-08-30"
    assert served.decision.exploration_rate == 0.0
    assert metadata["catalog_version"] == "networking-2026.1"
    assert metadata["experiment_arm"] == "treatment"
    assert metadata["candidate_content_versions"] == (("net-1", "2026.1"),)

    changed_retry = PilotRequest(
        request_id="request-treatment",
        user_id="learner-treatment",
        course_id="networking",
        module_id="routing",
        timestamp=101,
        candidate_item_ids=("net-1",),
        stratum="baseline-low",
    )
    with pytest.raises(ValueError, match="different pilot request"):
        pilot.serve(changed_retry)

    pilot.record_rendered(
        served.decision.decision_id,
        event_id="treatment-render-event",
        item_id="net-1",
        content_version="2026.1",
        timestamp=100.5,
    )
    pilot.record_submitted(
        served.decision.decision_id,
        event_id="treatment-submission-event",
        item_id="net-1",
        content_version="2026.1",
        timestamp=100.75,
    )
    pilot.record_scored(
        served.decision.decision_id,
        outcome=1,
        timestamp=101,
        outcome_event_id="treatment-score-event",
        item_id="net-1",
        content_version="2026.1",
    )
    frame = pilot.decision_frame(completed_only=True)
    assert frame["experiment_arm"].tolist() == ["treatment"]
    assert frame["model_artifact_id"].tolist() == ["orchid-empirical-2026-08-30"]


def test_sqlite_assignments_and_decisions_survive_adapter_restart(tmp_path: Path) -> None:
    database = tmp_path / "pilot.sqlite"
    catalog = _catalog(required_first=False)
    validated_catalog = PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1")
    request = PilotRequest(
        request_id="request-restart",
        user_id="learner-restart",
        course_id="networking",
        module_id="routing",
        timestamp=100,
        candidate_item_ids=("net-1",),
        stratum="baseline-high",
    )

    decision_store = SQLiteDecisionStore(database)
    assignment_store = SQLiteExperimentAssignmentStore(database)
    first = AdaptivePracticePilot(
        _ranker(catalog, decision_store=decision_store),
        validated_catalog,
        experiment_id="routing-pilot",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=1.0,
        assignment_store=assignment_store,
    ).serve(request)
    decision_store.close()
    assignment_store.close()

    reopened_decisions = SQLiteDecisionStore(database)
    reopened_assignments = SQLiteExperimentAssignmentStore(database)
    try:
        restarted = AdaptivePracticePilot(
            _ranker(catalog, decision_store=reopened_decisions),
            validated_catalog,
            experiment_id="routing-pilot",
            model_artifact_id="orchid-empirical-2026-08-30",
            authored_policy_version="routing-authored-v1",
            treatment_fraction=1.0,
            assignment_store=reopened_assignments,
        ).serve(request)

        assert restarted.decision == first.decision
        assert restarted.arm == "treatment"
        assert restarted.chosen_content_version == "2026.1"
        assert reopened_assignments.get_manifest("routing-pilot") is not None
    finally:
        reopened_decisions.close()
        reopened_assignments.close()


def test_request_ids_are_namespaced_by_experiment() -> None:
    catalog = _catalog(required_first=True)
    ranker = _ranker(catalog)
    first = AdaptivePracticePilot(
        ranker,
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="routing-pilot-a",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=0.0,
    ).serve(
        PilotRequest(
            request_id="lms-request-1",
            user_id="learner",
            course_id="networking",
            module_id="routing",
            timestamp=100,
            candidate_item_ids=("net-1",),
        )
    )
    second = AdaptivePracticePilot(
        ranker,
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="routing-pilot-b",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=0.0,
    ).serve(
        PilotRequest(
            request_id="lms-request-1",
            user_id="learner",
            course_id="networking",
            module_id="routing",
            timestamp=100,
            candidate_item_ids=("net-1",),
        )
    )

    assert first.decision.decision_id != second.decision.decision_id
    assert len(ranker.decision_store.decisions()) == 2


def test_manifest_rejects_changed_allocation_and_existing_log_configuration(tmp_path: Path) -> None:
    catalog = _catalog(required_first=False)
    snapshot = PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1")
    database = tmp_path / "manifest.sqlite"
    decisions = SQLiteDecisionStore(database)
    assignments = SQLiteExperimentAssignmentStore(database)
    request = PilotRequest(
        request_id="manifest-request",
        user_id="learner",
        course_id="networking",
        module_id="routing",
        timestamp=100,
        candidate_item_ids=("net-1",),
        stratum="baseline-low",
    )
    try:
        pilot = AdaptivePracticePilot(
            _ranker(catalog, decision_store=decisions),
            snapshot,
            experiment_id="manifest-pilot",
            model_artifact_id="orchid-empirical-2026-08-30",
            authored_policy_version="routing-authored-v1",
            treatment_fraction=1.0,
            assignment_store=assignments,
        )
        pilot.serve(request)

        with pytest.raises(ValueError, match="different immutable manifest"):
            AdaptivePracticePilot(
                _ranker(catalog, decision_store=decisions),
                snapshot,
                experiment_id="manifest-pilot",
                model_artifact_id="orchid-empirical-2026-08-30",
                authored_policy_version="routing-authored-v1",
                treatment_fraction=0.0,
                assignment_store=assignments,
            )

        # A misplaced fresh assignment store cannot weaken validation of an
        # already durable decision in the shared serving audit store.
        changed_adapter = AdaptivePracticePilot(
            _ranker(catalog, decision_store=decisions),
            snapshot,
            experiment_id="manifest-pilot",
            model_artifact_id="orchid-empirical-rebuilt",
            authored_policy_version="routing-authored-v1",
            treatment_fraction=1.0,
        )
        with pytest.raises(ValueError, match="immutable experiment configuration"):
            changed_adapter.serve(request)
    finally:
        decisions.close()
        assignments.close()


def test_treatment_rejects_a_ranker_response_outside_approved_eligibility(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _catalog(required_first=False)
    catalog["prerequisites"] = pd.Series([[], [], [], []], dtype=object)
    pilot = AdaptivePracticePilot(
        _ranker(catalog),
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="candidate-integrity-pilot",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=1.0,
    )
    original = pilot.ranker.recommend_and_log

    def malformed_response(*args: object, **kwargs: object) -> tuple[list[object], object]:
        recommendations, decision = original(*args, **kwargs)
        return recommendations, replace(decision, candidate_item_ids=("net-1",))

    monkeypatch.setattr(pilot.ranker, "recommend_and_log", malformed_response)
    with pytest.raises(ValueError, match="exact approved eligibility"):
        pilot.serve(
            PilotRequest(
                request_id="candidate-integrity-request",
                user_id="learner",
                course_id="networking",
                module_id="routing",
                timestamp=100,
                candidate_item_ids=("net-1", "net-2"),
            )
        )


def test_stratum_changes_deterministic_assignment_hash() -> None:
    catalog = _catalog(required_first=True)
    pilot = AdaptivePracticePilot(
        _ranker(catalog),
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="stratified-pilot",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=0.5,
    )
    learner = next(
        learner_id
        for learner_id in (f"learner-{index}" for index in range(100))
        if pilot._is_treatment(learner_id, "baseline-low")
        != pilot._is_treatment(learner_id, "baseline-high")
    )

    assert pilot._is_treatment(learner, "baseline-low") != pilot._is_treatment(learner, "baseline-high")


def _record_delivery(pilot: AdaptivePracticePilot, decision_id: str, *, prefix: str, timestamp: float = 100.0) -> None:
    pilot.record_rendered(
        decision_id,
        event_id=f"{prefix}-rendered",
        item_id="net-1",
        content_version="2026.1",
        timestamp=timestamp + 0.1,
    )
    pilot.record_submitted(
        decision_id,
        event_id=f"{prefix}-submitted",
        item_id="net-1",
        content_version="2026.1",
        timestamp=timestamp + 0.2,
    )


def test_lifecycle_requires_the_actual_render_and_keeps_control_outcomes_audit_only() -> None:
    catalog = _catalog(required_first=True)
    pilot = AdaptivePracticePilot(
        _ranker(catalog),
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="lifecycle-pilot",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=0.0,
    )
    served = pilot.serve(
        PilotRequest(
            request_id="lifecycle-request",
            user_id="new-learner",
            course_id="networking",
            module_id="routing",
            timestamp=100,
            candidate_item_ids=("net-1",),
        )
    )
    assert pilot.ranker.recommender_ is not None
    tracer = pilot.ranker.recommender_.tracer_
    user_count_before = tracer._user_count.get("new-learner", 0.0)

    with pytest.raises(RuntimeError, match="record_rendered and record_submitted"):
        pilot.record_scored(
            served.decision.decision_id,
            outcome_event_id="lifecycle-score",
            item_id="net-1",
            content_version="2026.1",
            outcome=1,
            timestamp=101,
        )
    pilot.record_rendered(
        served.decision.decision_id,
        event_id="lifecycle-rendered",
        item_id="net-1",
        content_version="2026.1",
        timestamp=100.1,
    )
    with pytest.raises(ValueError, match="content_version does not match"):
        pilot.record_submitted(
            served.decision.decision_id,
            event_id="lifecycle-wrong-submission",
            item_id="net-1",
            content_version="wrong-revision",
            timestamp=100.2,
        )
    pilot.record_submitted(
        served.decision.decision_id,
        event_id="lifecycle-submitted",
        item_id="net-1",
        content_version="2026.1",
        timestamp=100.2,
    )
    stored = pilot.record_scored(
        served.decision.decision_id,
        outcome_event_id="lifecycle-score",
        item_id="net-1",
        content_version="2026.1",
        outcome=1,
        timestamp=101,
    )

    assert stored.apply_state is False
    assert stored.update_global is False
    assert tracer._user_count.get("new-learner", 0.0) == user_count_before
    exported = pilot.analysis_frame()
    assert exported["rendered_event_id"].tolist() == ["lifecycle-rendered"]
    assert exported["submitted_event_id"].tolist() == ["lifecycle-submitted"]
    assert exported["scored_event_id"].tolist() == ["lifecycle-score"]


def test_shadow_and_halt_keep_assignments_but_force_authored_delivery() -> None:
    catalog = _catalog(required_first=False)
    pilot = AdaptivePracticePilot(
        _ranker(catalog),
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="operations-pilot",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=1.0,
    )
    pilot.set_mode("shadow", event_id="shadow-start", timestamp=10, reason="validate delivery")
    shadow = pilot.serve(
        PilotRequest(
            request_id="shadow-request",
            user_id="shadow-learner",
            course_id="networking",
            module_id="routing",
            course_run_id="run-1",
            timestamp=100,
            candidate_item_ids=("net-1",),
        )
    )
    assert shadow.arm == "treatment"
    assert shadow.effective_arm == "control"
    assert shadow.mode == "shadow"
    assert shadow.decision.policy_name == "authored-static"
    assert shadow.reason_code == "SHADOW_AUTHORED_CONTROL"
    assert any(
        event.event_type == "shadow_proposal"
        for event in pilot.lifecycle_store.events("operations-pilot", shadow.decision.decision_id)
    )

    pilot.set_mode("halted", event_id="halt-now", timestamp=110, reason="operator stop")
    halted = pilot.serve(
        PilotRequest(
            request_id="halted-request",
            user_id="halted-learner",
            course_id="networking",
            module_id="routing",
            timestamp=120,
            candidate_item_ids=("net-1",),
        )
    )
    assert halted.arm == "treatment"
    assert halted.effective_arm == "control"
    assert halted.mode == "halted"
    assert halted.reason_code == "KILL_SWITCH"
    assert any(
        event.event_type == "fallback"
        for event in pilot.lifecycle_store.events("operations-pilot", halted.decision.decision_id)
    )
    with pytest.raises(ValueError, match="cannot be re-enabled"):
        pilot.set_mode("active", event_id="unsafe-restart", timestamp=121)

    _record_delivery(pilot, shadow.decision.decision_id, prefix="shadow")
    pilot.record_scored(
        shadow.decision.decision_id,
        outcome_event_id="shadow-score",
        item_id="net-1",
        content_version="2026.1",
        outcome=1,
        timestamp=101,
    )
    assessments = pilot.import_delayed_assessments(
        pd.DataFrame(
            {
                "assessment_event_id": ["delayed-assessment-1"],
                "user_id": ["shadow-learner"],
                "course_run_id": ["run-1"],
                "assessment_form_version": ["assessment-v1"],
                "timestamp": [200],
                "score": [0.8],
                "independent": [True],
            }
        )
    )
    assert assessments["score"].tolist() == [0.8]
    analysis = pilot.analysis_frame()
    shadow_row = analysis.loc[analysis["decision_id"] == shadow.decision.decision_id].iloc[0]
    assert shadow_row["independent_assessments"][0]["score"] == 0.8
    assert shadow_row["shadow_proposal"]["kind"] == "adaptive"


def test_analysis_grouping_key_normalizes_pandas_missing_values() -> None:
    assert _analysis_grouping_key(None) == "null"
    assert _analysis_grouping_key(pd.NA) == "null"
    assert _analysis_grouping_key(np.nan) == "null"


def test_sqlite_lifecycle_and_operating_mode_survive_restart(tmp_path: Path) -> None:
    database = tmp_path / "operations.sqlite"
    catalog = _catalog(required_first=False)
    snapshot = PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1")
    request = PilotRequest(
        request_id="persistent-shadow-request",
        user_id="persistent-learner",
        course_id="networking",
        module_id="routing",
        timestamp=100,
        candidate_item_ids=("net-1",),
    )
    decisions = SQLiteDecisionStore(database)
    assignments = SQLiteExperimentAssignmentStore(database)
    lifecycle = SQLitePilotLifecycleStore(database)
    try:
        first_pilot = AdaptivePracticePilot(
            _ranker(catalog, decision_store=decisions),
            snapshot,
            experiment_id="persistent-operations-pilot",
            model_artifact_id="orchid-empirical-2026-08-30",
            authored_policy_version="routing-authored-v1",
            treatment_fraction=1.0,
            assignment_store=assignments,
            lifecycle_store=lifecycle,
        )
        first_pilot.set_mode("shadow", event_id="persistent-shadow-start", timestamp=10)
        first = first_pilot.serve(request)
        assert first.mode == "shadow"
    finally:
        lifecycle.close()
        assignments.close()
        decisions.close()

    reopened_decisions = SQLiteDecisionStore(database)
    reopened_assignments = SQLiteExperimentAssignmentStore(database)
    reopened_lifecycle = SQLitePilotLifecycleStore(database)
    try:
        restarted = AdaptivePracticePilot(
            _ranker(catalog, decision_store=reopened_decisions),
            snapshot,
            experiment_id="persistent-operations-pilot",
            model_artifact_id="orchid-empirical-2026-08-30",
            authored_policy_version="routing-authored-v1",
            treatment_fraction=1.0,
            assignment_store=reopened_assignments,
            lifecycle_store=reopened_lifecycle,
        ).serve(request)
        assert restarted.decision == first.decision
        assert restarted.mode == "shadow"
        assert any(
            event.event_type == "shadow_proposal"
            for event in reopened_lifecycle.events("persistent-operations-pilot", restarted.decision.decision_id)
        )
    finally:
        reopened_lifecycle.close()
        reopened_assignments.close()
        reopened_decisions.close()


def test_course_run_namespaces_request_ids_and_explanations_match_logged_action() -> None:
    catalog = _catalog(required_first=False)
    pilot = AdaptivePracticePilot(
        _ranker(catalog),
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="course-run-pilot",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=1.0,
    )
    first = pilot.serve(
        PilotRequest(
            request_id="shared-lms-request",
            user_id="learner",
            course_id="networking",
            module_id="routing",
            course_run_id="run-a",
            timestamp=100,
            candidate_item_ids=("net-1",),
        )
    )
    second = pilot.serve(
        PilotRequest(
            request_id="shared-lms-request",
            user_id="learner",
            course_id="networking",
            module_id="routing",
            course_run_id="run-b",
            timestamp=101,
            candidate_item_ids=("net-1",),
        )
    )
    assert first.decision.decision_id != second.decision.decision_id
    explanation = next(
        event.payload
        for event in pilot.lifecycle_store.events("course-run-pilot", first.decision.decision_id)
        if event.event_type == "explanation"
    )
    assert explanation["chosen_item_id"] == first.decision.chosen_item_id
    assert explanation["ranked_candidates"][0]["score"] == first.decision.scores[0]


def test_invalid_or_globally_reused_score_ids_do_not_create_lifecycle_evidence() -> None:
    catalog = _catalog(required_first=True)
    pilot = AdaptivePracticePilot(
        _ranker(catalog),
        PilotCatalog.from_frame(catalog, catalog_version="networking-2026.1"),
        experiment_id="score-validation-pilot",
        model_artifact_id="orchid-empirical-2026-08-30",
        authored_policy_version="routing-authored-v1",
        treatment_fraction=0.0,
    )
    first = pilot.serve(
        PilotRequest(
            request_id="score-first",
            user_id="first",
            course_id="networking",
            module_id="routing",
            timestamp=100,
            candidate_item_ids=("net-1",),
        )
    )
    _record_delivery(pilot, first.decision.decision_id, prefix="score-first")
    with pytest.raises(ValueError, match="outcome must be exactly"):
        pilot.record_scored(
            first.decision.decision_id,
            outcome_event_id="invalid-score-id",
            item_id="net-1",
            content_version="2026.1",
            outcome=2,
            timestamp=101,
        )
    assert pilot.lifecycle_store.get_event("invalid-score-id") is None
    pilot.record_scored(
        first.decision.decision_id,
        outcome_event_id="global-score-id",
        item_id="net-1",
        content_version="2026.1",
        outcome=1,
        timestamp=101,
    )
    second = pilot.serve(
        PilotRequest(
            request_id="score-second",
            user_id="second",
            course_id="networking",
            module_id="routing",
            timestamp=102,
            candidate_item_ids=("net-1",),
        )
    )
    _record_delivery(pilot, second.decision.decision_id, prefix="score-second", timestamp=102)
    with pytest.raises(ValueError, match="already belongs"):
        pilot.record_scored(
            second.decision.decision_id,
            outcome_event_id="global-score-id",
            item_id="net-1",
            content_version="2026.1",
            outcome=1,
            timestamp=103,
        )
    assert pilot.lifecycle_store.get_event("global-score-id").decision_id == first.decision.decision_id
