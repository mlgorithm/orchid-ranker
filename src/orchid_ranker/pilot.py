"""Reference integration primitives for a frozen adaptive-practice pilot.

The module deliberately keeps an LMS's responsibilities outside the ranker:
curriculum eligibility, sticky experiment assignment, the static authored
control, and evidence that a decision was actually rendered, submitted, and
scored. It is a small local reference adapter, not a hosted LMS.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ._pilot_primitives import (
    ExperimentAssignment,
    ExperimentAssignmentStore,
    ExperimentManifest,
    ExperimentOperation,
    InMemoryExperimentAssignmentStore,
    InMemoryPilotLifecycleStore,
    PilotArm,
    PilotCatalog,
    PilotCatalogSchema,
    PilotDecision,
    PilotDeliveryEvent,
    PilotEligibility,
    PilotEventType,
    PilotExercise,
    PilotLifecycleStore,
    PilotMode,
    PilotRequest,
    SQLiteExperimentAssignmentStore,
    SQLitePilotLifecycleStore,
    _analysis_grouping_key,
    _delivery_reason,
    _derived_model_config_identity,
    _effective_pilot_arm,
    _explanation_from_logged_decision,
    _fingerprint,
    _finite_float,
    _logged_content_versions,
    _namespaced_decision_id,
    _pilot_arm,
    _pilot_metadata,
    _pilot_metadata_or_none,
    _pilot_metadata_or_none_from_value,
    _pilot_mode,
    _pilot_request_fingerprint,
    _require_assignment_store,
    _require_exact_eligible_candidates,
    _require_lifecycle_store,
    _require_matching_pilot_configuration,
    _require_matching_pilot_request,
    _require_nonempty_string,
    _require_same_manifest,
    _shadow_proposal,
    _stable_unit_interval,
    _system_event_id,
    _validate_score_payload,
)
from .adaptive_ranker import AdaptiveRanker
from .adaptive_schema import DecisionOutcome, LoggedDecision, normalize_timestamp, stable_context_hash

__all__ = [
    "AdaptivePracticePilot",
    "ExperimentAssignment",
    "ExperimentAssignmentStore",
    "ExperimentManifest",
    "ExperimentOperation",
    "InMemoryExperimentAssignmentStore",
    "InMemoryPilotLifecycleStore",
    "PilotCatalog",
    "PilotCatalogSchema",
    "PilotDecision",
    "PilotDeliveryEvent",
    "PilotEligibility",
    "PilotExercise",
    "PilotRequest",
    "SQLiteExperimentAssignmentStore",
    "SQLitePilotLifecycleStore",
]
class AdaptivePracticePilot:
    """Route an LMS request to a frozen Orchid treatment or authored control.

    The class implements delivery mechanics for a first controlled pilot. It
    intentionally does not claim to power, analyse, or preregister the study.
    Keep the catalog snapshot, ranker configuration, model artifact ID, and
    authored control version fixed until the delayed assessment is complete.
    """

    def __init__(
        self,
        ranker: AdaptiveRanker,
        catalog: PilotCatalog,
        *,
        experiment_id: str,
        model_artifact_id: str,
        authored_policy_version: str,
        eligibility_rule_version: str = "v1",
        treatment_fraction: float = 0.5,
        randomization_salt: str = "orchid-pilot",
        model_config_identity: Optional[str] = None,
        assignment_store: Optional[ExperimentAssignmentStore] = None,
        lifecycle_store: Optional[PilotLifecycleStore] = None,
        initial_mode: PilotMode = "active",
    ) -> None:
        if not ranker.is_fitted:
            raise RuntimeError("fit AdaptiveRanker before creating an AdaptivePracticePilot")
        if ranker.offline_policy_ is not None:
            raise ValueError("a first pilot must not use an offline CQL policy; use the frozen adaptive baseline")
        if ranker.config.kt_backbone != "empirical":
            raise ValueError("the first frozen pilot requires kt_backbone='empirical'")
        _require_nonempty_string(experiment_id, "experiment_id")
        _require_nonempty_string(model_artifact_id, "model_artifact_id")
        _require_nonempty_string(authored_policy_version, "authored_policy_version")
        _require_nonempty_string(eligibility_rule_version, "eligibility_rule_version")
        _require_nonempty_string(randomization_salt, "randomization_salt")
        if model_config_identity is not None:
            _require_nonempty_string(model_config_identity, "model_config_identity")
        if not 0.0 <= float(treatment_fraction) <= 1.0:
            raise ValueError("treatment_fraction must be in [0, 1]")
        self.ranker = ranker
        self.catalog = catalog
        self.experiment_id = experiment_id
        self.model_artifact_id = model_artifact_id
        self.authored_policy_version = authored_policy_version
        self.eligibility_rule_version = eligibility_rule_version
        self.treatment_fraction = float(treatment_fraction)
        self.randomization_salt = randomization_salt
        self.assignment_store = assignment_store or InMemoryExperimentAssignmentStore()
        _require_assignment_store(self.assignment_store)
        self.lifecycle_store = lifecycle_store or (
            SQLitePilotLifecycleStore(self.assignment_store.database)
            if isinstance(self.assignment_store, SQLiteExperimentAssignmentStore)
            else InMemoryPilotLifecycleStore()
        )
        _require_lifecycle_store(self.lifecycle_store)
        self.manifest = ExperimentManifest(
            experiment_id=experiment_id,
            catalog_version=catalog.catalog_version,
            catalog_content_digest=catalog.content_digest,
            model_artifact_id=model_artifact_id,
            model_config_identity=(
                _derived_model_config_identity(ranker)
                if model_config_identity is None
                else model_config_identity
            ),
            authored_policy_version=authored_policy_version,
            eligibility_rule_version=eligibility_rule_version,
            allocation_method="stratified-stable-hash-v1",
            treatment_fraction=self.treatment_fraction,
            randomization_salt_digest=_fingerprint({"randomization_salt": randomization_salt}),
        )
        stored_manifest, _ = self.assignment_store.create_manifest(self.manifest)
        _require_same_manifest(stored_manifest, self.manifest)
        self.manifest = stored_manifest
        if initial_mode not in {"aa", "shadow", "active", "halted"}:
            raise ValueError("initial_mode must be one of 'aa', 'shadow', 'active', or 'halted'")
        initial_operation = ExperimentOperation(
            experiment_id=self.experiment_id,
            mode=initial_mode,
            operation_event_id=_system_event_id(self.experiment_id, "initial-mode"),
            timestamp=0.0,
            reason="initial pilot mode",
        )
        existing_operation = self.assignment_store.get_operation(self.experiment_id)
        if existing_operation is None:
            self.operation, _ = self.assignment_store.transition_operation(initial_operation)
        else:
            self.operation = existing_operation
        for operation in self.assignment_store.operation_events(self.experiment_id):
            self._record_operation_event(operation)

    @property
    def mode(self) -> PilotMode:
        """Return the current durable delivery mode."""
        operation = self.assignment_store.get_operation(self.experiment_id)
        if operation is None:
            raise RuntimeError("pilot operation state is missing")
        self.operation = operation
        return operation.mode

    def set_mode(
        self,
        mode: PilotMode,
        *,
        event_id: str,
        timestamp: Any,
        reason: Optional[str] = None,
    ) -> ExperimentOperation:
        """Persist a rollout transition; a halt is intentionally irreversible."""
        operation = ExperimentOperation(
            experiment_id=self.experiment_id,
            mode=mode,
            operation_event_id=event_id,
            timestamp=normalize_timestamp(timestamp),
            reason=reason,
        )
        stored, _ = self.assignment_store.transition_operation(operation)
        self._record_operation_event(stored)
        self.operation = stored
        return stored

    def _record_operation_event(self, operation: ExperimentOperation) -> None:
        self.lifecycle_store.create_event(
            PilotDeliveryEvent(
                event_id=_system_event_id(self.experiment_id, "mode-change", operation.operation_event_id),
                experiment_id=self.experiment_id,
                event_type="mode_change",
                timestamp=operation.timestamp,
                payload={
                    "mode": operation.mode,
                    "operation_event_id": operation.operation_event_id,
                    "reason": operation.reason,
                },
            )
        )

    def serve(self, request: PilotRequest) -> PilotDecision:
        """Return an idempotent control or treatment decision for one request."""
        timestamp = normalize_timestamp(request.timestamp)
        eligibility = self.catalog.eligible_items(
            course_id=request.course_id,
            module_id=request.module_id,
            completed_item_ids=request.completed_item_ids,
            candidate_item_ids=request.candidate_item_ids,
        )
        request_fingerprint = _pilot_request_fingerprint(
            request,
            timestamp=timestamp,
            catalog_version=self.catalog.catalog_version,
            eligibility_rule_version=self.eligibility_rule_version,
            eligible_item_ids=eligibility.item_ids,
        )
        decision_id = _namespaced_decision_id(
            self.experiment_id,
            request.request_id,
            request.course_run_id,
        )
        existing = self.ranker.decision_store.get_decision(decision_id)
        if existing is not None:
            metadata = _pilot_metadata(existing)
            _require_matching_pilot_request(metadata, request_fingerprint, decision_id)
            _require_matching_pilot_configuration(metadata, self.manifest, existing)
            _require_exact_eligible_candidates(existing, eligibility, decision_id)
            arm = _pilot_arm(metadata, decision_id)
            self._persist_assignment(request, arm)
            self._record_automatic_lifecycle_events(existing, metadata)
            return self._pilot_decision_from_logged(existing, metadata, eligibility=eligibility)

        mode = self.mode
        if timestamp < self.operation.timestamp:
            raise ValueError("new pilot request timestamp precedes the current delivery mode transition")
        assignment = self._assignment_for(request)
        effective_arm: PilotArm = "treatment" if mode == "active" and assignment.arm == "treatment" else "control"
        metadata = self._decision_metadata(
            request,
            timestamp=timestamp,
            eligibility=eligibility,
            arm=assignment.arm,
            effective_arm=effective_arm,
            mode=mode,
            request_fingerprint=request_fingerprint,
        )
        if effective_arm == "treatment":
            recommendations, decision = self.ranker.recommend_and_log(
                request.user_id,
                eligibility.item_ids,
                timestamp=timestamp,
                top_k=1,
                exploration=0.0,
                min_outcome_probability=0.0,
                max_outcome_probability=1.0,
                require_prerequisites=False,
                policy_version=self.model_artifact_id,
                context_hash=stable_context_hash(
                    self.experiment_id,
                    request.course_id,
                    request.module_id,
                    request.user_id,
                ),
                decision_id=decision_id,
                decision_metadata={"pilot": metadata},
            )
            if not recommendations:
                raise ValueError("ranker returned no treatment recommendation for an approved candidate set")
            chosen_item_id = decision.chosen_item_id
            reason_code = "ADAPTIVE_PRACTICE"
        else:
            decision = self._create_authored_decision(
                request,
                timestamp=timestamp,
                eligibility=eligibility,
                metadata=metadata,
            )
            chosen_item_id = decision.chosen_item_id
            reason_code = _delivery_reason(mode, eligibility.reason_code)
        stored_metadata = _pilot_metadata(decision)
        _require_matching_pilot_configuration(stored_metadata, self.manifest, decision)
        _require_exact_eligible_candidates(decision, eligibility, decision_id)
        self._record_automatic_lifecycle_events(decision, stored_metadata)
        version_by_item = dict(zip(eligibility.item_ids, eligibility.content_versions))
        return PilotDecision(
            decision=decision,
            arm=assignment.arm,
            effective_arm=effective_arm,
            mode=mode,
            chosen_content_version=version_by_item[chosen_item_id],
            reason_code=reason_code,
            eligible_item_ids=eligibility.item_ids,
        )

    def observe_decision(
        self,
        decision_id: str,
        *,
        outcome: Optional[Any] = None,
        reward: Optional[float] = None,
        timestamp: Optional[Any] = None,
        category_id: Optional[Any] = None,
        outcome_event_id: Optional[str] = None,
    ) -> DecisionOutcome:
        """Apply a score after the render/submission lifecycle has been recorded.

        New integrations should call :meth:`record_scored`, which writes the
        immutable score event before calling this method. This compatibility
        entrypoint still requires an already-recorded submitted event.
        """
        if not isinstance(outcome_event_id, str) or not outcome_event_id:
            raise ValueError("pilot outcomes require the LMS's unique outcome_event_id")
        decision = self.ranker.decision_store.get_decision(decision_id)
        metadata = None if decision is None else _pilot_metadata_or_none(decision)
        if decision is None or metadata is None:
            raise KeyError(f"unknown pilot decision_id: {decision_id}")
        _require_matching_pilot_configuration(metadata, self.manifest, decision)
        submitted = self._single_event(decision_id, "submitted")
        if submitted is None:
            raise RuntimeError("record_rendered and record_submitted before attaching a pilot outcome")
        return self.record_scored(
            decision_id,
            outcome_event_id=outcome_event_id,
            outcome=outcome,
            reward=reward,
            timestamp=decision.timestamp if timestamp is None else timestamp,
            category_id=category_id,
            item_id=submitted.item_id,
            content_version=submitted.content_version,
        )

    def record_rendered(
        self,
        decision_id: str,
        *,
        event_id: str,
        item_id: Any,
        content_version: Any,
        timestamp: Any,
    ) -> PilotDeliveryEvent:
        """Record the exact item and revision the LMS actually rendered."""
        decision, metadata = self._pilot_logged_decision(decision_id)
        self._require_actual_item(decision, metadata, item_id=item_id, content_version=content_version)
        event_timestamp = normalize_timestamp(timestamp)
        if event_timestamp < decision.timestamp:
            raise ValueError("render timestamp must not precede the serving decision")
        return self._create_single_delivery_event(
            PilotDeliveryEvent(
                event_id=event_id,
                experiment_id=self.experiment_id,
                event_type="rendered",
                timestamp=event_timestamp,
                decision_id=decision_id,
                item_id=item_id,
                content_version=content_version,
            )
        )

    def record_submitted(
        self,
        decision_id: str,
        *,
        event_id: str,
        item_id: Any,
        content_version: Any,
        timestamp: Any,
    ) -> PilotDeliveryEvent:
        """Record a learner submission only for the exact rendered revision."""
        decision, metadata = self._pilot_logged_decision(decision_id)
        self._require_actual_item(decision, metadata, item_id=item_id, content_version=content_version)
        rendered = self._single_event(decision_id, "rendered")
        if rendered is None:
            raise RuntimeError("record_rendered before record_submitted")
        if rendered.item_id != item_id or rendered.content_version != content_version:
            raise ValueError("submitted item/content_version does not match the rendered exercise")
        event_timestamp = normalize_timestamp(timestamp)
        if event_timestamp < rendered.timestamp:
            raise ValueError("submission timestamp must not precede rendering")
        return self._create_single_delivery_event(
            PilotDeliveryEvent(
                event_id=event_id,
                experiment_id=self.experiment_id,
                event_type="submitted",
                timestamp=event_timestamp,
                decision_id=decision_id,
                item_id=item_id,
                content_version=content_version,
                payload={"render_event_id": rendered.event_id},
            )
        )

    def record_scored(
        self,
        decision_id: str,
        *,
        outcome_event_id: str,
        item_id: Any,
        content_version: Any,
        outcome: Optional[Any] = None,
        reward: Optional[float] = None,
        timestamp: Any,
        category_id: Optional[Any] = None,
    ) -> DecisionOutcome:
        """Durably record and apply one score after a matching submission."""
        if not isinstance(outcome_event_id, str) or not outcome_event_id:
            raise ValueError("outcome_event_id must be a non-empty LMS-global event ID")
        _validate_score_payload(outcome=outcome, reward=reward)
        decision, metadata = self._pilot_logged_decision(decision_id)
        linked_event = self.ranker.decision_store.get_outcome_by_event_id(outcome_event_id)
        if linked_event is not None and linked_event.decision_id != decision_id:
            raise ValueError(
                "outcome_event_id already belongs to a different immutable outcome: "
                f"{outcome_event_id}"
            )
        if linked_event is not None and self.lifecycle_store.get_event(outcome_event_id) is None:
            raise ValueError("outcome_event_id was attached outside this pilot delivery lifecycle")
        self._require_actual_item(decision, metadata, item_id=item_id, content_version=content_version)
        submitted = self._single_event(decision_id, "submitted")
        if submitted is None:
            raise RuntimeError("record_rendered and record_submitted before record_scored")
        if submitted.item_id != item_id or submitted.content_version != content_version:
            raise ValueError("scored item/content_version does not match the submitted exercise")
        event_timestamp = normalize_timestamp(timestamp)
        if event_timestamp < submitted.timestamp:
            raise ValueError("score timestamp must not precede submission")
        event = PilotDeliveryEvent(
            event_id=outcome_event_id,
            experiment_id=self.experiment_id,
            event_type="scored",
            timestamp=event_timestamp,
            decision_id=decision_id,
            item_id=item_id,
            content_version=content_version,
            payload={
                "outcome": outcome,
                "reward": reward,
                "category_id": category_id,
                "submission_event_id": submitted.event_id,
            },
        )
        stored = self._create_single_delivery_event(event)
        return self._apply_score_event(stored)

    def recover_scored_events(self) -> list[DecisionOutcome]:
        """Finish every score event whose outcome projection remains pending."""
        recovered: list[DecisionOutcome] = []
        for event in self.lifecycle_store.events(self.experiment_id):
            if event.event_type != "scored":
                continue
            stored = self.ranker.decision_store.get_outcome(event.decision_id or "")
            if stored is None or not self.ranker.decision_store.is_outcome_applied(stored.decision_id):
                recovered.append(self._apply_score_event(event))
        return recovered

    def rebuild_state_from_baseline(self) -> list[DecisionOutcome]:
        """Rebuild adaptive treatment state from a freshly restored baseline.

        This explicit recovery path intentionally ignores the durable
        application checkpoints, which belong to a prior process projection.
        Call it only on a newly fitted/restored ranker baseline, never on a
        live ranker that may already contain the same outcomes.
        """
        for event in self.lifecycle_store.events(self.experiment_id):
            if event.event_type == "scored":
                self._persist_score_event(event)
        return self.ranker.replay_all_outcomes_from_baseline()

    def decision_frame(self, *, completed_only: bool = False) -> pd.DataFrame:
        """Return joined pilot decisions/outcomes for delivery monitoring."""
        frame = self.ranker.decision_log_frame(completed_only=completed_only)
        if frame.empty:
            return frame
        records: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            metadata = _pilot_metadata_or_none_from_value(row.get("policy_metadata"))
            if metadata is None or metadata.get("experiment_id") != self.experiment_id:
                continue
            record = row.to_dict()
            record["experiment_arm"] = metadata["experiment_arm"]
            record["effective_arm"] = metadata["effective_arm"]
            record["delivery_mode"] = metadata["delivery_mode"]
            record["catalog_version"] = metadata["catalog_version"]
            record["course_run_id"] = metadata["course_run_id"]
            record["model_artifact_id"] = metadata["model_artifact_id"]
            record["eligibility_rule_version"] = metadata["eligibility_rule_version"]
            record["reason_code"] = metadata["reason_code"]
            records.append(record)
        return pd.DataFrame(records)

    def assessment_frame(self) -> pd.DataFrame:
        """Return independent delayed-assessment evidence imported for this pilot."""
        records = [
            {
                "assessment_event_id": event.event_id,
                "timestamp": event.timestamp,
                **dict(event.payload),
            }
            for event in self.lifecycle_store.events(self.experiment_id)
            if event.event_type == "assessment"
        ]
        return pd.DataFrame(records)

    def enrollment_frame(self) -> pd.DataFrame:
        """Export durable course-run enrollment, including nonparticipants."""
        records = [
            {
                "enrollment_event_id": event.event_id,
                "enrollment_timestamp": event.timestamp,
                "user_id": event.payload["user_id"],
                "course_run_id": event.payload["course_run_id"],
                "assigned_arm": event.payload["experiment_arm"],
                "stratum": event.payload["stratum"],
            }
            for event in self.lifecycle_store.events(self.experiment_id)
            if event.event_type == "enrollment"
        ]
        return pd.DataFrame(records)

    def import_delayed_assessments(self, assessments: pd.DataFrame) -> pd.DataFrame:
        """Import independent delayed outcomes without feeding adaptive state.

        The assessment importer deliberately requires an explicit independence
        flag. A practice item or a score that has already influenced the
        treatment ranker cannot be used as the pilot's retained-mastery
        outcome. A learner assigned at enrollment can be assessed even if they
        never requested practice; keep those learners in the study roster.
        """
        required = {
            "assessment_event_id",
            "user_id",
            "course_run_id",
            "assessment_form_version",
            "timestamp",
            "score",
            "independent",
        }
        if not isinstance(assessments, pd.DataFrame):
            raise TypeError("assessments must be a pandas DataFrame")
        missing = sorted(required - set(assessments.columns))
        if missing:
            raise ValueError(f"assessments are missing required columns: {missing}")
        for _, row in assessments.iterrows():
            event_id = row["assessment_event_id"]
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("assessment_event_id must be a non-empty string")
            if not isinstance(row["independent"], (bool, np.bool_)) or not bool(row["independent"]):
                raise ValueError("delayed assessments must be explicitly marked independent=True")
            score = _finite_float(row["score"], field="score", row_index=row.name)
            assignment = self.assignment_store.get_assignment(self.experiment_id, row["user_id"])
            if assignment is None:
                raise KeyError("delayed assessment belongs to a learner without a pilot assignment")
            assessment_timestamp = normalize_timestamp(row["timestamp"])
            participated = any(
                decision.user_id == row["user_id"]
                and (metadata := _pilot_metadata_or_none(decision)) is not None
                and metadata.get("experiment_id") == self.experiment_id
                and _analysis_grouping_key(metadata.get("course_run_id"))
                == _analysis_grouping_key(row["course_run_id"])
                and decision.timestamp <= assessment_timestamp
                for decision in self.ranker.decision_store.decisions()
            )
            enrolled = any(
                event.event_type == "enrollment"
                and _analysis_grouping_key(event.payload.get("user_id"))
                == _analysis_grouping_key(row["user_id"])
                and _analysis_grouping_key(event.payload.get("course_run_id"))
                == _analysis_grouping_key(row["course_run_id"])
                and event.timestamp <= assessment_timestamp
                for event in self.lifecycle_store.events(self.experiment_id)
            )
            if not participated and not enrolled:
                raise ValueError("delayed assessment course_run_id has no matching pilot participation or enrollment")
            event = PilotDeliveryEvent(
                event_id=event_id,
                experiment_id=self.experiment_id,
                event_type="assessment",
                timestamp=assessment_timestamp,
                payload={
                    "user_id": row["user_id"],
                    "course_run_id": row["course_run_id"],
                    "assessment_form_version": row["assessment_form_version"],
                    "score": score,
                    "independent": True,
                    "experiment_arm": assignment.arm,
                    "stratum": assignment.stratum,
                },
            )
            self.lifecycle_store.create_event(event)
        return self.assessment_frame()

    def analysis_frame(self) -> pd.DataFrame:
        """Export a joined, auditable delivery frame for pilot analysis."""
        decisions = self.decision_frame(completed_only=False)
        if decisions.empty:
            return decisions
        event_records: dict[str, dict[str, Any]] = {}
        for event in self.lifecycle_store.events(self.experiment_id):
            if event.decision_id is None:
                continue
            record = event_records.setdefault(event.decision_id, {})
            if event.event_type in {"rendered", "submitted", "scored", "fallback"}:
                record[f"{event.event_type}_event_id"] = event.event_id
                record[f"{event.event_type}_timestamp"] = event.timestamp
            if event.event_type == "shadow_proposal":
                record["shadow_proposal"] = dict(event.payload)
            if event.event_type == "explanation":
                record["explanation_snapshot"] = dict(event.payload)
        exported = decisions.copy()
        for column in (
            "rendered_event_id",
            "rendered_timestamp",
            "submitted_event_id",
            "submitted_timestamp",
            "scored_event_id",
            "scored_timestamp",
            "fallback_event_id",
            "fallback_timestamp",
            "shadow_proposal",
            "explanation_snapshot",
        ):
            exported[column] = [event_records.get(value, {}).get(column) for value in exported["decision_id"]]
        assessments = self.assessment_frame()
        assessment_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for _, assessment in assessments.iterrows():
            key = (
                _analysis_grouping_key(assessment["user_id"]),
                _analysis_grouping_key(assessment["course_run_id"]),
            )
            assessment_records.setdefault(key, []).append(assessment.to_dict())
        exported["independent_assessments"] = [
            assessment_records.get(
                (_analysis_grouping_key(user_id), _analysis_grouping_key(course_run_id)),
                [],
            )
            for user_id, course_run_id in zip(exported["user_id"], exported["course_run_id"])
        ]
        return exported

    def _pilot_logged_decision(self, decision_id: str) -> tuple[LoggedDecision, Mapping[str, Any]]:
        decision = self.ranker.decision_store.get_decision(decision_id)
        metadata = None if decision is None else _pilot_metadata_or_none(decision)
        if decision is None or metadata is None:
            raise KeyError(f"unknown pilot decision_id: {decision_id}")
        _require_matching_pilot_configuration(metadata, self.manifest, decision)
        return decision, metadata

    def _require_actual_item(
        self,
        decision: LoggedDecision,
        metadata: Mapping[str, Any],
        *,
        item_id: Any,
        content_version: Any,
    ) -> None:
        if item_id != decision.chosen_item_id:
            raise ValueError("delivery event item_id does not match the logged chosen item")
        versions = _logged_content_versions(metadata, decision.decision_id)
        if versions.get(item_id) != content_version:
            raise ValueError("delivery event content_version does not match the logged chosen item")

    def _single_event(self, decision_id: str, event_type: PilotEventType) -> Optional[PilotDeliveryEvent]:
        events = [
            event
            for event in self.lifecycle_store.events(self.experiment_id, decision_id)
            if event.event_type == event_type
        ]
        if len(events) > 1:
            raise ValueError(f"pilot decision has multiple {event_type} events: {decision_id}")
        return events[0] if events else None

    def _create_single_delivery_event(
        self,
        event: PilotDeliveryEvent,
    ) -> PilotDeliveryEvent:
        existing = self._single_event(event.decision_id or "", event.event_type)
        if existing is not None and existing.event_id != event.event_id:
            raise ValueError(f"pilot decision already has a {event.event_type} event: {event.decision_id}")
        stored, _ = self.lifecycle_store.create_event(event)
        return stored

    def _apply_score_event(self, event: PilotDeliveryEvent) -> DecisionOutcome:
        if event.event_type != "scored" or event.decision_id is None:
            raise ValueError("only a scored event can be applied as a pilot outcome")
        stored = self._persist_score_event(event)
        return self.ranker.observe_decision(
            stored.decision_id,
            outcome=stored.outcome,
            reward=stored.reward,
            timestamp=stored.outcome_timestamp,
            category_id=stored.category_id,
            outcome_event_id=stored.outcome_event_id,
            apply_state=stored.apply_state,
            update_global=stored.update_global,
        )

    def _persist_score_event(self, event: PilotDeliveryEvent) -> DecisionOutcome:
        if event.event_type != "scored" or event.decision_id is None:
            raise ValueError("only a scored event can be persisted as a pilot outcome")
        decision, metadata = self._pilot_logged_decision(event.decision_id)
        payload = event.payload
        return self.ranker.persist_decision_outcome(
            decision.decision_id,
            outcome=payload.get("outcome"),
            reward=payload.get("reward"),
            timestamp=event.timestamp,
            category_id=payload.get("category_id"),
            outcome_event_id=event.event_id,
            apply_state=_effective_pilot_arm(metadata, decision.decision_id) == "treatment",
            update_global=False,
        )

    def _record_automatic_lifecycle_events(self, decision: LoggedDecision, metadata: Mapping[str, Any]) -> None:
        explanation = PilotDeliveryEvent(
            event_id=_system_event_id(decision.decision_id, "explanation"),
            experiment_id=self.experiment_id,
            event_type="explanation",
            timestamp=decision.timestamp,
            decision_id=decision.decision_id,
            payload=_explanation_from_logged_decision(decision, metadata),
        )
        self._create_single_delivery_event(explanation)
        mode = _pilot_mode(metadata, decision.decision_id)
        if mode == "shadow":
            event = PilotDeliveryEvent(
                event_id=_system_event_id(decision.decision_id, "shadow-proposal"),
                experiment_id=self.experiment_id,
                event_type="shadow_proposal",
                timestamp=decision.timestamp,
                decision_id=decision.decision_id,
                payload=_shadow_proposal(self.ranker, decision),
            )
            self._create_single_delivery_event(event)
        if mode == "halted":
            event = PilotDeliveryEvent(
                event_id=_system_event_id(decision.decision_id, "kill-switch"),
                experiment_id=self.experiment_id,
                event_type="fallback",
                timestamp=decision.timestamp,
                decision_id=decision.decision_id,
                payload={"reason_code": "KILL_SWITCH", "assigned_arm": _pilot_arm(metadata, decision.decision_id)},
            )
            self._create_single_delivery_event(event)

    def _assignment_for(self, request: PilotRequest) -> ExperimentAssignment:
        return self.assign(request.user_id, stratum=request.stratum)

    def assign(self, user_id: Any, *, stratum: Optional[Any] = None) -> ExperimentAssignment:
        """Persist a sticky arm at enrollment, before the first practice request.

        Keep an external enrollment roster for analysis, including learners who
        never request an exercise. Assignment is learner-level across course runs.
        """
        existing = self.assignment_store.get_assignment(self.experiment_id, user_id)
        if existing is not None:
            if existing.stratum != stratum:
                raise ValueError("learner already has a sticky assignment with a different stratum")
            return existing
        arm: PilotArm = "treatment" if self._is_treatment(user_id, stratum) else "control"
        proposed = ExperimentAssignment(self.experiment_id, user_id, arm, stratum)
        stored, _ = self.assignment_store.create_assignment(proposed)
        if stored.stratum != stratum:
            raise ValueError("learner already has a sticky assignment with a different stratum")
        return stored

    def enroll(
        self,
        user_id: Any,
        *,
        course_run_id: Any,
        timestamp: Any,
        stratum: Optional[Any] = None,
    ) -> ExperimentAssignment:
        """Persist assignment and immutable course-run enrollment before practice.

        Enrollment lets the assessment importer recognize randomized learners
        who never received a practice decision. The external study roster must
        still include every enrolled learner for intention-to-treat analysis.
        """
        if course_run_id is None or _analysis_grouping_key(course_run_id) == "null":
            raise ValueError("course_run_id is required for explicit enrollment")
        event_id = _system_event_id(self.experiment_id, "enrollment", user_id, course_run_id)
        if self.lifecycle_store.get_event(event_id) is None and any(
            decision.user_id == user_id
            and (metadata := _pilot_metadata_or_none(decision)) is not None
            and metadata.get("experiment_id") == self.experiment_id
            and _analysis_grouping_key(metadata.get("course_run_id")) == _analysis_grouping_key(course_run_id)
            for decision in self.ranker.decision_store.decisions()
        ):
            raise ValueError("course-run enrollment must precede the first pilot practice decision")
        assignment = self.assign(user_id, stratum=stratum)
        self.lifecycle_store.create_event(
            PilotDeliveryEvent(
                event_id=event_id,
                experiment_id=self.experiment_id,
                event_type="enrollment",
                timestamp=normalize_timestamp(timestamp),
                payload={
                    "user_id": user_id,
                    "course_run_id": course_run_id,
                    "experiment_arm": assignment.arm,
                    "stratum": assignment.stratum,
                },
            )
        )
        return assignment

    def _persist_assignment(self, request: PilotRequest, arm: PilotArm) -> None:
        stored, _ = self.assignment_store.create_assignment(
            ExperimentAssignment(self.experiment_id, request.user_id, arm, request.stratum)
        )
        if stored.arm != arm or stored.stratum != request.stratum:
            raise ValueError("persisted assignment conflicts with the logged pilot decision")

    def _is_treatment(self, user_id: Any, stratum: Optional[Any]) -> bool:
        """Allocate independently within each declared baseline stratum."""
        value = _stable_unit_interval(
            self.randomization_salt,
            self.experiment_id,
            "stratum",
            stratum,
            "user_id",
            user_id,
        )
        return value < self.treatment_fraction

    def _decision_metadata(
        self,
        request: PilotRequest,
        *,
        timestamp: float,
        eligibility: PilotEligibility,
        arm: PilotArm,
        effective_arm: PilotArm,
        mode: PilotMode,
        request_fingerprint: str,
    ) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_manifest_digest": self.manifest.digest,
            "experiment_arm": arm,
            "effective_arm": effective_arm,
            "delivery_mode": mode,
            "catalog_version": self.catalog.catalog_version,
            "catalog_content_digest": self.manifest.catalog_content_digest,
            "course_id": request.course_id,
            "module_id": request.module_id,
            "course_run_id": request.course_run_id,
            "source_request_id": request.request_id,
            "model_artifact_id": self.model_artifact_id,
            "model_config_identity": self.manifest.model_config_identity,
            "authored_policy_version": self.authored_policy_version,
            "eligibility_rule_version": self.eligibility_rule_version,
            "allocation_method": self.manifest.allocation_method,
            "allocation_treatment_fraction": self.manifest.treatment_fraction,
            "allocation_salt_digest": self.manifest.randomization_salt_digest,
            "stratum": request.stratum,
            "reason_code": (
                "ADAPTIVE_PRACTICE"
                if effective_arm == "treatment"
                else _delivery_reason(mode, eligibility.reason_code)
            ),
            "candidate_content_versions": [
                [item_id, content_version]
                for item_id, content_version in zip(eligibility.item_ids, eligibility.content_versions)
            ],
            "request_timestamp": timestamp,
            "request_fingerprint": request_fingerprint,
        }

    def _create_authored_decision(
        self,
        request: PilotRequest,
        *,
        timestamp: float,
        eligibility: PilotEligibility,
        metadata: Mapping[str, Any],
    ) -> LoggedDecision:
        scores = tuple(float(len(eligibility.item_ids) - index) for index in range(len(eligibility.item_ids)))
        decision = LoggedDecision(
            user_id=request.user_id,
            timestamp=timestamp,
            candidate_item_ids=eligibility.item_ids,
            chosen_item_id=eligibility.item_ids[0],
            propensity=1.0,
            policy_name="authored-static",
            policy_version=self.authored_policy_version,
            scores=scores,
            context_hash=stable_context_hash(
                self.experiment_id,
                request.course_id,
                request.module_id,
                request.user_id,
            ),
            decision_id=_namespaced_decision_id(
                self.experiment_id,
                request.request_id,
                request.course_run_id,
            ),
            action_probabilities=tuple([1.0, *([0.0] * (len(eligibility.item_ids) - 1))]),
            predicted_outcomes=None,
            policy_metadata={"decision_metadata": {"pilot": dict(metadata)}},
        )
        stored, created = self.ranker.decision_store.create_decision(decision)
        if not created:
            existing_metadata = _pilot_metadata(stored)
            _require_matching_pilot_request(
                existing_metadata,
                str(metadata["request_fingerprint"]),
                decision.decision_id,
            )
        return stored

    def _pilot_decision_from_logged(
        self,
        decision: LoggedDecision,
        metadata: Mapping[str, Any],
        *,
        eligibility: PilotEligibility,
    ) -> PilotDecision:
        arm = _pilot_arm(metadata, decision.decision_id)
        version_by_item = _logged_content_versions(metadata, decision.decision_id)
        try:
            chosen_content_version = version_by_item[decision.chosen_item_id]
        except KeyError as exc:
            raise ValueError(f"logged pilot decision has missing content-version metadata: {decision.decision_id}") from exc
        return PilotDecision(
            decision=decision,
            arm=arm,
            effective_arm=_effective_pilot_arm(metadata, decision.decision_id),
            mode=_pilot_mode(metadata, decision.decision_id),
            chosen_content_version=chosen_content_version,
            reason_code=str(metadata["reason_code"]),
            eligible_item_ids=eligibility.item_ids,
        )
