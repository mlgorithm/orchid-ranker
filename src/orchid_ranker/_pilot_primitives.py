"""Pilot catalog, experiment records, stores, and immutable record helpers."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, cast

import numpy as np
import pandas as pd

from .adaptive_ranker import AdaptiveRanker
from .adaptive_schema import LoggedDecision, normalize_timestamp
from .learning_catalog import LearningCatalogSchema, LearningCatalogValidation, validate_learning_catalog

PilotArm = Literal["control", "treatment"]
PilotMode = Literal["aa", "shadow", "active", "halted"]
PilotEventType = Literal[
    "rendered",
    "submitted",
    "scored",
    "fallback",
    "shadow_proposal",
    "explanation",
    "mode_change",
    "enrollment",
    "assessment",
]


@dataclass(frozen=True)
class ExperimentOperation:
    """The mutable, durable delivery state for an otherwise frozen experiment."""

    experiment_id: str
    mode: PilotMode
    operation_event_id: str
    timestamp: float
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.experiment_id, "experiment_id")
        if self.mode not in {"aa", "shadow", "active", "halted"}:
            raise ValueError("mode must be one of 'aa', 'shadow', 'active', or 'halted'")
        _require_nonempty_string(self.operation_event_id, "operation_event_id")
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))


@dataclass(frozen=True)
class PilotDeliveryEvent:
    """One immutable operational event in the decision-to-score lifecycle."""

    event_id: str
    experiment_id: str
    event_type: PilotEventType
    timestamp: float
    decision_id: Optional[str] = None
    item_id: Optional[Any] = None
    content_version: Optional[Any] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.event_id, "event_id")
        _require_nonempty_string(self.experiment_id, "experiment_id")
        if self.event_type not in {
            "rendered",
            "submitted",
            "scored",
            "fallback",
            "shadow_proposal",
            "explanation",
            "mode_change",
            "enrollment",
            "assessment",
        }:
            raise ValueError("unsupported pilot delivery event type")
        if self.event_type in {"rendered", "submitted", "scored", "fallback", "shadow_proposal", "explanation"}:
            if not isinstance(self.decision_id, str) or not self.decision_id:
                raise ValueError(f"{self.event_type} events require a non-empty decision_id")
        if self.event_type in {"rendered", "submitted", "scored"}:
            if self.item_id is None:
                raise ValueError(f"{self.event_type} events require item_id")
            if self.content_version is None:
                raise ValueError(f"{self.event_type} events require content_version")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        object.__setattr__(self, "payload", _plain_json_value(self.payload))
        object.__setattr__(self, "timestamp", normalize_timestamp(self.timestamp))


@dataclass(frozen=True)
class PilotCatalogSchema:
    """Column contract for an active course catalog used by a pilot.

    A pilot accepts exactly one active content version per ``item_id``. Keep
    historical revisions in the authoring system, then publish one immutable
    catalog snapshot for the experiment.
    """

    item_id_col: str = "item_id"
    content_version_col: Optional[str] = "content_version"
    course_id_col: str = "course_id"
    module_id_col: str = "module_id"
    skill_id_col: Optional[str] = "skill_id"
    category_id_col: Optional[str] = "category_id"
    difficulty_col: str = "difficulty"
    assessment_only_col: str = "assessment_only"
    prerequisites_col: str = "prerequisites"
    available_col: str = "available"
    required_col: str = "required"
    authored_sequence_position_col: str = "authored_sequence_position"

    def learning_catalog_schema(self) -> LearningCatalogSchema:
        """Return the shared authoring-validator schema."""
        return LearningCatalogSchema(
            item_id_col=self.item_id_col,
            content_version_col=self.content_version_col,
            course_id_col=self.course_id_col,
            module_id_col=self.module_id_col,
            skill_id_col=self.skill_id_col,
            category_id_col=self.category_id_col,
            difficulty_col=self.difficulty_col,
            assessment_only_col=self.assessment_only_col,
            prerequisites_col=self.prerequisites_col,
        )


@dataclass(frozen=True)
class PilotExercise:
    """One active, author-approved exercise in the frozen pilot catalog."""

    item_id: Any
    content_version: Any
    course_id: Any
    module_id: Any
    prerequisite_item_ids: tuple[Any, ...]
    assessment_only: bool
    available: bool
    required: bool
    authored_sequence_position: float


@dataclass(frozen=True)
class PilotEligibility:
    """The exact author-approved candidate set for one request."""

    item_ids: tuple[Any, ...]
    content_versions: tuple[Any, ...]
    reason_code: str


@dataclass(frozen=True)
class PilotCatalog:
    """Validated immutable catalog snapshot used to enforce author controls."""

    catalog_version: str
    schema: PilotCatalogSchema
    exercises: tuple[PilotExercise, ...]
    validation: LearningCatalogValidation

    @property
    def content_digest(self) -> str:
        """Return a stable digest of every author-approved catalog field.

        ``catalog_version`` is useful for people and import workflows, but it
        is not enough to make a pilot immutable: a source system can
        accidentally republish different content under the same label.  The
        experiment manifest binds this digest as well as the human-readable
        version.
        """
        return _fingerprint(
            {
                "catalog_version": self.catalog_version,
                "schema": asdict(self.schema),
                "exercises": [asdict(exercise) for exercise in self.exercises],
            }
        )

    @classmethod
    def from_frame(
        cls,
        catalog: pd.DataFrame,
        *,
        catalog_version: str,
        schema: PilotCatalogSchema = PilotCatalogSchema(),
    ) -> "PilotCatalog":
        """Validate and freeze an active authoring snapshot for one pilot."""
        if not isinstance(catalog, pd.DataFrame):
            raise TypeError("catalog must be a pandas DataFrame")
        _require_nonempty_string(catalog_version, "catalog_version")
        if schema.content_version_col is None:
            raise ValueError("a pilot catalog requires a content_version column for rendered-item auditability")
        validation = validate_learning_catalog(
            catalog,
            schema=schema.learning_catalog_schema(),
            require_complete_metadata=True,
        )
        validation.raise_for_errors()

        required_columns = [
            schema.item_id_col,
            schema.course_id_col,
            schema.module_id_col,
            schema.difficulty_col,
            schema.assessment_only_col,
            schema.prerequisites_col,
            schema.available_col,
            schema.required_col,
            schema.authored_sequence_position_col,
        ]
        if schema.content_version_col is not None:
            required_columns.append(schema.content_version_col)
        missing = [column for column in required_columns if column not in catalog.columns]
        if missing:
            raise ValueError(f"pilot catalog is missing required columns: {sorted(missing)}")

        exercises: list[PilotExercise] = []
        exercises_by_identity = validation.exercise_by_identity
        seen_item_ids: set[Any] = set()
        seen_positions: set[tuple[str, str, float]] = set()
        for row_index, row in catalog.iterrows():
            item_id = row[schema.item_id_col]
            if item_id in seen_item_ids:
                raise ValueError(
                    "pilot catalog must publish one active content version per item_id; "
                    f"duplicate item_id={item_id!r} at row {row_index!r}"
                )
            seen_item_ids.add(item_id)
            content_version = None if schema.content_version_col is None else row[schema.content_version_col]
            identity = (item_id,) if schema.content_version_col is None else (item_id, content_version)
            canonical = exercises_by_identity[identity]
            available = _coerce_bool(row[schema.available_col], field=schema.available_col, row_index=row_index)
            required = _coerce_bool(row[schema.required_col], field=schema.required_col, row_index=row_index)
            position = _finite_float(
                row[schema.authored_sequence_position_col],
                field=schema.authored_sequence_position_col,
                row_index=row_index,
            )
            position_key = (repr(canonical.course_id), repr(canonical.module_id), position)
            if position_key in seen_positions:
                raise ValueError(
                    "authored_sequence_position must be unique within a course/module; "
                    f"duplicate position={position!r} at row {row_index!r}"
                )
            seen_positions.add(position_key)
            exercises.append(
                PilotExercise(
                    item_id=item_id,
                    content_version=content_version,
                    course_id=canonical.course_id,
                    module_id=canonical.module_id,
                    prerequisite_item_ids=canonical.prerequisite_item_ids,
                    assessment_only=bool(canonical.assessment_only),
                    available=available,
                    required=required,
                    authored_sequence_position=position,
                )
            )
        return cls(str(catalog_version), schema, tuple(exercises), validation)

    def eligible_items(
        self,
        *,
        course_id: Any,
        module_id: Any,
        completed_item_ids: Sequence[Any],
        candidate_item_ids: Optional[Sequence[Any]],
    ) -> PilotEligibility:
        """Return the exact candidate set after hard authoring constraints.

        The optional candidate set is an additional LMS rule, not an escape
        hatch: every supplied ID must survive the catalog's availability,
        assessment, completion, and prerequisite checks.
        """
        completed = _unique_tuple(completed_item_ids, "completed_item_ids")
        completed_set = set(completed)
        scope = [
            exercise
            for exercise in self.exercises
            if exercise.course_id == course_id and exercise.module_id == module_id
        ]
        if not scope:
            raise ValueError(f"no exercises match course_id={course_id!r}, module_id={module_id!r}")
        requested = None if candidate_item_ids is None else _unique_tuple(candidate_item_ids, "candidate_item_ids")
        scope_ids = {exercise.item_id for exercise in scope}
        if requested is not None:
            unknown = [item_id for item_id in requested if item_id not in scope_ids]
            if unknown:
                raise ValueError(f"candidate_item_ids are outside the course/module catalog scope: {unknown!r}")

        author_eligible = [
            exercise
            for exercise in scope
            if exercise.available
            and not exercise.assessment_only
            and exercise.item_id not in completed_set
            and set(exercise.prerequisite_item_ids).issubset(completed_set)
        ]
        author_eligible.sort(key=lambda exercise: (exercise.authored_sequence_position, str(exercise.item_id)))
        if requested is not None:
            eligible_ids = {exercise.item_id for exercise in author_eligible}
            rejected = [item_id for item_id in requested if item_id not in eligible_ids]
            if rejected:
                raise ValueError(
                    "candidate_item_ids include unavailable, completed, assessment-only, or prerequisite-blocked "
                    f"exercises: {rejected!r}"
                )
        required = [exercise for exercise in author_eligible if exercise.required]
        if required:
            chosen = required[0]
            if requested is not None and chosen.item_id not in requested:
                raise ValueError(
                    "candidate_item_ids exclude the author-required exercise "
                    f"{chosen.item_id!r}; do not bypass a required item"
                )
            return PilotEligibility((chosen.item_id,), (chosen.content_version,), "AUTHORED_REQUIRED")

        if requested is not None:
            requested_set = set(requested)
            author_eligible = [exercise for exercise in author_eligible if exercise.item_id in requested_set]
        if not author_eligible:
            raise ValueError("no exercise is eligible after authoring and LMS constraints")
        return PilotEligibility(
            tuple(exercise.item_id for exercise in author_eligible),
            tuple(exercise.content_version for exercise in author_eligible),
            "AUTHORED_ELIGIBILITY",
        )


@dataclass(frozen=True)
class PilotRequest:
    """One LMS request for the next eligible practice exercise."""

    request_id: str
    user_id: Any
    course_id: Any
    module_id: Any
    timestamp: Any
    course_run_id: Optional[Any] = None
    completed_item_ids: tuple[Any, ...] = ()
    candidate_item_ids: Optional[tuple[Any, ...]] = None
    stratum: Optional[Any] = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.request_id, "request_id")
        object.__setattr__(self, "completed_item_ids", _unique_tuple(self.completed_item_ids, "completed_item_ids"))
        if self.candidate_item_ids is None:
            raise ValueError("candidate_item_ids must be the exact non-null LMS-eligible set")
        object.__setattr__(
            self,
            "candidate_item_ids",
            _unique_tuple(self.candidate_item_ids, "candidate_item_ids"),
        )


@dataclass(frozen=True)
class ExperimentAssignment:
    """Sticky learner-level assignment for the whole pilot."""

    experiment_id: str
    user_id: Any
    arm: PilotArm
    stratum: Optional[Any] = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.experiment_id, "experiment_id")
        if self.arm not in {"control", "treatment"}:
            raise ValueError("arm must be 'control' or 'treatment'")


@dataclass(frozen=True)
class ExperimentManifest:
    """Immutable identity and allocation contract for one pilot experiment."""

    experiment_id: str
    catalog_version: str
    catalog_content_digest: str
    model_artifact_id: str
    model_config_identity: str
    authored_policy_version: str
    eligibility_rule_version: str
    allocation_method: str
    treatment_fraction: float
    randomization_salt_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "catalog_version",
            "catalog_content_digest",
            "model_artifact_id",
            "model_config_identity",
            "authored_policy_version",
            "eligibility_rule_version",
            "allocation_method",
            "randomization_salt_digest",
        ):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if not 0.0 <= float(self.treatment_fraction) <= 1.0:
            raise ValueError("treatment_fraction must be in [0, 1]")

    @property
    def digest(self) -> str:
        """Return the durable identity compared with every pilot decision."""
        return _fingerprint(asdict(self))


class ExperimentAssignmentStore(Protocol):
    """Durable manifest and sticky-assignment contract for one experiment."""

    def get_manifest(self, experiment_id: str) -> Optional[ExperimentManifest]: ...

    def create_manifest(self, manifest: ExperimentManifest) -> tuple[ExperimentManifest, bool]: ...

    def get_assignment(self, experiment_id: str, user_id: Any) -> Optional[ExperimentAssignment]: ...

    def create_assignment(self, assignment: ExperimentAssignment) -> tuple[ExperimentAssignment, bool]: ...

    def get_operation(self, experiment_id: str) -> Optional[ExperimentOperation]: ...

    def transition_operation(self, operation: ExperimentOperation) -> tuple[ExperimentOperation, bool]: ...

    def operation_events(self, experiment_id: str) -> list[ExperimentOperation]: ...


class PilotLifecycleStore(Protocol):
    """Durable append-only evidence for the delivery lifecycle."""

    def get_event(self, event_id: str) -> Optional[PilotDeliveryEvent]: ...

    def create_event(self, event: PilotDeliveryEvent) -> tuple[PilotDeliveryEvent, bool]: ...

    def events(self, experiment_id: str, decision_id: Optional[str] = None) -> list[PilotDeliveryEvent]: ...


class InMemoryExperimentAssignmentStore:
    """Thread-safe assignment store for tests and one-process prototypes."""

    def __init__(self) -> None:
        self._manifests: dict[str, ExperimentManifest] = {}
        self._assignments: dict[tuple[str, str], ExperimentAssignment] = {}
        self._operations: dict[str, ExperimentOperation] = {}
        self._operation_events: dict[str, ExperimentOperation] = {}
        self._lock = threading.RLock()

    def get_manifest(self, experiment_id: str) -> Optional[ExperimentManifest]:
        with self._lock:
            return self._manifests.get(experiment_id)

    def create_manifest(self, manifest: ExperimentManifest) -> tuple[ExperimentManifest, bool]:
        with self._lock:
            existing = self._manifests.get(manifest.experiment_id)
            if existing is None:
                self._manifests[manifest.experiment_id] = manifest
                return manifest, True
            _require_same_manifest(existing, manifest)
            return existing, False

    def get_assignment(self, experiment_id: str, user_id: Any) -> Optional[ExperimentAssignment]:
        with self._lock:
            return self._assignments.get((experiment_id, _identifier_key(user_id)))

    def create_assignment(self, assignment: ExperimentAssignment) -> tuple[ExperimentAssignment, bool]:
        key = (assignment.experiment_id, _identifier_key(assignment.user_id))
        with self._lock:
            existing = self._assignments.get(key)
            if existing is None:
                self._assignments[key] = assignment
                return assignment, True
            _require_same_assignment(existing, assignment)
            return existing, False

    def get_operation(self, experiment_id: str) -> Optional[ExperimentOperation]:
        with self._lock:
            return self._operations.get(experiment_id)

    def transition_operation(self, operation: ExperimentOperation) -> tuple[ExperimentOperation, bool]:
        with self._lock:
            recorded_event = self._operation_events.get(operation.operation_event_id)
            if recorded_event is not None:
                _require_same_operation(recorded_event, operation)
            existing = self._operations.get(operation.experiment_id)
            if existing is None:
                self._operations[operation.experiment_id] = operation
                self._operation_events[operation.operation_event_id] = operation
                return operation, True
            if existing.operation_event_id == operation.operation_event_id:
                _require_same_operation(existing, operation)
                return existing, False
            if existing.mode == "halted" and operation.mode != "halted":
                raise ValueError("a halted experiment cannot be re-enabled; start a new experiment_id")
            if operation.timestamp < existing.timestamp:
                raise ValueError("experiment operation timestamp must not move backwards")
            self._operations[operation.experiment_id] = operation
            self._operation_events[operation.operation_event_id] = operation
            return operation, True

    def operation_events(self, experiment_id: str) -> list[ExperimentOperation]:
        with self._lock:
            return sorted(
                [event for event in self._operation_events.values() if event.experiment_id == experiment_id],
                key=lambda event: (event.timestamp, event.operation_event_id),
            )


class InMemoryPilotLifecycleStore:
    """Thread-safe lifecycle store for tests and one-process prototypes."""

    def __init__(self) -> None:
        self._events: dict[str, PilotDeliveryEvent] = {}
        self._events_by_transition: dict[tuple[str, str, str], PilotDeliveryEvent] = {}
        self._lock = threading.RLock()

    def get_event(self, event_id: str) -> Optional[PilotDeliveryEvent]:
        with self._lock:
            event = self._events.get(str(event_id))
            return None if event is None else _copy_delivery_event(event)

    def create_event(self, event: PilotDeliveryEvent) -> tuple[PilotDeliveryEvent, bool]:
        with self._lock:
            existing = self._events.get(event.event_id)
            if existing is None:
                transition_key = None
                if event.decision_id is not None:
                    transition_key = (event.experiment_id, event.decision_id, event.event_type)
                    existing_transition = self._events_by_transition.get(transition_key)
                    if existing_transition is not None:
                        _require_same_delivery_event(existing_transition, event)
                        return _copy_delivery_event(existing_transition), False
                stored = _copy_delivery_event(event)
                self._events[stored.event_id] = stored
                if transition_key is not None:
                    self._events_by_transition[transition_key] = stored
                return _copy_delivery_event(stored), True
            _require_same_delivery_event(existing, event)
            return _copy_delivery_event(existing), False

    def events(self, experiment_id: str, decision_id: Optional[str] = None) -> list[PilotDeliveryEvent]:
        with self._lock:
            return [
                _copy_delivery_event(event)
                for event in self._events.values()
                if event.experiment_id == experiment_id
                and (decision_id is None or event.decision_id == decision_id)
            ]


class SQLiteExperimentAssignmentStore:
    """SQLite-backed sticky assignments for a single-host pilot service."""

    def __init__(self, database: str | Path) -> None:
        self.database = str(Path(database))
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orchid_pilot_manifests (
                experiment_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orchid_pilot_assignments (
                experiment_id TEXT NOT NULL,
                user_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (experiment_id, user_key)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orchid_pilot_operations (
                experiment_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orchid_pilot_operation_events (
                operation_event_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            INSERT OR IGNORE INTO orchid_pilot_operation_events
                (operation_event_id, experiment_id, payload)
            SELECT
                json_extract(payload, '$.operation_event_id'),
                experiment_id,
                payload
            FROM orchid_pilot_operations
            WHERE json_extract(payload, '$.operation_event_id') IS NOT NULL
            """
        )
        self._connection.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteExperimentAssignmentStore":
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()

    def get_assignment(self, experiment_id: str, user_id: Any) -> Optional[ExperimentAssignment]:
        user_key = _identifier_key(user_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM orchid_pilot_assignments WHERE experiment_id = ? AND user_key = ?",
                (experiment_id, user_key),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return ExperimentAssignment(
            experiment_id=experiment_id,
            user_id=user_id,
            arm=payload["arm"],
            stratum=payload.get("stratum"),
        )

    def get_manifest(self, experiment_id: str) -> Optional[ExperimentManifest]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM orchid_pilot_manifests WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return None if row is None else ExperimentManifest(**json.loads(row[0]))

    def create_manifest(self, manifest: ExperimentManifest) -> tuple[ExperimentManifest, bool]:
        payload = _canonical_json(asdict(manifest))
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT payload FROM orchid_pilot_manifests WHERE experiment_id = ?",
                    (manifest.experiment_id,),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO orchid_pilot_manifests (experiment_id, payload) VALUES (?, ?)",
                        (manifest.experiment_id, payload),
                    )
                    self._connection.commit()
                    return manifest, True
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        existing = ExperimentManifest(**json.loads(row[0]))
        _require_same_manifest(existing, manifest)
        return existing, False

    def create_assignment(self, assignment: ExperimentAssignment) -> tuple[ExperimentAssignment, bool]:
        user_key = _identifier_key(assignment.user_id)
        payload = _canonical_json({"arm": assignment.arm, "stratum": assignment.stratum})
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT payload FROM orchid_pilot_assignments WHERE experiment_id = ? AND user_key = ?",
                    (assignment.experiment_id, user_key),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO orchid_pilot_assignments (experiment_id, user_key, payload) VALUES (?, ?, ?)",
                        (assignment.experiment_id, user_key, payload),
                    )
                    self._connection.commit()
                    return assignment, True
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        existing_payload = json.loads(row[0])
        existing = ExperimentAssignment(
            experiment_id=assignment.experiment_id,
            user_id=assignment.user_id,
            arm=existing_payload["arm"],
            stratum=existing_payload.get("stratum"),
        )
        _require_same_assignment(existing, assignment)
        return existing, False

    def get_operation(self, experiment_id: str) -> Optional[ExperimentOperation]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM orchid_pilot_operations WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return None if row is None else ExperimentOperation(**json.loads(row[0]))

    def transition_operation(self, operation: ExperimentOperation) -> tuple[ExperimentOperation, bool]:
        payload = _canonical_json(asdict(operation))
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT payload FROM orchid_pilot_operations WHERE experiment_id = ?",
                    (operation.experiment_id,),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        """
                        INSERT INTO orchid_pilot_operation_events
                            (operation_event_id, experiment_id, payload)
                        VALUES (?, ?, ?)
                        """,
                        (operation.operation_event_id, operation.experiment_id, payload),
                    )
                    self._connection.execute(
                        "INSERT INTO orchid_pilot_operations (experiment_id, payload) VALUES (?, ?)",
                        (operation.experiment_id, payload),
                    )
                    self._connection.commit()
                    return operation, True
                existing = ExperimentOperation(**json.loads(row[0]))
                if existing.operation_event_id == operation.operation_event_id:
                    self._connection.commit()
                elif existing.mode == "halted" and operation.mode != "halted":
                    raise ValueError("a halted experiment cannot be re-enabled; start a new experiment_id")
                elif operation.timestamp < existing.timestamp:
                    raise ValueError("experiment operation timestamp must not move backwards")
                else:
                    previous_event = self._connection.execute(
                        """
                        SELECT payload FROM orchid_pilot_operation_events
                        WHERE operation_event_id = ?
                        """,
                        (operation.operation_event_id,),
                    ).fetchone()
                    if previous_event is not None:
                        _require_same_operation(
                            ExperimentOperation(**json.loads(previous_event[0])),
                            operation,
                        )
                    else:
                        self._connection.execute(
                            """
                            INSERT INTO orchid_pilot_operation_events
                                (operation_event_id, experiment_id, payload)
                            VALUES (?, ?, ?)
                            """,
                            (operation.operation_event_id, operation.experiment_id, payload),
                        )
                    self._connection.execute(
                        "UPDATE orchid_pilot_operations SET payload = ? WHERE experiment_id = ?",
                        (payload, operation.experiment_id),
                    )
                    self._connection.commit()
                    return operation, True
            except BaseException:
                self._connection.rollback()
                raise
        _require_same_operation(existing, operation)
        return existing, False

    def operation_events(self, experiment_id: str) -> list[ExperimentOperation]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload FROM orchid_pilot_operation_events
                WHERE experiment_id = ? ORDER BY rowid
                """,
                (experiment_id,),
            ).fetchall()
        return [ExperimentOperation(**json.loads(row[0])) for row in rows]


class SQLitePilotLifecycleStore:
    """SQLite-backed append-only evidence store for a single-host pilot."""

    def __init__(self, database: str | Path) -> None:
        self.database = str(Path(database))
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orchid_pilot_delivery_events (
                event_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                decision_id TEXT,
                event_type TEXT,
                payload TEXT NOT NULL
            )
            """
        )
        self._ensure_transition_index()
        self._connection.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLitePilotLifecycleStore":
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()

    def get_event(self, event_id: str) -> Optional[PilotDeliveryEvent]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM orchid_pilot_delivery_events WHERE event_id = ?", (str(event_id),)
            ).fetchone()
        return None if row is None else _delivery_event_from_payload(row[0])

    def create_event(self, event: PilotDeliveryEvent) -> tuple[PilotDeliveryEvent, bool]:
        payload = _canonical_json(_delivery_event_payload(event))
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT payload FROM orchid_pilot_delivery_events WHERE event_id = ?", (event.event_id,)
                ).fetchone()
                if row is None:
                    if event.decision_id is not None:
                        transition = self._connection.execute(
                            """
                            SELECT payload FROM orchid_pilot_delivery_events
                            WHERE experiment_id = ? AND decision_id = ? AND event_type = ?
                            """,
                            (event.experiment_id, event.decision_id, event.event_type),
                        ).fetchone()
                        if transition is not None:
                            self._connection.commit()
                            existing = _delivery_event_from_payload(transition[0])
                            _require_same_delivery_event(existing, event)
                            return existing, False
                    self._connection.execute(
                        """
                        INSERT INTO orchid_pilot_delivery_events
                            (event_id, experiment_id, decision_id, event_type, payload)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (event.event_id, event.experiment_id, event.decision_id, event.event_type, payload),
                    )
                    self._connection.commit()
                    return _copy_delivery_event(event), True
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        existing = _delivery_event_from_payload(row[0])
        _require_same_delivery_event(existing, event)
        return existing, False

    def events(self, experiment_id: str, decision_id: Optional[str] = None) -> list[PilotDeliveryEvent]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM orchid_pilot_delivery_events ORDER BY rowid"
            ).fetchall()
        events = [_delivery_event_from_payload(row[0]) for row in rows]
        return [
            event
            for event in events
            if event.experiment_id == experiment_id
            and (decision_id is None or event.decision_id == decision_id)
        ]

    def _ensure_transition_index(self) -> None:
        """Migrate old lifecycle stores and make one transition type atomic."""
        columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(orchid_pilot_delivery_events)").fetchall()
        }
        for column in ("experiment_id", "decision_id", "event_type"):
            if column not in columns:
                self._connection.execute(f"ALTER TABLE orchid_pilot_delivery_events ADD COLUMN {column} TEXT")
        rows = self._connection.execute(
            """
            SELECT event_id, payload FROM orchid_pilot_delivery_events
            WHERE experiment_id IS NULL OR event_type IS NULL
            """
        ).fetchall()
        for event_id, payload in rows:
            event = _delivery_event_from_payload(payload)
            self._connection.execute(
                """
                UPDATE orchid_pilot_delivery_events
                SET experiment_id = ?, decision_id = ?, event_type = ?
                WHERE event_id = ?
                """,
                (event.experiment_id, event.decision_id, event.event_type, event_id),
            )
        duplicate = self._connection.execute(
            """
            SELECT experiment_id, decision_id, event_type
            FROM orchid_pilot_delivery_events
            WHERE decision_id IS NOT NULL
            GROUP BY experiment_id, decision_id, event_type
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise ValueError("stored lifecycle has duplicate decision transition events")
        self._connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS orchid_pilot_delivery_transition_unique
            ON orchid_pilot_delivery_events (experiment_id, decision_id, event_type)
            WHERE decision_id IS NOT NULL
            """
        )


@dataclass(frozen=True)
class PilotDecision:
    """One assigned and actually delivered decision returned to an LMS adapter."""

    decision: LoggedDecision
    arm: PilotArm
    effective_arm: PilotArm
    mode: PilotMode
    chosen_content_version: Any
    reason_code: str
    eligible_item_ids: tuple[Any, ...]



def _require_assignment_store(store: Any) -> None:
    required = (
        "get_manifest",
        "create_manifest",
        "get_assignment",
        "create_assignment",
        "get_operation",
        "transition_operation",
        "operation_events",
    )
    missing = [name for name in required if not callable(getattr(store, name, None))]
    if missing:
        raise TypeError(f"assignment_store is missing required methods: {missing}")


def _require_lifecycle_store(store: Any) -> None:
    required = ("get_event", "create_event", "events")
    missing = [name for name in required if not callable(getattr(store, name, None))]
    if missing:
        raise TypeError(f"lifecycle_store is missing required methods: {missing}")


def _require_nonempty_string(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _unique_tuple(values: Sequence[Any], field: str) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        try:
            duplicate = value in seen
        except TypeError as exc:
            raise TypeError(f"{field} IDs must be hashable") from exc
        if duplicate:
            raise ValueError(f"{field} must not contain duplicate IDs")
        seen.add(value)
        result.append(value)
    return tuple(result)


def _coerce_bool(value: Any, *, field: str, row_index: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    raise ValueError(f"{field} must be boolean at row {row_index!r}")


def _finite_float(value: Any, *, field: str, row_index: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric at row {row_index!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite at row {row_index!r}")
    return numeric


def _identifier_key(value: Any) -> str:
    return _fingerprint({"identifier": value})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _analysis_grouping_key(value: Any) -> str:
    """Serialize a DataFrame grouping value, treating its missing sentinel as null."""
    if value is None or value is pd.NA:
        return _canonical_json(None)
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return _canonical_json(None)
    return _canonical_json(value)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _namespaced_decision_id(experiment_id: str, request_id: str, course_run_id: Optional[Any] = None) -> str:
    """Derive a collision-free Orchid decision ID from the LMS request ID."""
    fingerprint = _fingerprint(
        {
            "experiment_id": experiment_id,
            "request_id": request_id,
            "course_run_id": course_run_id,
        }
    )
    return f"pilot-{fingerprint}"


def _system_event_id(*parts: Any) -> str:
    """Derive a stable internal event ID without consuming an LMS event ID."""
    return f"pilot-event-{_fingerprint(parts)}"


def _derived_model_config_identity(ranker: AdaptiveRanker) -> str:
    """Bind an implicit local model artifact to its config and learned state."""
    return _fingerprint(
        {
            "ranker_config": asdict(ranker.config),
            "deployment_version": getattr(ranker, "_deployment_version", None),
        }
    )


def _require_same_manifest(existing: ExperimentManifest, incoming: ExperimentManifest) -> None:
    if _canonical_json(asdict(existing)) != _canonical_json(asdict(incoming)):
        raise ValueError(
            "experiment_id already has a different immutable manifest; create a new experiment_id for changed "
            "catalog, model, authored policy, eligibility rule, or allocation settings"
        )


def _require_same_assignment(existing: ExperimentAssignment, incoming: ExperimentAssignment) -> None:
    if existing.arm != incoming.arm or _canonical_json(existing.stratum) != _canonical_json(incoming.stratum):
        raise ValueError(f"learner already has a different immutable assignment for {incoming.experiment_id!r}")


def _require_same_operation(existing: ExperimentOperation, incoming: ExperimentOperation) -> None:
    if _canonical_json(asdict(existing)) != _canonical_json(asdict(incoming)):
        raise ValueError("operation event ID already exists with different immutable content")


def _copy_delivery_event(event: PilotDeliveryEvent) -> PilotDeliveryEvent:
    return PilotDeliveryEvent(**json.loads(_canonical_json(_delivery_event_payload(event))))


def _delivery_event_from_payload(payload: str) -> PilotDeliveryEvent:
    return PilotDeliveryEvent(**json.loads(payload))


def _require_same_delivery_event(existing: PilotDeliveryEvent, incoming: PilotDeliveryEvent) -> None:
    if _canonical_json(_delivery_event_payload(existing)) != _canonical_json(_delivery_event_payload(incoming)):
        raise ValueError(f"pilot event ID already exists with different immutable content: {incoming.event_id}")


def _delivery_event_payload(event: PilotDeliveryEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "experiment_id": event.experiment_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "decision_id": event.decision_id,
        "item_id": _plain_json_value(event.item_id),
        "content_version": _plain_json_value(event.content_version),
        "payload": _plain_json_value(event.payload),
    }


def _plain_json_value(value: Any) -> Any:
    """Turn immutable Orchid metadata into plain JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain_json_value(item) for item in value.tolist()]
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_score_payload(*, outcome: Optional[Any], reward: Optional[float]) -> None:
    """Validate score evidence before making the lifecycle event durable."""
    if outcome is None and reward is None:
        raise ValueError("record_scored requires outcome or reward")
    if outcome is not None:
        try:
            numeric_outcome = float(outcome)
        except (TypeError, ValueError) as exc:
            raise ValueError("outcome must be exactly 0 or 1") from exc
        if not math.isfinite(numeric_outcome) or numeric_outcome not in {0.0, 1.0}:
            raise ValueError("outcome must be exactly 0 or 1")
    if reward is not None and not math.isfinite(float(reward)):
        raise ValueError("reward must be finite")


def _explanation_from_logged_decision(
    decision: LoggedDecision, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive explanation evidence from one immutable logged decision."""
    versions = _logged_content_versions(metadata, decision.decision_id)
    predicted: Sequence[Optional[float]] = (
        decision.predicted_outcomes
        if decision.predicted_outcomes is not None
        else tuple([None] * len(decision.candidate_item_ids))
    )
    candidates = [
        {
            "item_id": item_id,
            "content_version": versions[item_id],
            "score": score,
            "outcome_probability": outcome_probability,
        }
        for item_id, score, outcome_probability in zip(
            decision.candidate_item_ids,
            decision.scores,
            predicted,
        )
    ]
    return {
        "kind": "adaptive" if _effective_pilot_arm(metadata, decision.decision_id) == "treatment" else "authored",
        "policy_name": decision.policy_name,
        "policy_version": decision.policy_version,
        "reason_code": metadata["reason_code"],
        "chosen_item_id": decision.chosen_item_id,
        "chosen_content_version": versions[decision.chosen_item_id],
        "ranked_candidates": candidates,
    }


def _shadow_proposal(ranker: AdaptiveRanker, decision: LoggedDecision) -> dict[str, Any]:
    """Capture a non-delivered Orchid proposal during shadow delivery."""
    recommendations = ranker.recommend(
        decision.user_id,
        decision.candidate_item_ids,
        top_k=len(decision.candidate_item_ids),
        context_hash=decision.context_hash,
    )
    return {
        "kind": "adaptive",
        "chosen_item_id": None if not recommendations else recommendations[0].item_id,
        "ranked_candidates": [recommendation.to_dict() for recommendation in recommendations],
        "candidate_set_preserved": {
            recommendation.item_id for recommendation in recommendations
        }
        == set(decision.candidate_item_ids),
    }


def _stable_unit_interval(*values: Any) -> float:
    digest = hashlib.sha256(_canonical_json(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _pilot_request_fingerprint(
    request: PilotRequest,
    *,
    timestamp: float,
    catalog_version: str,
    eligibility_rule_version: str,
    eligible_item_ids: Sequence[Any],
) -> str:
    return _fingerprint(
        {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "course_id": request.course_id,
            "module_id": request.module_id,
            "course_run_id": request.course_run_id,
            "timestamp": timestamp,
            "completed_item_ids": list(request.completed_item_ids),
            "candidate_item_ids": None if request.candidate_item_ids is None else list(request.candidate_item_ids),
            "stratum": request.stratum,
            "catalog_version": catalog_version,
            "eligibility_rule_version": eligibility_rule_version,
            "eligible_item_ids": list(eligible_item_ids),
        }
    )


def _pilot_metadata(decision: LoggedDecision) -> Mapping[str, Any]:
    metadata = _pilot_metadata_or_none(decision)
    if metadata is None:
        raise ValueError(f"decision_id is not an AdaptivePracticePilot record: {decision.decision_id}")
    return metadata


def _pilot_metadata_or_none(decision: LoggedDecision) -> Optional[Mapping[str, Any]]:
    return _pilot_metadata_or_none_from_value(decision.policy_metadata)


def _pilot_metadata_or_none_from_value(value: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    decision_metadata = value.get("decision_metadata")
    if not isinstance(decision_metadata, Mapping):
        return None
    pilot = decision_metadata.get("pilot")
    if not isinstance(pilot, Mapping):
        return None
    required = {
        "experiment_id",
        "experiment_manifest_digest",
        "experiment_arm",
        "effective_arm",
        "delivery_mode",
        "catalog_version",
        "catalog_content_digest",
        "course_run_id",
        "source_request_id",
        "model_artifact_id",
        "model_config_identity",
        "authored_policy_version",
        "eligibility_rule_version",
        "allocation_method",
        "allocation_treatment_fraction",
        "allocation_salt_digest",
        "stratum",
        "reason_code",
        "candidate_content_versions",
        "request_fingerprint",
    }
    if not required.issubset(pilot):
        return None
    return pilot


def _pilot_arm(metadata: Mapping[str, Any], decision_id: str) -> PilotArm:
    arm = metadata.get("experiment_arm")
    if arm not in {"control", "treatment"}:
        raise ValueError(f"logged pilot decision has invalid experiment arm: {decision_id}")
    return cast(PilotArm, arm)


def _effective_pilot_arm(metadata: Mapping[str, Any], decision_id: str) -> PilotArm:
    arm = metadata.get("effective_arm")
    if arm not in {"control", "treatment"}:
        raise ValueError(f"logged pilot decision has invalid effective arm: {decision_id}")
    return cast(PilotArm, arm)


def _pilot_mode(metadata: Mapping[str, Any], decision_id: str) -> PilotMode:
    mode = metadata.get("delivery_mode")
    if mode not in {"aa", "shadow", "active", "halted"}:
        raise ValueError(f"logged pilot decision has invalid delivery mode: {decision_id}")
    return cast(PilotMode, mode)


def _delivery_reason(mode: PilotMode, authored_reason: str) -> str:
    if mode == "aa":
        return "AA_AUTHORED_CONTROL"
    if mode == "shadow":
        return "SHADOW_AUTHORED_CONTROL"
    if mode == "halted":
        return "KILL_SWITCH"
    return authored_reason


def _require_matching_pilot_request(metadata: Mapping[str, Any], fingerprint: str, request_id: str) -> None:
    if metadata.get("request_fingerprint") != fingerprint:
        raise ValueError(f"request_id already exists for a different pilot request: {request_id}")


def _require_matching_pilot_configuration(
    metadata: Mapping[str, Any], manifest: ExperimentManifest, decision: LoggedDecision
) -> None:
    """Reject logs whose immutable pilot context differs from this adapter."""
    expected = {
        "experiment_id": manifest.experiment_id,
        "experiment_manifest_digest": manifest.digest,
        "catalog_version": manifest.catalog_version,
        "catalog_content_digest": manifest.catalog_content_digest,
        "model_artifact_id": manifest.model_artifact_id,
        "model_config_identity": manifest.model_config_identity,
        "authored_policy_version": manifest.authored_policy_version,
        "eligibility_rule_version": manifest.eligibility_rule_version,
        "allocation_method": manifest.allocation_method,
        "allocation_treatment_fraction": manifest.treatment_fraction,
        "allocation_salt_digest": manifest.randomization_salt_digest,
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if _canonical_json(metadata.get(field)) != _canonical_json(expected_value)
    ]
    if mismatches:
        raise ValueError(
            "logged pilot decision does not match this immutable experiment configuration "
            f"({', '.join(mismatches)}): {decision.decision_id}"
        )
    source_request_id = metadata.get("source_request_id")
    if not isinstance(source_request_id, str) or not source_request_id:
        raise ValueError(f"logged pilot decision has invalid source request ID: {decision.decision_id}")
    if decision.decision_id != _namespaced_decision_id(
        manifest.experiment_id,
        source_request_id,
        metadata.get("course_run_id"),
    ):
        raise ValueError(f"logged pilot decision has an invalid experiment namespace: {decision.decision_id}")
    assigned_arm = _pilot_arm(metadata, decision.decision_id)
    effective_arm = _effective_pilot_arm(metadata, decision.decision_id)
    mode = _pilot_mode(metadata, decision.decision_id)
    if effective_arm == "treatment" and (assigned_arm != "treatment" or mode != "active"):
        raise ValueError(f"logged pilot decision has an invalid treatment delivery state: {decision.decision_id}")
    expected_policy_version = (
        manifest.authored_policy_version if effective_arm == "control" else manifest.model_artifact_id
    )
    if decision.policy_version != expected_policy_version:
        raise ValueError(f"logged pilot decision has a mismatched policy version: {decision.decision_id}")
    if effective_arm == "control" and decision.policy_name != "authored-static":
        raise ValueError(f"logged control decision is not an authored-static decision: {decision.decision_id}")
    if effective_arm == "treatment" and (decision.exploration_rate != 0.0 or decision.was_exploration):
        raise ValueError(f"logged treatment decision is not frozen: {decision.decision_id}")


def _require_exact_eligible_candidates(
    decision: LoggedDecision, eligibility: PilotEligibility, decision_id: str
) -> None:
    """Ensure Orchid logged precisely the LMS-approved candidate set and revisions."""
    actual_candidates = tuple(decision.candidate_item_ids)
    expected_candidates = eligibility.item_ids
    try:
        exact_items = (
            len(actual_candidates) == len(expected_candidates)
            and len(set(actual_candidates)) == len(actual_candidates)
            and set(actual_candidates) == set(expected_candidates)
        )
    except TypeError as exc:
        raise ValueError(f"logged pilot decision has unhashable candidate IDs: {decision_id}") from exc
    if not exact_items:
        raise ValueError(f"ranker logged a candidate set outside the exact approved eligibility: {decision_id}")
    if decision.chosen_item_id not in set(expected_candidates):
        raise ValueError(f"ranker chose an item outside the exact approved eligibility: {decision_id}")
    logged_versions = _logged_content_versions(_pilot_metadata(decision), decision_id)
    expected_versions = dict(zip(eligibility.item_ids, eligibility.content_versions))
    if set(logged_versions) != set(expected_versions) or any(
        _canonical_json(logged_versions[item_id]) != _canonical_json(content_version)
        for item_id, content_version in expected_versions.items()
    ):
        raise ValueError(f"logged pilot decision has content versions outside the frozen eligibility: {decision_id}")


def _logged_content_versions(metadata: Mapping[str, Any], decision_id: str) -> dict[Any, Any]:
    values = metadata.get("candidate_content_versions")
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"logged pilot decision has invalid content-version metadata: {decision_id}")
    mapping: dict[Any, Any] = {}
    for pair in values:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"logged pilot decision has invalid content-version metadata: {decision_id}")
        item_id, content_version = pair
        try:
            if item_id in mapping:
                raise ValueError(f"logged pilot decision has duplicate content-version metadata: {decision_id}")
            mapping[item_id] = content_version
        except TypeError as exc:
            raise ValueError(f"logged pilot decision has unhashable content-version metadata: {decision_id}") from exc
    return mapping
