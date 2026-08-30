"""Validation contracts for a versioned adaptive-learning exercise catalog.

The catalog is deliberately validated independently from :class:`AdaptiveRanker`.
It lets a curriculum system check and repair authoring data before that data is
used to train or serve a learning policy.  The validator returns diagnostics
rather than failing fast so an import job can show an author every issue in a
single pass.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

__all__ = [
    "CanonicalExercise",
    "CatalogDiagnostic",
    "LearningCatalogSchema",
    "LearningCatalogValidation",
    "validate_learning_catalog",
]


DiagnosticSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class LearningCatalogSchema:
    """Column names for an exercise catalog.

    ``item_id_col`` is always required.  The remaining fields describe the
    authoring metadata Orchid needs for curriculum-aware learning.  A content
    version column is optional: when it is present, an exercise's canonical
    identity is ``(item_id, content_version)``; otherwise it is ``item_id``.
    Set a column field to ``None`` when the upstream system does not maintain
    that field.
    """

    item_id_col: str = "item_id"
    content_version_col: Optional[str] = "content_version"
    course_id_col: Optional[str] = "course_id"
    module_id_col: Optional[str] = "module_id"
    skill_id_col: Optional[str] = "skill_id"
    category_id_col: Optional[str] = "category_id"
    difficulty_col: Optional[str] = "difficulty"
    assessment_only_col: Optional[str] = "assessment_only"
    prerequisites_col: Optional[str] = "prerequisites"


@dataclass(frozen=True)
class CatalogDiagnostic:
    """One actionable catalog validation finding.

    ``rows`` contains source DataFrame index values, not positional offsets.
    A diagnostic can therefore be attached directly to an import report even
    when the authoring source uses its own row identifiers.
    """

    code: str
    severity: DiagnosticSeverity
    message: str
    rows: tuple[Any, ...] = ()
    item_id: Any = None
    content_version: Any = None
    related_item_ids: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CanonicalExercise:
    """One normalized exercise record retained from a catalog import."""

    item_id: Any
    content_version: Any = None
    course_id: Any = None
    module_id: Any = None
    skill_id: Any = None
    category_id: Any = None
    difficulty: Optional[float] = None
    assessment_only: Optional[bool] = None
    prerequisite_item_ids: tuple[Any, ...] = ()
    source_rows: tuple[Any, ...] = ()

    @property
    def identity(self) -> tuple[Any, ...]:
        """Return the stable identity used for duplicate detection."""
        if self.content_version is None:
            return (self.item_id,)
        return (self.item_id, self.content_version)


@dataclass(frozen=True)
class LearningCatalogValidation:
    """Normalized catalog rows plus every validation finding."""

    schema: LearningCatalogSchema
    exercises: tuple[CanonicalExercise, ...]
    diagnostics: tuple[CatalogDiagnostic, ...]

    @property
    def is_valid(self) -> bool:
        """Whether validation found no errors."""
        return not self.errors

    @property
    def errors(self) -> tuple[CatalogDiagnostic, ...]:
        """Return blocking findings in source order."""
        return tuple(issue for issue in self.diagnostics if issue.severity == "error")

    @property
    def warnings(self) -> tuple[CatalogDiagnostic, ...]:
        """Return non-blocking data-quality findings in source order."""
        return tuple(issue for issue in self.diagnostics if issue.severity == "warning")

    @property
    def exercise_by_identity(self) -> dict[tuple[Any, ...], CanonicalExercise]:
        """Return canonical records keyed by item and optional content version."""
        return {exercise.identity: exercise for exercise in self.exercises}

    def raise_for_errors(self) -> None:
        """Raise a concise error after callers have had a chance to inspect findings."""
        if not self.errors:
            return
        preview = "; ".join(f"{issue.code}: {issue.message}" for issue in self.errors[:3])
        remaining = len(self.errors) - 3
        if remaining > 0:
            preview = f"{preview}; and {remaining} more error(s)"
        raise ValueError(f"learning catalog validation failed: {preview}")


def validate_learning_catalog(
    catalog: pd.DataFrame,
    *,
    schema: LearningCatalogSchema = LearningCatalogSchema(),
    require_complete_metadata: bool = False,
    allow_external_prerequisites: bool = False,
) -> LearningCatalogValidation:
    """Validate and normalize an author-maintained, versioned exercise catalog.

    Args:
        catalog: One row per exercise version. Exact duplicate rows are
            condensed with a warning; conflicting duplicates are errors.
        schema: The source column contract.
        require_complete_metadata: Escalate absent or missing curriculum,
            skill/category, difficulty, assessment, and prerequisite metadata
            from warnings to errors. This is useful as a production import
            gate, while a first pilot can still inspect partial catalog data.
        allow_external_prerequisites: Permit prerequisite IDs that are not in
            this catalog. Use this only when a separate catalog owns those
            exercises; otherwise dangling prerequisites are errors.

    The function never mutates ``catalog`` and does not raise for ordinary data
    quality issues.  Inspect ``is_valid`` or call ``raise_for_errors`` after
    displaying the full diagnostic list to the content author.
    """
    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("catalog must be a pandas DataFrame")

    diagnostics: list[CatalogDiagnostic] = []
    if catalog.empty:
        diagnostics.append(
            CatalogDiagnostic("empty_catalog", "error", "catalog must contain at least one exercise")
        )
        return LearningCatalogValidation(schema, (), tuple(diagnostics))

    if schema.item_id_col not in catalog.columns:
        diagnostics.append(
            CatalogDiagnostic(
                "missing_item_id_column",
                "error",
                f"catalog must include item identifier column {schema.item_id_col!r}",
            )
        )
        return LearningCatalogValidation(schema, (), tuple(diagnostics))

    available_columns = set(catalog.columns)
    optional_columns = _optional_columns(schema)
    for label, column in optional_columns.items():
        if column is not None and column not in available_columns:
            _append_missing_metadata_column(
                diagnostics,
                label=label,
                column=column,
                require_complete_metadata=require_complete_metadata,
            )

    has_skill_column = schema.skill_id_col is not None and schema.skill_id_col in available_columns
    has_category_column = schema.category_id_col is not None and schema.category_id_col in available_columns
    if not has_skill_column and not has_category_column:
        diagnostics.append(
            CatalogDiagnostic(
                "missing_skill_or_category_column",
                _metadata_severity(require_complete_metadata),
                "catalog should include at least one of skill or category metadata columns "
                f"({schema.skill_id_col!r}, {schema.category_id_col!r})",
            )
        )

    version_is_supplied = (
        schema.content_version_col is not None and schema.content_version_col in available_columns
    )
    raw_exercises: list[CanonicalExercise] = []
    for row_index, row in catalog.iterrows():
        item_id = _validated_identifier(
            row[schema.item_id_col],
            field="item_id",
            row_index=row_index,
            diagnostics=diagnostics,
        )
        if item_id is None:
            continue

        content_version = None
        if version_is_supplied:
            assert schema.content_version_col is not None
            content_version = _validated_identifier(
                row[schema.content_version_col],
                field="content_version",
                row_index=row_index,
                diagnostics=diagnostics,
            )
            if content_version is None:
                continue

        course_id = _metadata_scalar(
            row,
            schema.course_id_col,
            label="course_id",
            item_id=item_id,
            content_version=content_version,
            row_index=row_index,
            diagnostics=diagnostics,
            require_complete_metadata=require_complete_metadata,
        )
        module_id = _metadata_scalar(
            row,
            schema.module_id_col,
            label="module_id",
            item_id=item_id,
            content_version=content_version,
            row_index=row_index,
            diagnostics=diagnostics,
            require_complete_metadata=require_complete_metadata,
        )
        skill_id = _metadata_scalar(
            row,
            schema.skill_id_col,
            label="skill_id",
            item_id=item_id,
            content_version=content_version,
            row_index=row_index,
            diagnostics=diagnostics,
            require_complete_metadata=False,
        )
        category_id = _metadata_scalar(
            row,
            schema.category_id_col,
            label="category_id",
            item_id=item_id,
            content_version=content_version,
            row_index=row_index,
            diagnostics=diagnostics,
            require_complete_metadata=False,
        )
        if skill_id is None and category_id is None:
            diagnostics.append(
                CatalogDiagnostic(
                    "missing_skill_or_category",
                    _metadata_severity(require_complete_metadata),
                    "exercise needs a skill_id or category_id for adaptive learning",
                    rows=(row_index,),
                    item_id=item_id,
                    content_version=content_version,
                )
            )

        difficulty = _difficulty_value(
            row,
            schema.difficulty_col,
            item_id=item_id,
            content_version=content_version,
            row_index=row_index,
            diagnostics=diagnostics,
            require_complete_metadata=require_complete_metadata,
        )
        assessment_only = _assessment_only_value(
            row,
            schema.assessment_only_col,
            item_id=item_id,
            content_version=content_version,
            row_index=row_index,
            diagnostics=diagnostics,
            require_complete_metadata=require_complete_metadata,
        )
        prerequisites = _prerequisites_value(
            row,
            schema.prerequisites_col,
            item_id=item_id,
            content_version=content_version,
            row_index=row_index,
            diagnostics=diagnostics,
            require_complete_metadata=require_complete_metadata,
        )
        raw_exercises.append(
            CanonicalExercise(
                item_id=item_id,
                content_version=content_version,
                course_id=course_id,
                module_id=module_id,
                skill_id=skill_id,
                category_id=category_id,
                difficulty=difficulty,
                assessment_only=assessment_only,
                prerequisite_item_ids=prerequisites,
                source_rows=(row_index,),
            )
        )

    exercises = _deduplicate_exercises(raw_exercises, diagnostics)
    _validate_prerequisites(
        exercises,
        diagnostics,
        allow_external_prerequisites=allow_external_prerequisites,
    )
    return LearningCatalogValidation(schema, tuple(exercises), tuple(diagnostics))


def _optional_columns(schema: LearningCatalogSchema) -> dict[str, Optional[str]]:
    return {
        "content_version": schema.content_version_col,
        "course_id": schema.course_id_col,
        "module_id": schema.module_id_col,
        "difficulty": schema.difficulty_col,
        "assessment_only": schema.assessment_only_col,
        "prerequisites": schema.prerequisites_col,
    }


def _append_missing_metadata_column(
    diagnostics: list[CatalogDiagnostic],
    *,
    label: str,
    column: str,
    require_complete_metadata: bool,
) -> None:
    if label == "content_version":
        # Versioning is opt-in: a catalog without this column has item-level
        # identities and is still valid.
        return
    diagnostics.append(
        CatalogDiagnostic(
            "missing_metadata_column",
            _metadata_severity(require_complete_metadata),
            f"catalog is missing optional {label} metadata column {column!r}",
        )
    )


def _metadata_severity(require_complete_metadata: bool) -> DiagnosticSeverity:
    return "error" if require_complete_metadata else "warning"


def _validated_identifier(
    value: Any,
    *,
    field: str,
    row_index: Any,
    diagnostics: list[CatalogDiagnostic],
) -> Any:
    value = _normalise_missing(value)
    if value is None:
        diagnostics.append(
            CatalogDiagnostic(
                f"missing_{field}",
                "error",
                f"{field} must not be missing",
                rows=(row_index,),
            )
        )
        return None
    if isinstance(value, str) and not value.strip():
        diagnostics.append(
            CatalogDiagnostic(
                f"missing_{field}",
                "error",
                f"{field} must not be blank",
                rows=(row_index,),
            )
        )
        return None
    try:
        hash(value)
    except TypeError:
        diagnostics.append(
            CatalogDiagnostic(
                f"unhashable_{field}",
                "error",
                f"{field} must be a scalar, hashable identifier",
                rows=(row_index,),
            )
        )
        return None
    return value


def _metadata_scalar(
    row: pd.Series,
    column: Optional[str],
    *,
    label: str,
    item_id: Any,
    content_version: Any,
    row_index: Any,
    diagnostics: list[CatalogDiagnostic],
    require_complete_metadata: bool,
) -> Any:
    if column is None or column not in row.index:
        return None
    value = _normalise_missing(row[column])
    if value is None or (isinstance(value, str) and not value.strip()):
        diagnostics.append(
            CatalogDiagnostic(
                f"missing_{label}",
                _metadata_severity(require_complete_metadata),
                f"{label} is missing",
                rows=(row_index,),
                item_id=item_id,
                content_version=content_version,
            )
        )
        return None
    if isinstance(value, (Mapping, list, set, tuple)):
        diagnostics.append(
            CatalogDiagnostic(
                f"invalid_{label}",
                "error",
                f"{label} must be a scalar identifier",
                rows=(row_index,),
                item_id=item_id,
                content_version=content_version,
            )
        )
        return None
    return value


def _difficulty_value(
    row: pd.Series,
    column: Optional[str],
    *,
    item_id: Any,
    content_version: Any,
    row_index: Any,
    diagnostics: list[CatalogDiagnostic],
    require_complete_metadata: bool,
) -> Optional[float]:
    if column is None or column not in row.index:
        return None
    raw_value = _normalise_missing(row[column])
    if raw_value is None:
        diagnostics.append(
            CatalogDiagnostic(
                "missing_difficulty",
                _metadata_severity(require_complete_metadata),
                "difficulty is missing",
                rows=(row_index,),
                item_id=item_id,
                content_version=content_version,
            )
        )
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        diagnostics.append(
            CatalogDiagnostic(
                "invalid_difficulty",
                "error",
                "difficulty must be a finite numeric value",
                rows=(row_index,),
                item_id=item_id,
                content_version=content_version,
            )
        )
        return None
    return value


def _assessment_only_value(
    row: pd.Series,
    column: Optional[str],
    *,
    item_id: Any,
    content_version: Any,
    row_index: Any,
    diagnostics: list[CatalogDiagnostic],
    require_complete_metadata: bool,
) -> Optional[bool]:
    if column is None or column not in row.index:
        return None
    raw_value = _normalise_missing(row[column])
    if raw_value is None:
        diagnostics.append(
            CatalogDiagnostic(
                "missing_assessment_only",
                _metadata_severity(require_complete_metadata),
                "assessment_only is missing",
                rows=(row_index,),
                item_id=item_id,
                content_version=content_version,
            )
        )
        return None
    normalised = _coerce_bool(raw_value)
    if normalised is None:
        diagnostics.append(
            CatalogDiagnostic(
                "invalid_assessment_only",
                "error",
                "assessment_only must be boolean (or a 0/1, true/false value)",
                rows=(row_index,),
                item_id=item_id,
                content_version=content_version,
            )
        )
    return normalised


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _prerequisites_value(
    row: pd.Series,
    column: Optional[str],
    *,
    item_id: Any,
    content_version: Any,
    row_index: Any,
    diagnostics: list[CatalogDiagnostic],
    require_complete_metadata: bool,
) -> tuple[Any, ...]:
    if column is None or column not in row.index:
        return ()
    raw_value = _normalise_missing(row[column])
    if raw_value is None:
        diagnostics.append(
            CatalogDiagnostic(
                "missing_prerequisites",
                _metadata_severity(require_complete_metadata),
                "prerequisites is missing; use an empty list when there are none",
                rows=(row_index,),
                item_id=item_id,
                content_version=content_version,
            )
        )
        return ()
    values = _as_prerequisite_values(raw_value)
    if values is None:
        diagnostics.append(
            CatalogDiagnostic(
                "invalid_prerequisites",
                "error",
                "prerequisites must be an identifier or an iterable of identifiers",
                rows=(row_index,),
                item_id=item_id,
                content_version=content_version,
            )
        )
        return ()
    prerequisites: list[Any] = []
    seen: set[Any] = set()
    for prerequisite in values:
        identifier = _validated_identifier(
            prerequisite,
            field="prerequisite_item_id",
            row_index=row_index,
            diagnostics=diagnostics,
        )
        if identifier is None:
            continue
        if identifier in seen:
            diagnostics.append(
                CatalogDiagnostic(
                    "duplicate_prerequisite",
                    "warning",
                    "duplicate prerequisite was condensed",
                    rows=(row_index,),
                    item_id=item_id,
                    content_version=content_version,
                    related_item_ids=(identifier,),
                )
            )
            continue
        seen.add(identifier)
        prerequisites.append(identifier)
    return tuple(prerequisites)


def _as_prerequisite_values(value: Any) -> Optional[tuple[Any, ...]]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            if not isinstance(decoded, list):
                return None
            return tuple(decoded)
        return (value,)
    if isinstance(value, Mapping):
        return None
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _deduplicate_exercises(
    exercises: list[CanonicalExercise], diagnostics: list[CatalogDiagnostic]
) -> list[CanonicalExercise]:
    canonical: dict[tuple[Any, ...], CanonicalExercise] = {}
    ordered: list[CanonicalExercise] = []
    for exercise in exercises:
        existing = canonical.get(exercise.identity)
        if existing is None:
            canonical[exercise.identity] = exercise
            ordered.append(exercise)
            continue
        if _exercise_payload(existing) == _exercise_payload(exercise):
            replacement = CanonicalExercise(
                **{
                    **existing.__dict__,
                    "source_rows": (*existing.source_rows, *exercise.source_rows),
                }
            )
            canonical[exercise.identity] = replacement
            ordered[ordered.index(existing)] = replacement
            diagnostics.append(
                CatalogDiagnostic(
                    "duplicate_exercise_version",
                    "warning",
                    "identical exercise-version row was condensed",
                    rows=replacement.source_rows,
                    item_id=exercise.item_id,
                    content_version=exercise.content_version,
                )
            )
            continue
        diagnostics.append(
            CatalogDiagnostic(
                "conflicting_exercise_metadata",
                "error",
                "the same item_id/content_version has conflicting canonical metadata",
                rows=(*existing.source_rows, *exercise.source_rows),
                item_id=exercise.item_id,
                content_version=exercise.content_version,
            )
        )
    return ordered


def _exercise_payload(exercise: CanonicalExercise) -> tuple[Any, ...]:
    return (
        exercise.item_id,
        exercise.content_version,
        exercise.course_id,
        exercise.module_id,
        exercise.skill_id,
        exercise.category_id,
        exercise.difficulty,
        exercise.assessment_only,
        exercise.prerequisite_item_ids,
    )


def _validate_prerequisites(
    exercises: list[CanonicalExercise],
    diagnostics: list[CatalogDiagnostic],
    *,
    allow_external_prerequisites: bool,
) -> None:
    known_item_ids = {exercise.item_id for exercise in exercises}
    graph: dict[Any, list[Any]] = {}
    for exercise in exercises:
        graph.setdefault(exercise.item_id, [])
        for prerequisite in exercise.prerequisite_item_ids:
            if prerequisite not in known_item_ids:
                diagnostics.append(
                    CatalogDiagnostic(
                        "unknown_prerequisite",
                        "warning" if allow_external_prerequisites else "error",
                        "prerequisite does not identify an exercise in this catalog",
                        rows=exercise.source_rows,
                        item_id=exercise.item_id,
                        content_version=exercise.content_version,
                        related_item_ids=(prerequisite,),
                    )
                )
                continue
            if prerequisite not in graph[exercise.item_id]:
                graph[exercise.item_id].append(prerequisite)
    _append_cycle_diagnostics(graph, exercises, diagnostics)


def _append_cycle_diagnostics(
    graph: dict[Any, list[Any]],
    exercises: list[CanonicalExercise],
    diagnostics: list[CatalogDiagnostic],
) -> None:
    state: dict[Any, int] = {}
    path: list[Any] = []
    cycles: set[frozenset[Any]] = set()
    rows_by_item: dict[Any, list[Any]] = {}
    for exercise in exercises:
        rows_by_item.setdefault(exercise.item_id, []).extend(exercise.source_rows)

    def visit(item_id: Any) -> None:
        state[item_id] = 1
        path.append(item_id)
        for prerequisite in graph[item_id]:
            if state.get(prerequisite, 0) == 0:
                visit(prerequisite)
            elif state.get(prerequisite) == 1:
                cycle = path[path.index(prerequisite) :]
                cycle_key = frozenset(cycle)
                if cycle_key in cycles:
                    continue
                cycles.add(cycle_key)
                cycle_display = (*cycle, prerequisite)
                cycle_rows = tuple(row for member in cycle for row in rows_by_item[member])
                diagnostics.append(
                    CatalogDiagnostic(
                        "prerequisite_cycle",
                        "error",
                        "prerequisites contain a cycle: " + " -> ".join(map(str, cycle_display)),
                        rows=cycle_rows,
                        item_id=prerequisite,
                        related_item_ids=tuple(cycle_display),
                    )
                )
        path.pop()
        state[item_id] = 2

    for item_id in graph:
        if state.get(item_id, 0) == 0:
            visit(item_id)


def _normalise_missing(value: Any) -> Any:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value
