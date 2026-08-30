"""Focused tests for versioned adaptive-learning catalog validation."""
from __future__ import annotations

import pandas as pd

from orchid_ranker.learning_catalog import LearningCatalogSchema, validate_learning_catalog


def _complete_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["add-1", "frac-1", "alg-1"],
            "content_version": ["2026.1", "2026.1", "2026.1"],
            "course_id": ["math", "math", "math"],
            "module_id": ["arithmetic", "fractions", "algebra"],
            "skill_id": ["addition", "fractions", "algebra"],
            "difficulty": [0.1, 0.5, 0.8],
            "assessment_only": [False, False, True],
            "prerequisites": [[], ["add-1"], ["frac-1"]],
        },
        index=[10, 11, 12],
    )


def test_versioned_catalog_normalizes_canonical_exercises() -> None:
    report = validate_learning_catalog(_complete_catalog(), require_complete_metadata=True)

    assert report.is_valid
    assert report.diagnostics == ()
    algebra = report.exercise_by_identity[("alg-1", "2026.1")]
    assert algebra.assessment_only is True
    assert algebra.prerequisite_item_ids == ("frac-1",)
    assert algebra.source_rows == (12,)


def test_conflicting_duplicate_version_is_a_blocking_diagnostic() -> None:
    catalog = pd.concat([_complete_catalog(), _complete_catalog().iloc[[0]].assign(difficulty=0.9)])

    report = validate_learning_catalog(catalog, require_complete_metadata=True)

    assert not report.is_valid
    conflict = next(issue for issue in report.errors if issue.code == "conflicting_exercise_metadata")
    assert conflict.item_id == "add-1"
    assert conflict.content_version == "2026.1"
    assert conflict.rows == (10, 10)


def test_missing_metadata_is_reported_and_can_be_an_import_gate() -> None:
    catalog = _complete_catalog().drop(columns=["module_id", "assessment_only"])
    catalog.at[10, "difficulty"] = None
    catalog.at[11, "skill_id"] = None

    exploratory = validate_learning_catalog(catalog)
    strict = validate_learning_catalog(catalog, require_complete_metadata=True)

    assert exploratory.is_valid
    assert {issue.code for issue in exploratory.warnings} >= {
        "missing_metadata_column",
        "missing_difficulty",
        "missing_skill_or_category",
    }
    assert not strict.is_valid
    assert {issue.code for issue in strict.errors} >= {
        "missing_metadata_column",
        "missing_difficulty",
        "missing_skill_or_category",
    }


def test_prerequisite_cycles_and_dangling_references_are_diagnosed() -> None:
    catalog = _complete_catalog()
    catalog.at[10, "prerequisites"] = ["alg-1"]
    catalog.at[11, "prerequisites"] = ["add-1", "unknown-item"]

    report = validate_learning_catalog(catalog, require_complete_metadata=True)

    cycle = next(issue for issue in report.errors if issue.code == "prerequisite_cycle")
    dangling = next(issue for issue in report.errors if issue.code == "unknown_prerequisite")
    assert cycle.related_item_ids == ("add-1", "alg-1", "frac-1", "add-1")
    assert dangling.related_item_ids == ("unknown-item",)
    assert dangling.rows == (11,)


def test_external_prerequisites_can_be_explicitly_allowed() -> None:
    catalog = _complete_catalog()
    catalog.at[10, "prerequisites"] = ["outside-catalog"]

    report = validate_learning_catalog(
        catalog,
        require_complete_metadata=True,
        allow_external_prerequisites=True,
    )

    assert report.is_valid
    issue = next(issue for issue in report.warnings if issue.code == "unknown_prerequisite")
    assert issue.related_item_ids == ("outside-catalog",)


def test_schema_supports_an_unversioned_category_catalog() -> None:
    catalog = pd.DataFrame(
        {
            "exercise": [1, 2],
            "category": ["basics", "advanced"],
            "difficulty": [1, 2],
            "is_assessment": [0, 1],
            "needs": [[], [1]],
        }
    )
    schema = LearningCatalogSchema(
        item_id_col="exercise",
        content_version_col=None,
        course_id_col=None,
        module_id_col=None,
        skill_id_col=None,
        category_id_col="category",
        difficulty_col="difficulty",
        assessment_only_col="is_assessment",
        prerequisites_col="needs",
    )

    report = validate_learning_catalog(catalog, schema=schema, require_complete_metadata=True)

    assert report.is_valid
    assert tuple(report.exercise_by_identity) == ((1,), (2,))
