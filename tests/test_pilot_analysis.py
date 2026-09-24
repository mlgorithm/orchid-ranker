"""Checks for randomized-unit analysis of the one-course pilot."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from orchid_ranker.pilot_analysis import analyze_pilot_retention


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    enrollment = pd.DataFrame(
        [
            {"user_id": "c1", "course_run_id": "routing", "assigned_arm": "control", "stratum": "low", "assessment_due_timestamp": 1000},
            {"user_id": "c2", "course_run_id": "routing", "assigned_arm": "control", "stratum": "high", "assessment_due_timestamp": 1000},
            {"user_id": "t1", "course_run_id": "routing", "assigned_arm": "treatment", "stratum": "low", "assessment_due_timestamp": 1000},
            {"user_id": "t2", "course_run_id": "routing", "assigned_arm": "treatment", "stratum": "high", "assessment_due_timestamp": 1000},
        ]
    )
    assessments = pd.DataFrame(
        [
            {"assessment_event_id": "a1", "user_id": "c1", "course_run_id": "routing", "assessment_form_version": "v1", "timestamp": 1000, "score": 0.4, "independent": True},
            {"assessment_event_id": "a2", "user_id": "t1", "course_run_id": "routing", "assessment_form_version": "v1", "timestamp": 1000, "score": 0.8, "independent": True},
            {"assessment_event_id": "a3", "user_id": "t2", "course_run_id": "routing", "assessment_form_version": "v1", "timestamp": 1000 + 4 * 86_400, "score": 0.9, "independent": True},
        ]
    )
    return enrollment, assessments


def test_retention_analysis_counts_every_randomized_learner_once() -> None:
    enrollment, assessments = _inputs()
    report = analyze_pilot_retention(enrollment, assessments, bootstrap_samples=100)

    assert report["arms"]["control"]["randomized"] == 2
    assert report["arms"]["treatment"]["randomized"] == 2
    assert report["arms"]["control"]["assessed_in_window"] == 1
    assert report["arms"]["treatment"]["out_of_window"] == 1
    assert report["treatment_minus_control"] == pytest.approx(0.2)
    assert report["missing_score_bounds"] == pytest.approx([-0.3, 0.7])
    assert report["assessment_rate_difference"] == 0
    assert report["strata_without_both_arms"] == []


def test_retention_analysis_waits_for_every_assessment_window_to_close() -> None:
    enrollment, assessments = _inputs()
    enrollment.loc[3, "assessment_due_timestamp"] = 2000
    assessments = assessments.loc[assessments["user_id"] != "t2"]

    with pytest.raises(ValueError, match="window has not closed"):
        analyze_pilot_retention(
            enrollment, assessments, analysis_timestamp=2000 + 3 * 86_400 - 1,
        )

    report = analyze_pilot_retention(
        enrollment, assessments, analysis_timestamp=2000 + 3 * 86_400,
        bootstrap_samples=100,
    )
    assert report["assessment_window_end_timestamp"] == 2000 + 3 * 86_400
    assert report["analysis_timestamp"] == report["assessment_window_end_timestamp"]


def test_retention_analysis_rejects_assessments_after_analysis_cutoff() -> None:
    enrollment, assessments = _inputs()
    assessments.loc[0, "timestamp"] = 1000 + 4 * 86_400

    with pytest.raises(ValueError, match="after analysis_timestamp"):
        analyze_pilot_retention(
            enrollment, assessments, analysis_timestamp=1000 + 3 * 86_400,
        )


def test_retention_analysis_rejects_future_analysis_cutoff() -> None:
    enrollment, assessments = _inputs()
    with pytest.raises(ValueError, match="cannot be in the future"):
        analyze_pilot_retention(enrollment, assessments, analysis_timestamp=10**12)


def test_retention_analysis_reports_no_bootstrap_interval_with_one_learner_per_arm() -> None:
    enrollment, assessments = _inputs()
    enrollment = enrollment.loc[enrollment["user_id"].isin(["c1", "t1"])]
    assessments = assessments.loc[assessments["user_id"].isin(["c1", "t1"])]

    report = analyze_pilot_retention(enrollment, assessments)

    assert report["treatment_minus_control"] == pytest.approx(0.4)
    assert report["bootstrap_95_percent_interval"] is None
    assert report["strata_without_both_arms"] == []


def test_retention_analysis_accepts_unstratified_assignment_and_flags_no_overlap() -> None:
    enrollment, assessments = _inputs()
    enrollment["stratum"] = None

    report = analyze_pilot_retention(enrollment, assessments, bootstrap_samples=100)
    assert report["arms"]["control"]["strata"] == {"(unstratified)": 2}
    assert report["strata_without_both_arms"] == []

    enrollment.loc[enrollment["assigned_arm"] == "treatment", "stratum"] = "another"
    with pytest.raises(ValueError, match="cannot mix missing and specified strata"):
        analyze_pilot_retention(enrollment, assessments)

    enrollment.loc[enrollment["assigned_arm"] == "control", "stratum"] = "control-only"
    report = analyze_pilot_retention(enrollment, assessments, bootstrap_samples=100)
    assert report["strata_without_both_arms"] == ["another", "control-only"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("duplicate_enrollment", "duplicate randomized learners"),
        ("duplicate_assessment", "only one delayed assessment"),
        ("unknown_learner", "outside the frozen enrollment roster"),
        ("not_independent", "independent=True"),
        ("bad_score", "scores must be in"),
        ("other_course", "outside the frozen enrollment roster"),
    ],
)
def test_retention_analysis_rejects_broken_study_data(change: str, message: str) -> None:
    enrollment, assessments = _inputs()
    if change == "duplicate_enrollment":
        enrollment = pd.concat([enrollment, enrollment.iloc[[0]]], ignore_index=True)
    elif change == "duplicate_assessment":
        assessments = pd.concat([assessments, assessments.iloc[[0]].assign(assessment_event_id="a4")], ignore_index=True)
    elif change == "unknown_learner":
        assessments.loc[0, "user_id"] = "stranger"
    elif change == "not_independent":
        assessments.loc[0, "independent"] = False
    elif change == "bad_score":
        assessments.loc[0, "score"] = 1.5
    elif change == "other_course":
        assessments.loc[0, "course_run_id"] = "another-course"
    with pytest.raises(ValueError, match=message):
        analyze_pilot_retention(enrollment, assessments)


def test_sample_course_keeps_nonparticipants_in_frozen_roster(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "networking_routing_pilot.py"
    spec = importlib.util.spec_from_file_location("networking_routing_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    report = module.run(tmp_path)
    enrollment = pd.read_csv(tmp_path / "enrollment.csv")
    decisions = pd.read_json(tmp_path / "delivery-audit.jsonl", lines=True)
    assert report["synthetic_data"] is True
    assert sum(arm["randomized"] for arm in report["arms"].values()) == 40
    assert len(decisions) < len(enrollment)
    assert report["delivery_audit"]["learners_with_decisions"] == len(decisions)
    assert report["delivery_audit"]["missing_rendered"] == 0
    assert (tmp_path / "retention-report.json").is_file()


def test_delivery_audit_rejects_wrong_assigned_arm() -> None:
    enrollment, assessments = _inputs()
    audit = pd.DataFrame(
        [
            {
                "decision_id": "decision-1",
                "user_id": "t1",
                "course_run_id": "routing",
                "experiment_arm": "control",
                "effective_arm": "control",
                "delivery_mode": "active",
                "rendered_event_id": "render-1",
                "submitted_event_id": "submit-1",
                "scored_event_id": "score-1",
                "fallback_event_id": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="assignment disagrees"):
        analyze_pilot_retention(enrollment, assessments, delivery_audit=audit)

    audit.loc[0, "experiment_arm"] = "treatment"
    audit.loc[0, "delivery_mode"] = "unknown"
    with pytest.raises(ValueError, match="invalid delivery mode"):
        analyze_pilot_retention(enrollment, assessments, delivery_audit=audit)
