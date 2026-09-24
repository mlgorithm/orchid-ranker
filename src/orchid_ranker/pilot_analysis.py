"""One-course, learner-level analysis for a pre-registered pilot.

The primary endpoint is an independent retention score in [0, 1] within a
fixed window around each learner's pre-declared assessment due date. A missing
score contributes zero to the primary composite endpoint. The report also
shows observed-only scores and worst-case missing-outcome bounds so that the
composite cannot be mistaken for retained mastery alone.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["analyze_pilot_retention"]

SECONDS_PER_DAY = 86_400


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _finite_numbers(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return numeric.astype(float)


def analyze_pilot_retention(
    enrollment: pd.DataFrame,
    assessments: pd.DataFrame,
    *,
    delivery_audit: pd.DataFrame | None = None,
    window_days: float = 3.0,
    analysis_timestamp: float | None = None,
    bootstrap_samples: int = 2_000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Analyze all randomized learners in one course run, by assigned arm.

    ``enrollment`` must be frozen when learners are assigned, before any
    intervention outcome. ``assessment_due_timestamp`` is a fixed Unix-second
    deadline from the study calendar, not a post-treatment completion time.
    ``assessments`` contains independent, out-of-bank retention scores. An
    assessment outside the due-date window is counted as missing. Analysis
    waits until the last learner's window has closed; ``analysis_timestamp``
    can freeze that cutoff for a reproducible report.

    The percentile bootstrap interval is descriptive, especially for small
    samples. Power, success thresholds, and stop rules belong in a protocol
    written before the experiment starts.
    """
    _require_columns(
        enrollment,
        {"user_id", "course_run_id", "assigned_arm", "stratum", "assessment_due_timestamp"},
        "enrollment",
    )
    _require_columns(
        assessments,
        {
            "assessment_event_id",
            "user_id",
            "course_run_id",
            "assessment_form_version",
            "timestamp",
            "score",
            "independent",
        },
        "assessments",
    )
    if not np.isfinite(window_days) or window_days < 0:
        raise ValueError("window_days must be finite and non-negative")
    window_seconds = window_days * SECONDS_PER_DAY
    if not np.isfinite(window_seconds):
        raise ValueError("window_days is too large")
    current_timestamp = time.time()
    if analysis_timestamp is None:
        analysis_timestamp = current_timestamp
    if isinstance(analysis_timestamp, (bool, np.bool_)):
        raise ValueError("analysis_timestamp must be a finite Unix timestamp")
    try:
        analysis_timestamp = float(analysis_timestamp)
    except (TypeError, ValueError) as error:
        raise ValueError("analysis_timestamp must be a finite Unix timestamp") from error
    if not np.isfinite(analysis_timestamp):
        raise ValueError("analysis_timestamp must be a finite Unix timestamp")
    if analysis_timestamp > current_timestamp:
        raise ValueError("analysis_timestamp cannot be in the future")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if enrollment.empty:
        raise ValueError("enrollment cannot be empty")
    if enrollment[["user_id", "course_run_id", "assigned_arm"]].isna().any().any():
        raise ValueError("enrollment identities and arms cannot be missing")
    if enrollment["stratum"].isna().any() and not enrollment["stratum"].isna().all():
        raise ValueError("enrollment cannot mix missing and specified strata")
    if enrollment["course_run_id"].nunique() != 1:
        raise ValueError("analyze one course run at a time")
    if enrollment.duplicated(["user_id", "course_run_id"]).any():
        raise ValueError("enrollment has duplicate randomized learners")
    if set(enrollment["assigned_arm"]) != {"control", "treatment"}:
        raise ValueError("enrollment must contain control and treatment arms only")

    roster = enrollment.copy()
    roster["assessment_due_timestamp"] = _finite_numbers(
        roster["assessment_due_timestamp"], "assessment_due_timestamp"
    )
    if roster["stratum"].isna().all():
        roster["stratum"] = "(unstratified)"
    assessment_window_end = float(roster["assessment_due_timestamp"].max() + window_seconds)
    if not np.isfinite(assessment_window_end):
        raise ValueError("assessment window end must be a finite Unix timestamp")
    if analysis_timestamp < assessment_window_end:
        raise ValueError("the assessment window has not closed for every randomized learner")
    scores = assessments.copy()
    if not scores.empty:
        if scores[["assessment_event_id", "user_id", "course_run_id", "assessment_form_version"]].isna().any().any():
            raise ValueError("assessment identities and form versions cannot be missing")
        if not scores["independent"].map(lambda value: isinstance(value, (bool, np.bool_)) and value).all():
            raise ValueError("all delayed assessments must be explicitly independent=True")
        if scores.duplicated(["user_id", "course_run_id"]).any():
            raise ValueError("only one delayed assessment per randomized learner is allowed")
        if scores["assessment_event_id"].duplicated().any():
            raise ValueError("assessment_event_id must be unique")
        if scores["assessment_form_version"].nunique() != 1:
            raise ValueError("all assessments must use the same form version")
        scores["timestamp"] = _finite_numbers(scores["timestamp"], "timestamp")
        if (scores["timestamp"] > analysis_timestamp).any():
            raise ValueError("assessment timestamp cannot be after analysis_timestamp")
        scores["score"] = _finite_numbers(scores["score"], "score")
        if not scores["score"].between(0, 1).all():
            raise ValueError("assessment scores must be in [0, 1]")
        unknown = scores.merge(
            roster[["user_id", "course_run_id"]],
            on=["user_id", "course_run_id"],
            how="left",
            indicator=True,
        )
        if (unknown["_merge"] != "both").any():
            raise ValueError("assessments contain learners outside the frozen enrollment roster")

    joined = roster.merge(
        scores[["user_id", "course_run_id", "timestamp", "score"]],
        on=["user_id", "course_run_id"],
        how="left",
        validate="one_to_one",
    )
    joined["in_window"] = joined["timestamp"].notna() & (
        (joined["timestamp"] - joined["assessment_due_timestamp"]).abs() <= window_seconds
    )
    joined["primary_score"] = joined["score"].where(joined["in_window"], 0.0)
    group_scores: dict[str, np.ndarray] = {}
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("control", "treatment"):
        group = joined.loc[joined["assigned_arm"] == arm]
        observed = group.loc[group["in_window"], "score"]
        values = group["primary_score"].to_numpy(dtype=float)
        group_scores[arm] = values
        arms[arm] = {
            "randomized": len(group),
            "assessed_in_window": len(observed),
            "assessment_rate": float(len(observed) / len(group)),
            "out_of_window": int((group["timestamp"].notna() & ~group["in_window"]).sum()),
            "primary_composite_mean": float(values.mean()),
            "observed_score_mean": float(observed.mean()) if len(observed) else None,
            "strata": {str(key): int(value) for key, value in group["stratum"].value_counts().items()},
        }

    difference = float(group_scores["treatment"].mean() - group_scores["control"].mean())
    rng = np.random.default_rng(random_seed)
    control = group_scores["control"]
    treatment = group_scores["treatment"]
    interval: list[float] | None = None
    if len(control) >= 2 and len(treatment) >= 2:
        sampled_differences = np.empty(bootstrap_samples)
        for sample in range(bootstrap_samples):
            sampled_differences[sample] = (
                rng.choice(treatment, size=len(treatment), replace=True).mean()
                - rng.choice(control, size=len(control), replace=True).mean()
            )
        interval = [float(value) for value in np.quantile(sampled_differences, [0.025, 0.975])]
    balance = pd.crosstab(roster["stratum"], roster["assigned_arm"])
    balance = balance.reindex(columns=["control", "treatment"], fill_value=0)
    strata_without_both_arms = [
        str(stratum) for stratum, counts in balance.iterrows()
        if counts["control"] == 0 or counts["treatment"] == 0
    ]
    missing_control = 1 - arms["control"]["assessment_rate"]
    missing_treatment = 1 - arms["treatment"]["assessment_rate"]
    report: dict[str, Any] = {
        "course_run_id": str(roster["course_run_id"].iloc[0]),
        "assessment_form_version": str(scores["assessment_form_version"].iloc[0]) if not scores.empty else None,
        "window_days": float(window_days),
        "analysis_timestamp": analysis_timestamp,
        "assessment_window_end_timestamp": assessment_window_end,
        "endpoint": "in-window independent score; missing or out-of-window assessment = 0",
        "arms": arms,
        "strata_without_both_arms": strata_without_both_arms,
        "treatment_minus_control": difference,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "bootstrap_95_percent_interval": interval,
        "treatment_minus_control_missing_bounds": [difference - missing_control, difference + missing_treatment],
        "assessment_rate_difference": float(
            arms["treatment"]["assessment_rate"] - arms["control"]["assessment_rate"]
        ),
    }
    if delivery_audit is not None:
        _require_columns(
            delivery_audit,
            {
                "decision_id", "user_id", "course_run_id", "experiment_arm", "effective_arm",
                "delivery_mode", "rendered_event_id", "submitted_event_id", "scored_event_id",
                "fallback_event_id",
            },
            "delivery_audit",
        )
        if delivery_audit["decision_id"].duplicated().any():
            raise ValueError("delivery_audit contains duplicate decision IDs")
        if not delivery_audit["delivery_mode"].isin({"aa", "shadow", "active", "halted"}).all():
            raise ValueError("delivery_audit contains an invalid delivery mode")
        if not delivery_audit["effective_arm"].isin({"control", "treatment"}).all():
            raise ValueError("delivery_audit contains an invalid effective arm")
        merged = delivery_audit.merge(
            roster[["user_id", "course_run_id", "assigned_arm"]],
            on=["user_id", "course_run_id"],
            how="left",
            validate="many_to_one",
        )
        if merged["assigned_arm"].isna().any():
            raise ValueError("delivery_audit contains decisions outside the enrollment roster")
        if (merged["experiment_arm"] != merged["assigned_arm"]).any():
            raise ValueError("delivery_audit assignment disagrees with the enrollment roster")
        expected_treatment = (merged["delivery_mode"] == "active") & (merged["assigned_arm"] == "treatment")
        if (merged["effective_arm"].eq("treatment") != expected_treatment).any():
            raise ValueError("delivery_audit has an arm inconsistent with mode and assignment")
        report["delivery_audit"] = {
            "decisions": len(merged),
            "learners_with_decisions": int(merged["user_id"].nunique()),
            "active_decisions": int(merged["delivery_mode"].eq("active").sum()),
            "treatment_deliveries": int(merged["effective_arm"].eq("treatment").sum()),
            "missing_rendered": int(merged["rendered_event_id"].isna().sum()),
            "missing_submitted": int(merged["submitted_event_id"].isna().sum()),
            "missing_scored": int(merged["scored_event_id"].isna().sum()),
            "fallback_decisions": int(merged["fallback_event_id"].notna().sum()),
        }
    return report
