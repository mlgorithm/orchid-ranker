"""Analyze a frozen enrollment roster and independent delayed assessments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from orchid_ranker.pilot_analysis import analyze_pilot_retention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("enrollment", type=Path, help="frozen randomized roster CSV")
    parser.add_argument("assessments", type=Path, help="independent retention assessment CSV")
    parser.add_argument("--window-days", type=float, default=3.0, help="pre-specified half-width of the assessment window (default: 3 days)")
    parser.add_argument("--analysis-timestamp", type=float, help="Unix seconds at analysis cutoff; default: current UTC time")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000, help="bootstrap resamples (default: 2000)")
    parser.add_argument("--random-seed", type=int, default=42, help="bootstrap random seed (default: 42)")
    parser.add_argument("--delivery-audit", type=Path, help="decision-level JSONL from pilot.analysis_frame()")
    parser.add_argument("--output", type=Path, help="write report JSON here; default: stdout")
    args = parser.parse_args()
    enrollment = pd.read_csv(
        args.enrollment,
        dtype={"user_id": "string", "course_run_id": "string", "assigned_arm": "string", "stratum": "string"},
    )
    assessments = pd.read_csv(
        args.assessments,
        dtype={
            "assessment_event_id": "string", "user_id": "string", "course_run_id": "string",
            "assessment_form_version": "string",
        },
    )
    delivery_audit = None
    if args.delivery_audit:
        delivery_audit = pd.read_json(
            args.delivery_audit,
            lines=True,
            dtype={"decision_id": "string", "user_id": "string", "course_run_id": "string"},
        )
    synthetic_marker = None
    if "synthetic_data" in enrollment:
        markers = enrollment["synthetic_data"].astype(str).str.lower().unique().tolist()
        if len(markers) != 1 or markers[0] not in ("true", "false"):
            parser.error("enrollment.synthetic_data must contain the same true or false value for every learner")
        synthetic_marker = markers[0] == "true"
    try:
        report = analyze_pilot_retention(
            enrollment,
            assessments,
            delivery_audit=delivery_audit,
            window_days=args.window_days,
            analysis_timestamp=args.analysis_timestamp,
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.random_seed,
        )
    except ValueError as error:
        parser.error(str(error))
    if synthetic_marker is not None:
        report["synthetic_data"] = synthetic_marker
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
