"""Synthetic one-course pilot; run with: python scripts/networking_routing_pilot.py DIR."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from orchid_ranker import AdaptiveRanker
from orchid_ranker.decision_store import InMemoryDecisionStore
from orchid_ranker.pilot import (
    AdaptivePracticePilot,
    InMemoryExperimentAssignmentStore,
    InMemoryPilotLifecycleStore,
    PilotCatalog,
    PilotRequest,
)
from orchid_ranker.pilot_analysis import analyze_pilot_retention

START = 1_780_000_000.0
COURSE_RUN = "networking-routing-sample"


def run(output_dir: Path) -> dict:
    """Exercise the adapter and write explicitly synthetic analysis inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    historical = pd.DataFrame(
        [
            {"user_id": f"history-{user}", "item_id": f"route-{item}", "outcome": outcome, "timestamp": 10 * user + item}
            for user, outcomes in enumerate(((1, 0), (0, 1), (1, 1), (0, 0)), start=1)
            for item, outcome in enumerate(outcomes, start=1)
        ]
    )
    catalog_frame = pd.DataFrame(
        [
            {
                "item_id": f"route-{item}",
                "content_version": "sample-v1",
                "course_id": "networking",
                "module_id": "routing",
                "category_id": "routing",
                "difficulty": difficulty,
                "assessment_only": assessment_only,
                "prerequisites": [],
                "available": True,
                "required": False,
                "authored_sequence_position": item * 10,
            }
            for item, difficulty, assessment_only in ((1, 0.25, False), (2, 0.65, False), (3, 0.50, True))
        ]
    )
    ranker = AdaptiveRanker(
        kt_backbone="empirical",
        random_state=42,
        decision_store=InMemoryDecisionStore(),
    ).fit(historical, catalog=catalog_frame)
    pilot = AdaptivePracticePilot(
        ranker,
        PilotCatalog.from_frame(catalog_frame, catalog_version="sample-v1"),
        experiment_id="networking-routing-sample",
        model_artifact_id="sample-empirical-v1",
        authored_policy_version="sample-authored-v1",
        eligibility_rule_version="sample-eligibility-v1",
        treatment_fraction=0.5,
        randomization_salt="synthetic-example-only",
        assignment_store=InMemoryExperimentAssignmentStore(),
        lifecycle_store=InMemoryPilotLifecycleStore(),
        initial_mode="active",
    )
    assessment_rows = []
    # Freeze the full randomized roster before any learner receives practice.
    for index in range(40):
        user_id = f"learner-{index:03d}"
        stratum = "baseline-low" if index % 2 else "baseline-high"
        pilot.enroll(
            user_id,
            course_run_id=COURSE_RUN,
            timestamp=START - 3_600,
            stratum=stratum,
        )
    enrollment = pilot.enrollment_frame()
    enrollment["assessment_due_timestamp"] = START + 14 * 86_400
    enrollment["synthetic_data"] = True
    for index, enrolled in enrollment.iterrows():
        user_id = str(enrolled["user_id"])
        # A few randomized learners have no practice request, but remain in ITT.
        if index % 13 != 0:
            served = pilot.serve(
                PilotRequest(
                    request_id=f"practice-{index}",
                    user_id=user_id,
                    course_id="networking",
                    module_id="routing",
                    course_run_id=COURSE_RUN,
                    timestamp=START + index * 60,
                    candidate_item_ids=("route-1", "route-2"),
                    stratum=enrolled["stratum"],
                )
            )
            decision_id = served.decision.decision_id
            item_id = served.decision.chosen_item_id
            version = served.chosen_content_version
            pilot.record_rendered(
                decision_id, event_id=f"render-{index}", item_id=item_id, content_version=version,
                timestamp=START + index * 60 + 5,
            )
            pilot.record_submitted(
                decision_id, event_id=f"submit-{index}", item_id=item_id, content_version=version,
                timestamp=START + index * 60 + 30,
            )
            pilot.record_scored(
                decision_id, outcome_event_id=f"score-{index}", item_id=item_id, content_version=version,
                outcome=int(index % 3 != 0), timestamp=START + index * 60 + 40,
            )
        # Synthetic retention scores have no treatment effect by construction.
        if index % 11 != 0:
            assessment_rows.append(
                {
                    "assessment_event_id": f"retention-{index}",
                    "user_id": user_id,
                    "course_run_id": COURSE_RUN,
                    "assessment_form_version": "sample-retention-v1",
                    "timestamp": START + 14 * 86_400,
                    "score": (index % 6 + 3) / 10,
                    "independent": True,
                }
            )
    assessments = pd.DataFrame(assessment_rows)
    # All preassigned learners can have independent outcomes, including those
    # who never requested practice. The separate roster defines the ITT cohort.
    pilot.import_delayed_assessments(assessments)
    enrollment.to_csv(output_dir / "enrollment.csv", index=False)
    assessments.to_csv(output_dir / "assessments.csv", index=False)
    audit = pilot.analysis_frame()
    audit.to_json(output_dir / "delivery-audit.jsonl", orient="records", lines=True)
    report = analyze_pilot_retention(enrollment, assessments, delivery_audit=audit)
    report["synthetic_data"] = True
    (output_dir / "retention-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/networking_routing_pilot.py OUTPUT_DIR")
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
