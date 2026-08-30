# Run an end-to-end adaptive-practice pilot

This tutorial runs Orchid's reference pilot adapter from historical outcomes to
an analysis-ready export. It is intended for one controlled learning
experiment, not a general LMS integration. The learning product owns identity,
eligibility, rendering, scoring, and the study protocol; Orchid ranks only the
product's approved next exercises.

The reference adapter is deliberately narrow for a first pilot:

- fit an AdaptiveRanker with kt_backbone set to "empirical";
- freeze the model artifact, authoring snapshot, control sequence, and
  eligibility policy for the experiment; and
- use a delayed assessment independent of the exercises Orchid can serve.

For the experimental rationale and operational contract, read [Run a
learning-efficacy pilot](03-learning-pilot.md) and the [pilot integration
contract](04-pilot-integration.md) first.

## 1. Fit a frozen baseline and create the pilot

The inputs below have separate purposes. The completed_attempts table is
historical, scored practice used to fit the model. The authoring_catalog is the
one active content revision that the pilot can deliver; it contains curriculum
constraints as well as item features.

The small data here makes the mechanics runnable. Replace it with a
chronological export from a comparable course population. A real pilot needs
enough historical coverage and a pre-specified sample-size calculation; this
example is not evidence that the policy is ready for learners.

~~~python
from pathlib import Path

import pandas as pd

from orchid_ranker import AdaptiveRanker
from orchid_ranker.decision_store import SQLiteDecisionStore
from orchid_ranker.pilot import (
    AdaptivePracticePilot,
    PilotCatalog,
    PilotRequest,
    SQLiteExperimentAssignmentStore,
    SQLitePilotLifecycleStore,
)

# Each row is a completed binary-scored practice attempt.
completed_attempts = pd.DataFrame(
    [
        {"user_id": "history-1", "item_id": "route-1", "outcome": 1, "timestamp": 1},
        {"user_id": "history-1", "item_id": "route-2", "outcome": 0, "timestamp": 2},
        {"user_id": "history-2", "item_id": "route-1", "outcome": 1, "timestamp": 3},
        {"user_id": "history-2", "item_id": "route-2", "outcome": 1, "timestamp": 4},
        {"user_id": "history-3", "item_id": "route-1", "outcome": 0, "timestamp": 5},
        {"user_id": "history-3", "item_id": "route-3", "outcome": 1, "timestamp": 6},
    ]
)

# One active revision per item for this experiment. Assessment-only content is
# deliberately present but can never appear in a recommendation candidate set.
authoring_catalog = pd.DataFrame(
    [
        {
            "item_id": "route-1",
            "content_version": "2026.1",
            "course_id": "networking",
            "module_id": "routing",
            "category_id": "routing",
            "difficulty": 0.20,
            "assessment_only": False,
            "prerequisites": [],
            "available": True,
            "required": False,
            "authored_sequence_position": 10,
        },
        {
            "item_id": "route-2",
            "content_version": "2026.1",
            "course_id": "networking",
            "module_id": "routing",
            "category_id": "routing",
            "difficulty": 0.45,
            "assessment_only": False,
            "prerequisites": [],
            "available": True,
            "required": False,
            "authored_sequence_position": 20,
        },
        {
            "item_id": "route-3",
            "content_version": "2026.1",
            "course_id": "networking",
            "module_id": "routing",
            "category_id": "routing",
            "difficulty": 0.65,
            "assessment_only": False,
            "prerequisites": ["route-1"],
            "available": True,
            "required": False,
            "authored_sequence_position": 30,
        },
        {
            "item_id": "routing-assessment-1",
            "content_version": "2026.1",
            "course_id": "networking",
            "module_id": "routing",
            "category_id": "routing",
            "difficulty": 0.70,
            "assessment_only": True,
            "prerequisites": [],
            "available": True,
            "required": False,
            "authored_sequence_position": 40,
        },
    ]
)

# All three stores use one durable database. SQLite is appropriate for the
# reference single-host adapter; a multi-host service needs equivalent
# idempotency and immutability guarantees.
database = Path("networking-pilot.sqlite")
decision_store = SQLiteDecisionStore(database)
assignment_store = SQLiteExperimentAssignmentStore(database)
lifecycle_store = SQLitePilotLifecycleStore(database)

ranker = AdaptiveRanker(
    kt_backbone="empirical",
    random_state=42,
    decision_store=decision_store,
).fit(completed_attempts, catalog=authoring_catalog)

catalog = PilotCatalog.from_frame(
    authoring_catalog,
    catalog_version="networking-2026.1",
)
pilot = AdaptivePracticePilot(
    ranker,
    catalog,
    experiment_id="networking-routing-2026q3",
    model_artifact_id="orchid-empirical-2026-08-30",
    authored_policy_version="networking-routing-authored-v3",
    eligibility_rule_version="lms-eligibility-v5",
    treatment_fraction=0.5,
    randomization_salt="a-secret-kept-in-your-service-config",
    assignment_store=assignment_store,
    lifecycle_store=lifecycle_store,
    initial_mode="aa",
)
~~~

PilotCatalog.from_frame validates the required fields and accepts exactly one
active content version for each item. Constructing the pilot writes an
immutable experiment manifest: it binds the catalog digest, model/config
identity, authoring and eligibility versions, allocation fraction, and a digest
of the randomization salt. Reusing an experiment ID with a changed input fails;
start a new experiment instead.

## 2. Serve one idempotent decision

Your LMS calculates the exact approved candidate set before calling Orchid.
Apply accommodations, pacing and attempt limits, availability, prerequisites,
and other course rules here. Do not hand the ranker a broad catalog and expect
it to infer curriculum policy.

~~~python
request = PilotRequest(
    # Stable across HTTP retries for this single next-exercise request.
    request_id="lms-next-501",
    user_id="learner-17",
    course_id="networking",
    module_id="routing",
    course_run_id="networking-2026q3-cohort-a",
    timestamp=1_725_000_000,
    completed_item_ids=(),
    # The LMS has already determined these are the only approved choices.
    candidate_item_ids=("route-1", "route-2"),
    # A baseline band declared before allocation, not learned later.
    stratum="baseline-low",
)

served = pilot.serve(request)

decision_id = served.decision.decision_id
item_id = served.decision.chosen_item_id
content_version = served.chosen_content_version
print(served.arm, served.effective_arm, served.mode, item_id, content_version)
~~~

The arm field is the learner's durable random assignment. The effective_arm
field is what was actually delivered; it differs during A/A, shadow, or halted
modes. The decision ID is derived from the experiment, course run, and LMS
request ID, so request IDs may repeat across experiments and course runs.

Call serve again with the exact same PilotRequest after a timeout; it returns
the original immutable decision. A changed timestamp, candidate set, course
run, or stratum with the same request ID is rejected rather than silently
creating a new recommendation.

The adapter records a decision explanation event automatically. For a designer
audit, read that immutable event rather than recreating an explanation from the
current model:

~~~python
events = lifecycle_store.events(pilot.experiment_id, decision_id)
explanation = next(event.payload for event in events if event.event_type == "explanation")
print(explanation["reason_code"], explanation["ranked_candidates"])
~~~

## 3. Record what really happened

Persist the Orchid decision before rendering. Then record the exact item
version that the learner saw, submitted, and scored. Each event ID is an
application-generated immutable ID. Retrying the same payload is safe; using
one event ID for different content fails.

~~~python
# Render only the item and version returned by served.
pilot.record_rendered(
    decision_id,
    event_id="lms-render-9001",
    item_id=item_id,
    content_version=content_version,
    timestamp=1_725_000_005,
)

pilot.record_submitted(
    decision_id,
    event_id="lms-submission-9001",
    item_id=item_id,
    content_version=content_version,
    timestamp=1_725_000_080,
)

# The score event ID must be globally unique in the LMS, not only unique per
# learner or decision. Outcome is binary; reward can be supplied instead for a
# finite non-binary product reward.
linked_outcome = pilot.record_scored(
    decision_id,
    outcome_event_id="lms-score-9001",
    item_id=item_id,
    content_version=content_version,
    outcome=1,
    timestamp=1_725_000_100,
)
~~~

record_scored rejects a score without a prior matching render and submission.
It stores score evidence before applying it. In active treatment, the learner's
local state advances but the aggregate empirical model remains frozen; control,
A/A, shadow, and halted scores are audit-only.

## 4. Import the independent outcome and export the analysis frame

The efficacy outcome should be a later assessment with unserved or isomorphic
content. Import it separately and explicitly mark it independent. The importer
does not use assessment scores to update the live ranker.

~~~python
pilot.import_delayed_assessments(
    pd.DataFrame(
        [
            {
                "assessment_event_id": "routing-retention-17-day14",
                "user_id": "learner-17",
                "course_run_id": "networking-2026q3-cohort-a",
                "assessment_form_version": "routing-retention-v2",
                "timestamp": 1_726_209_600,  # about 14 days later
                "score": 0.80,
                "independent": True,
            }
        ]
    )
)

# One row per served decision, with joined delivery evidence, immutable policy
# metadata, explanations/shadow proposals, and independent assessment records.
analysis = pilot.analysis_frame()
analysis.to_csv("networking-routing-2026q3-analysis.csv", index=False)
~~~

Use this export to check linkage and protocol health by assigned arm before
running the pre-registered analysis: candidate-set violations, missing render
or submission evidence, outcome coverage, fallbacks, difficulty jumps, and
assessment completion. It intentionally is not a causal-estimation API.

## 5. Roll out deliberately

Pilot mode is durable operational state. Set it through a privileged service
path with a unique operation event ID; never infer it from deployment
configuration.

~~~python
# 1. initial_mode="aa" already routes both assigned arms to authored control.
# After validating those joins, transition later requests to shadow mode.
# Orchid computes and records a proposal, but the authored item still renders.
pilot.set_mode(
    "shadow",
    event_id="ops-shadow-start",
    timestamp=1_725_000_200,
    reason="review candidate and explanation quality",
)

# 2. Only assigned treatment learners now receive Orchid's frozen action.
pilot.set_mode(
    "active",
    event_id="ops-active-start",
    timestamp=1_725_000_300,
    reason="start randomized delivery",
)

# 3. Kill switch: future requests receive authored control. This is
# irreversible for this experiment ID, preserving the operating history.
pilot.set_mode(
    "halted",
    event_id="ops-halt",
    timestamp=1_725_000_400,
    reason="pre-specified safety stop",
)
~~~

In shadow, every learner still receives authored control while a
shadow_proposal event captures Orchid's non-delivered ranking. In halted,
sticky assignment remains available for intention-to-treat analysis but future
delivery is forced to control and a KILL_SWITCH fallback event is saved. A
halted pilot cannot be re-enabled; use a fresh experiment ID after reviewing
the stop.

## 6. Recover safely after an interruption

The decision, assignment, and lifecycle stores are separate durable
boundaries. A score lifecycle event can commit before its outcome projection,
and an outcome can commit before a live learner-state update. Repair either
case idempotently while continuing the same in-memory ranker projection:

~~~python
# A scored lifecycle event may have committed immediately before its outcome
# projection. This completes those score events idempotently.
repaired_scores = pilot.recover_scored_events()

# If an outcome was durable but the process stopped before learner-state
# application, replay only the pending projection. Repeated calls are no-ops.
replayed = ranker.replay_pending_outcomes()
~~~

After a process restart, reopen all stores and recreate the ranker from the
same frozen baseline and catalog. Recreate the pilot with the same immutable
constructor arguments, then use the explicit rebuild path once instead of
pending replay:

~~~python
replayed_from_baseline = pilot.rebuild_state_from_baseline()
~~~

It deliberately ignores prior in-memory application checkpoints and replays all
stateful treatment outcomes in chronological learner order. Do not call it on
a ranker that already contains those updates, or learner state will advance
twice.

When the service is finished, close the SQLite stores during orderly shutdown:

~~~python
lifecycle_store.close()
assignment_store.close()
decision_store.close()
~~~

## Production checklist

Before moving beyond the reference pilot, verify the integration can answer
"what was eligible, selected, rendered, submitted, scored, and assessed?" for
every decision. Keep these invariants in monitoring and deployment checks:

- every request has a stable LMS request ID, every score has an LMS-global
  score-event ID, and all retries are byte-for-byte equivalent;
- assignment is learner-level and sticky, with strata determined before
  allocation;
- candidate_item_ids is the exact LMS-approved set and assessment-only content
  never appears in it;
- catalog/version, model artifact/config, authored policy, eligibility rule,
  and allocation contract do not change under an experiment ID;
- the control sequence and independent delayed assessment stay unchanged
  through the study; and
- a kill switch, on-call ownership, pre-specified stop criteria, and a
  missing-data plan are in place before active delivery.

Once the study is complete, use the saved analysis export and original
protocol to make a narrow claim about the tested course population. A positive
result is a reason to replicate, not evidence that every learning product
should use the same policy.
