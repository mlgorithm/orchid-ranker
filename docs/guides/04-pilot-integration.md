# Pilot integration contract

Use Orchid as a decision component inside a learning product. The product
keeps ownership of curriculum eligibility, experiment assignment, durable
state, and the learner experience; Orchid ranks the approved exercise set and
records its chosen action.

## Keep these records separate and joinable

| Record | Minimum fields |
| --- | --- |
| Catalog import | `catalog_version`, `course_id`, `module_id`, `item_id`, `content_version`, skill/category, difficulty, prerequisites, `assessment_only` |
| Recommendation request | `request_id`, learner, course run, catalog version, experiment arm, eligibility-rule version, exact eligible IDs, timestamp |
| Served decision | `request_id`/`decision_id`, chosen and actually rendered item, Orchid policy/model artifact ID, catalog and eligibility versions, candidate IDs, fallback/reason code |
| Practice outcome | unique outcome-event ID, `decision_id`, item/content version, rendered/submitted/scored timestamps, binary result, invalid/abandoned status |
| Delayed assessment | learner, assessment instance/form version, timestamp, score or mastery result, and proof its items were not practice candidates |

Store the complete candidate list—not only its hash. A candidate hash can help
detect tampering or accidental changes, but it cannot reproduce the decision.

## Route the experiment outside Orchid

Randomize each learner once, persist that assignment, and retain it for the
entire experiment. Stratify by baseline mastery band or cohort when the study
requires it. The application then routes each request:

```text
learner request
  -> eligibility + author controls
  -> sticky experiment assignment
  -> authored control OR frozen Orchid treatment
  -> immutable decision record
  -> rendered exercise
  -> scored outcome joined by decision ID
```

The control is the existing static/authored practice path. Log control choices
in the same external decision schema, even though Orchid did not choose them.
For the first efficacy study, freeze the empirical/hybrid treatment with
`exploration=0.0`; do not introduce CQL or adaptive exploration as a second
uncontrolled intervention.

## Run the reference adapter locally

`orchid_ranker.pilot` implements the contract for a single-host reference
integration. It validates one active catalog snapshot, enforces author rules,
stores sticky learner-level assignments, and routes both the authored control
and frozen Orchid treatment through the same decision/outcome store. It is not
an LMS connector: map your platform's requests and score events to this small
boundary.

```python
from orchid_ranker import AdaptiveRanker
from orchid_ranker.decision_store import SQLiteDecisionStore
from orchid_ranker.pilot import (
    AdaptivePracticePilot,
    PilotCatalog,
    PilotRequest,
    SQLiteExperimentAssignmentStore,
    SQLitePilotLifecycleStore,
)

catalog = PilotCatalog.from_frame(
    authoring_catalog,
    catalog_version="networking-2026.1",
)
database = "networking-pilot.sqlite"
ranker = AdaptiveRanker(
    kt_backbone="empirical",
    decision_store=SQLiteDecisionStore(database),
).fit(completed_attempts, catalog=authoring_catalog)
pilot = AdaptivePracticePilot(
    ranker,
    catalog,
    experiment_id="networking-routing-pilot",
    model_artifact_id="orchid-empirical-2026-08-30",
    authored_policy_version="routing-authored-v1",
    assignment_store=SQLiteExperimentAssignmentStore(database),
    lifecycle_store=SQLitePilotLifecycleStore(database),
)

served = pilot.serve(PilotRequest(
    request_id=lms_request_id,
    user_id=learner_id,
    course_id="networking",
    module_id="routing",
    timestamp=event_time,
    course_run_id=cohort_id,
    completed_item_ids=completed_exercise_ids,
    candidate_item_ids=platform_eligible_ids,
    stratum=baseline_mastery_band,
))
render_exercise(served.decision.chosen_item_id, served.chosen_content_version)
pilot.record_rendered(
    served.decision.decision_id,
    event_id=lms_render_event_id,
    item_id=served.decision.chosen_item_id,
    content_version=served.chosen_content_version,
    timestamp=rendered_at,
)

pilot.record_submitted(
    served.decision.decision_id,
    event_id=lms_submission_event_id,
    item_id=served.decision.chosen_item_id,
    content_version=served.chosen_content_version,
    timestamp=submitted_at,
)

# When it is scored, retrying this exact payload is safe. The LMS-global
# score-event ID is also globally unique in Orchid's decision store.
pilot.record_scored(
    served.decision.decision_id,
    outcome=correct,
    timestamp=scored_at,
    outcome_event_id=lms_score_event_id,
    item_id=served.decision.chosen_item_id,
    content_version=served.chosen_content_version,
)
```

The active catalog requires `content_version`, `course_id`, `module_id`, a
skill/category, `difficulty`, `assessment_only`, `prerequisites`, `available`,
`required`, and `authored_sequence_position`. The provided
`candidate_item_ids` must already be exact LMS-eligible IDs: the adapter
rejects a candidate that is assessment-only, unavailable, completed, or
prerequisite-blocked rather than silently relaxing the authoring rules.

`model_artifact_id`, `catalog_version`, eligibility-rule version, assigned and
effective arm, content versions, and reason code are persisted in immutable
decision metadata. A second immutable lifecycle event derives the explanation
from that exact logged decision, so the explanation cannot drift from the
served action. The adapter rejects a score unless the exact selected content
version was recorded as both rendered and submitted. Pass the LMS's globally
unique outcome-event ID when a score arrives; Orchid rejects any attempt to
reuse it for another decision.

The treatment’s learner state may adapt, but its aggregate empirical model is
kept frozen for the study; control, A/A, shadow, and halted outcomes are kept
as audit evidence only. If score persistence succeeds but the process fails
before model application, restore the same model baseline and call
`ranker.replay_pending_outcomes()`; if the score event itself was saved first,
call `pilot.recover_scored_events()`. After restoring a new model baseline
rather than continuing the old in-memory state, call
`pilot.rebuild_state_from_baseline()` exactly once instead; it deliberately
replays all stateful outcomes regardless of the prior process's checkpoints.

## Make author rules hard constraints

Build the candidate list from the current catalog version after applying:

- course/module scope and availability;
- assessment holdouts;
- prerequisite completion;
- required/locked items, attempt caps, and accommodations; and
- a deterministic authored fallback sequence.

The resulting list is the only set Orchid may rank. Include an explanation or
fallback code in your integration record, such as `AUTHORED_REQUIRED`,
`PREREQUISITE_MET`, `TARGET_SKILL`, `PREDICTED_CHALLENGE`,
`SPARSE_SUPPORT_FALLBACK`, or `SERVICE_FALLBACK`.

## Roll out in four stages

1. **A/A:** call `pilot.set_mode("aa", ...)`. Both assigned arms render the
   authored sequence while the assignment and all delivery evidence are
   audited.
2. **Shadow:** call `pilot.set_mode("shadow", ...)`. Orchid computes and
   stores a proposal, but the LMS still renders the authored item. Review
   candidate construction, prerequisites, challenge predictions, and fallback
   rates with learning designers.
3. **Frozen randomized pilot:** call `pilot.set_mode("active", ...)`. Orchid
   is served only to its assigned treatment arm; retain the same authored
   control, catalog version, eligibility rules, and time allowance through the
   delayed assessment.
4. **Kill switch:** call `pilot.set_mode("halted", ...)` to force all future
   decisions to the authored control with `KILL_SWITCH` evidence. A halted
   experiment cannot be re-enabled; start a new experiment ID for a later run.

Use `pilot.import_delayed_assessments(...)` only for explicitly independent
assessment events, then use `pilot.analysis_frame()` as the joined delivery
export. The adapter does not calculate a causal effect or replace a
pre-registered analysis plan.
5. **Replication:** Re-run the fixed protocol in a second course before making
   a broad product claim.

Keep a one-action kill switch that routes all future treatment requests to the
authored fallback. Log each fallback or rollback and preserve learners in an
intention-to-treat analysis.

## Decide success before launch

The primary outcome should be retained mastery on an independently authored,
delayed assessment using unserved or isomorphic items—for example, 14 or 28
days after module completion. Immediate exercise correctness informs
adaptation, but it is not the efficacy claim.

Pre-specify sample size, minimum detectable effect, missing-data handling,
randomization unit/strata, stop rules, and subgroups. Monitor candidate,
decision, outcome, and assessment join coverage by arm; assessment-holdout and
prerequisite violations must remain zero. Also watch dropout, repeated
failures, excessive repetition, difficulty jumps, and service fallbacks.

This design follows the [IES practice guide on organizing instruction and
practice](https://ies.ed.gov/ncee/wwc/Docs/PracticeGuide/20072004.pdf) and the
[WWC procedures handbook](https://ies.ed.gov/ncee/wwc/Docs/referenceresources/Final_WWC-HandbookVer5.0-0-508.pdf)
on assignment, clustering, and missing outcome data.
