# API reference

`AdaptiveRanker` is the package-root API. It fits from completed learning
attempts, ranks an application-approved candidate set, and accepts feedback.
Supporting modules add durable audit records, catalog validation, and a
reference controlled-pilot adapter.

```python
from orchid_ranker import AdaptiveRanker
```

Use the [production-serving guide](guides/02-serve-streaming.md) for the
ordinary request loop and the [pilot integration contract](guides/04-pilot-integration.md)
when a next-item choice is part of an efficacy study.

## At a glance

| Need | API | Result |
| --- | --- | --- |
| Fit a learning policy | `AdaptiveRanker().fit(events, ...)` | The fitted ranker |
| Rank safe exercises | `ranker.recommend(user_id, ids)` | `AdaptiveLearningRecommendation` values |
| Adapt after an attempt | `ranker.observe(...)` | Updates the learner state |
| Keep a reproducible serving record | `ranker.recommend_and_log(...)` | Ranked items and immutable `LoggedDecision` |
| Attach a delayed score safely | `ranker.observe_decision(...)` | Durable `DecisionOutcome` and normally a learner update |
| Validate authoring data | `validate_learning_catalog(frame)` | Normalized exercises and all diagnostics |
| Run a reference experiment adapter | `AdaptivePracticePilot` | Assignment, decisions, lifecycle evidence, analysis frames |

All timestamps must be finite, non-negative numeric values in one consistent
application-defined unit. Outcomes are binary `0` or `1`; do not turn missing,
abandoned, or unscored work into `0`.

## AdaptiveRanker

### Construct and fit

```python
ranker = AdaptiveRanker(
    kt_backbone="empirical",  # optional AdaptiveRankerConfig override
    decision_store=store,     # optional durable store
)

ranker.fit(
    events,
    user_col="user_id",
    item_col="item_id",
    outcome_col="outcome",
    timestamp_col="timestamp",
    category_col=None,
    difficulty_col=None,
    catalog=None,
    catalog_item_col="item_id",
    catalog_category_col="category_id",
    catalog_difficulty_col="difficulty",
    prerequisite_by_concept=None,
)
```

Pass either an `AdaptiveRankerConfig` as the first argument or its fields as
keyword overrides; unknown names raise `TypeError`. `fit` returns the same
ranker and expects a pandas DataFrame of completed attempts with learner, item,
binary outcome, and timestamp columns.

A canonical catalog is the preferred source of item category and difficulty. It
validates historical coverage and registers catalog-only items for future
serving. Its default columns are `item_id`, `category_id`, and `difficulty`;
use the three `catalog_*_col` arguments for a different source schema.
`prerequisite_by_concept` constrains ranking, but it does not replace the
application's item-level eligibility.

```python
ranker = AdaptiveRanker().fit(
    completed_attempts,
    catalog=exercise_catalog,
    prerequisite_by_concept={"advanced": ["fundamentals"]},
)
```

Use `ranker.is_fitted` after a successful fit. `ranker.learning_readiness()`
returns data support, outcome balance, metadata coverage, selected tracer, and
recommended next steps; it is rollout evidence, not a model-quality guarantee.
`ranker.diagnostics()` additionally reports adaptive, policy, semantic, and
decision-store state.

Sparse learning data automatically uses Orchid's empirical learner. For a first
frozen pilot, explicitly set `kt_backbone="empirical"`.

### `recommend`

```python
recommendations = ranker.recommend(
    user_id,
    candidate_item_ids,
    top_k=3,
    concept_goal=None,
    context_hash=None,
)
```

Ranks the supplied candidate set, returning
`AdaptiveLearningRecommendation` values in descending order.

| Field | Meaning |
| --- | --- |
| `item_id` | Candidate in ranking order |
| `score` | Relative score within this request |
| `outcome_probability` | Estimated probability of a positive next attempt |
| `difficulty`, `category_id`, `competence` | Available learning context |
| `item_support`, `concept_support`, `feedback_supported` | Evidence/support indicators |
| `prerequisites_met`, `recent_repetition` | Constraint and sequencing signals |

Scores are for ranking the current candidates, not globally calibrated learning
values. `outcome_probability` estimates the next practice result, not retained
mastery. An explicit empty sequence returns `[]` and never means “all known
items.” `candidate_item_ids=None` only permits a configured candidate generator
or explicit catalog fallback.

The application owns eligibility: remove unavailable, completed,
accommodation-inappropriate, prerequisite-blocked, and assessment-only items
before calling `recommend`. `PilotCatalog` enforces a published version of
these controls for the reference pilot.

### `observe` and `register_items`

```python
ranker.observe(
    user_id=learner_id,
    item_id=exercise_id,
    outcome=1,
    timestamp=attempt_time,
    update_global=True,
)

ranker.register_items(catalog)
```

`observe` advances the live learner state from a completed attempt. Set
`update_global=False` only where a learner may adapt while shared aggregate
statistics must remain frozen, such as treatment traffic in a controlled pilot.

`register_items` adds catalog items absent from fitting history with an OOV
prior, making them available for recommendation and feedback immediately.
Refit later to learn item-specific parameters from their accumulated outcomes.

## Durable decisions and delayed outcomes

Use a decision store whenever a served exercise must be reproducible or
evaluated later.

```python
from orchid_ranker.decision_store import SQLiteDecisionStore

store = SQLiteDecisionStore("orchid.sqlite")
ranker = AdaptiveRanker(decision_store=store).fit(completed_attempts)
```

`LoggedDecision` records the full candidate list, chosen item, score/probability
evidence, policy identity, and immutable application metadata.
`DecisionOutcome` links one later result to one decision. Its `apply_state`
flag says whether that event may change adaptive state; `update_global` says
whether a permitted update may change shared aggregate statistics.

### `recommend_and_log`

```python
ranked, decision = ranker.recommend_and_log(
    user_id=learner_id,
    candidate_item_ids=eligible_ids,
    timestamp=request_time,
    top_k=1,
    exploration=0.0,
    decision_id=request_id,
    policy_version="course-2026.1",
    decision_metadata={"catalog_version": "2026.1"},
)
```

This is the serving method for auditable decisions. It requires a non-empty
explicit candidate set and returns the selected item first plus an immutable
`LoggedDecision`.

| Parameter | Operational meaning |
| --- | --- |
| `decision_id` | Stable application request ID. An exact retry returns the saved choice; reuse with different inputs raises `ValueError`. |
| `exploration` | Epsilon-uniform sampling among candidates that survive configured serving constraints. Start pilots at `0.0`. |
| `decision_metadata` | JSON-compatible application context retained immutably and included in idempotency checks. |
| `allow_unsupported_feedback` | Use only if an external system supplies the complete feedback path for an unsupported item. |
| `min_*`, `max_*`, `require_prerequisites` | Support, predicted-outcome, difficulty, and prerequisite guards applied before sampling. |

Persist the decision before rendering the item. Do not change the candidate
list, model/catalog identity, or metadata on a delivery retry.

### `observe_decision`, persistence, and recovery

```python
linked = ranker.observe_decision(
    decision.decision_id,
    outcome=correct,
    timestamp=scored_at,
    outcome_event_id=lms_score_event_id,
)
```

`observe_decision` first durably attaches the outcome, then applies it to the
live learner state. A decision accepts one immutable outcome. Repeating the
same payload is safe; a conflicting payload raises `ValueError`. Pass an
LMS-global `outcome_event_id` whenever it exists: Orchid enforces uniqueness
across decisions, not merely within one decision.

`persist_decision_outcome(...)` accepts the same outcome, reward, timestamp,
category, event ID, `apply_state`, and `update_global` arguments but does not
change state. Use it when a lifecycle adapter must persist score evidence
before projecting it into a learner model.

| Method | Use |
| --- | --- |
| `ranker.replay_pending_outcomes()` | Continue from the same restored model baseline. Applies durable, uncheckpointed outcomes in learner-time order. |
| `ranker.replay_all_outcomes_from_baseline()` | Rebuild a newly fitted/restored baseline. Do not call it on a live projection or outcomes will be applied twice. |
| `ranker.decision_log_frame(completed_only=False)` | Export decisions joined with outcomes, event IDs, and state-update flags. |

`InMemoryDecisionStore` is thread-safe but process-local.
`SQLiteDecisionStore` is transactional, uses WAL mode, and is appropriate for
a single-host service; close it explicitly or use it as a context manager.
Both implement `DecisionOutcomeStore`; a custom backend must support immutable,
idempotent decision/outcome creation plus outcome-application checkpoints.

## Catalog validation

```python
from orchid_ranker.learning_catalog import (
    LearningCatalogSchema,
    validate_learning_catalog,
)

validation = validate_learning_catalog(
    authoring_catalog,
    schema=LearningCatalogSchema(),
    require_complete_metadata=True,
)
validation.raise_for_errors()
```

`validate_learning_catalog` never mutates the DataFrame and returns all
findings rather than failing at the first one. It validates canonical exercise
identities (`item_id`, optionally `content_version`), metadata,
duplicate/conflicting versions, prerequisites, dangling references, and cycles.

| Input/result | Use |
| --- | --- |
| `LearningCatalogSchema` | Maps source columns. `item_id_col` is required; set an optional column to `None` only where it does not exist upstream. |
| `require_complete_metadata=True` | Makes absent curriculum, skill/category, difficulty, assessment, and prerequisite metadata blocking errors. Use for a pilot import gate. |
| `allow_external_prerequisites=True` | Permits prerequisite IDs owned by another catalog; otherwise dangling IDs are errors. |
| `LearningCatalogValidation.exercises` | Normalized immutable `CanonicalExercise` values. |
| `.diagnostics`, `.errors`, `.warnings` | Actionable findings, including source row indices. |
| `.is_valid`, `.raise_for_errors()` | Boolean gate or concise exception after inspection. |

This is an import-quality gate, not a serving eligibility engine. Use
`PilotCatalog` when delivery must be checked against a frozen catalog.

## Controlled pilot API

`orchid_ranker.pilot` is a reference **single-host** LMS adapter. It creates an
immutable experiment manifest, deterministic sticky assignment, append-only
delivery evidence, and analysis exports. It is not an LMS connector and does
not replace a study design, power calculation, or governance process.

### Build a frozen pilot

```python
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
pilot = AdaptivePracticePilot(
    ranker,
    catalog,
    experiment_id="networking-routing",
    model_artifact_id="empirical-2026-08-30",
    authored_policy_version="routing-static-v1",
    eligibility_rule_version="v1",
    treatment_fraction=0.5,
    randomization_salt="deployment-secret",
    assignment_store=SQLiteExperimentAssignmentStore(database),
    lifecycle_store=SQLitePilotLifecycleStore(database),
    initial_mode="aa",
)
```

`PilotCatalog.from_frame` requires one active content version per `item_id`.
It validates `content_version`, course/module, a skill or category, difficulty,
assessment flag, prerequisites, availability, required flag, and unique
authored sequence position. `eligible_items(...)` rejects—not silently
drops—an LMS candidate that is outside scope, unavailable, completed,
assessment-only, prerequisite-blocked, or inconsistent with a required item.

`AdaptivePracticePilot` requires a fitted empirical baseline without a CQL
overlay. It persists an `ExperimentManifest` with catalog digest/version, model
artifact/config identity, authored policy, eligibility version, and allocation
configuration. Reusing an experiment ID with a changed artifact or catalog
fails rather than mixing evidence.

For durable use, put `SQLiteDecisionStore`,
`SQLiteExperimentAssignmentStore`, and `SQLitePilotLifecycleStore` on the same
database path. The in-memory variants are for tests and one-process prototypes.

### Serve an LMS request

```python
served = pilot.serve(PilotRequest(
    request_id=lms_request_id,
    user_id=learner_id,
    course_id="networking",
    module_id="routing",
    course_run_id=cohort_id,
    timestamp=request_time,
    completed_item_ids=tuple(completed_ids),
    candidate_item_ids=tuple(exact_lms_eligible_ids),
    stratum=baseline_band,
))
```

`candidate_item_ids` is required and must be the exact non-null LMS-approved
set. `request_id` is namespaced by experiment and course run, preventing
cross-experiment collisions. Assignment is stable per learner and stratified
by the supplied `stratum`.

`serve` returns a `PilotDecision`:

| Field | Meaning |
| --- | --- |
| `decision` | Immutable `LoggedDecision`; render `decision.chosen_item_id`. |
| `arm` | Sticky randomized assignment (`control` or `treatment`). |
| `effective_arm` | Actually delivered arm after applying the current mode. |
| `mode` | Mode recorded on this decision. |
| `chosen_content_version` | Exact revision the LMS must render. |
| `reason_code`, `eligible_item_ids` | Why it was chosen and the exact candidates. |

Every decision automatically records an immutable explanation snapshot. Shadow
decisions also record the Orchid proposal; halted decisions record a kill-switch
fallback event.

### Modes and operation events

```python
pilot.set_mode(
    "shadow",
    event_id=change_ticket_id,
    timestamp=changed_at,
    reason="validated A/A delivery joins",
)
```

| Mode | Actual delivery |
| --- | --- |
| `"aa"` | Authored control for both assignments; assignments and lifecycle evidence are still audited. |
| `"shadow"` | Authored control; Orchid proposal is computed and recorded but not rendered. |
| `"active"` | Frozen Orchid treatment only for assigned treatment learners; assigned controls remain authored. |
| `"halted"` | All future requests use authored control with `KILL_SWITCH` evidence. A halted experiment cannot be re-enabled. |

`set_mode` is idempotent for the same event payload and records an operation
event. A transition cannot move backward in time; use a new `experiment_id` to
restart after a halt.

### Record actual delivery and score it

```python
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
outcome = pilot.record_scored(
    served.decision.decision_id,
    outcome_event_id=lms_score_event_id,
    item_id=served.decision.chosen_item_id,
    content_version=served.chosen_content_version,
    outcome=correct,
    timestamp=scored_at,
)
```

These calls are append-only and idempotent for the same event ID/payload. The
adapter enforces **decision → rendered → submitted → scored**, the selected
item/version, and nondecreasing timestamps. `record_scored` first stores its
score event, then persists/uses its outcome. It requires an LMS-global
`outcome_event_id`.

Only active effective-treatment scores update learner-local state, and those
updates leave aggregate empirical statistics frozen. Control, A/A, shadow, and
halted scores remain durable audit evidence with `apply_state=False`.

| Recovery method | Purpose |
| --- | --- |
| `pilot.recover_scored_events()` | Finish the outcome projection after a crash between score-event persistence and application. |
| `pilot.rebuild_state_from_baseline()` | Persist score evidence and rebuild treatment learner state from a freshly restored/fitted ranker baseline. Do not call on a live projection. |
| `pilot.observe_decision(...)` | Compatibility method for a score that already has a submission event. New integrations should call `record_scored`. |

### Import delayed assessments and export evidence

```python
assessments = pilot.import_delayed_assessments(assessment_frame)
analysis = pilot.analysis_frame()
```

`import_delayed_assessments` requires a pandas DataFrame with
`assessment_event_id`, `user_id`, `course_run_id`, `assessment_form_version`,
`timestamp`, `score`, and `independent`. `independent` must be literal `True`;
the method rejects an assessment for a learner/course run without pilot
participation and never feeds it into adaptive state.

Use `assessment_frame()` for imported assessments,
`decision_frame(completed_only=False)` for joined delivery decisions/outcomes,
and `analysis_frame()` for decisions plus rendered/submitted/scored IDs and
timestamps, fallback/shadow/explanation evidence, and matching assessments.

## Offline policy and evaluation APIs

The base adaptive policy is the recommended first deployment. Offline CQL is an
opt-in promotion path, not a shortcut around a controlled learning study.

```python
report = ranker.fit_policy(
    earlier_completed_decisions,
    evaluation_decisions=later_completed_decisions,
)
```

`fit_policy` trains tabular CQL from a training decision log and promotes it
only if a strictly later, disjoint evaluation log passes the
user-cluster-bootstrap rollout gate. By default it requires at least 30
evaluation events and 30 users. `fit_policy_rolling(...)` creates the same
strict chronological split from a trailing source window.

Use `ope_report(...)` or `bootstrap_ope_report(...)` for logged-policy
evaluation and `shadow_report()` to summarize traffic, outcome coverage,
calibration/drift, and, when applicable, policy evidence. These reports require
complete candidate lists, propensities, and linked outcomes; they do not prove
a causal learning benefit on their own.
