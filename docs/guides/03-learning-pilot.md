# Run a learning-efficacy pilot

A successful replay, fixture, or KT benchmark does not show that Orchid
improves learning. Make a learning claim only after a controlled experiment on
one defined learning objective.

## Choose one narrow first use case

The best first partner has a scored exercise bank, a stable skill map, multiple
valid next exercises, and enough repeat learners to run an experiment. Good
examples are technical certification practice, test preparation, coding
education, language practice, or professional-skills courses.

Avoid a first pilot where the sequence is mandatory, outcomes are only clicks,
learners make one or two attempts, or the recommendation controls a high-stakes
access decision.

## Design the experiment before adapting traffic

Pre-register these decisions with the learning team:

| Decision | Recommended choice |
| --- | --- |
| Population | One course, cohort, and learner objective |
| Control | Existing authored/static practice path |
| Treatment | Same content and eligibility rules; only next-exercise order changes |
| Primary outcome | Delayed assessment or retained mastery on items not served by Orchid |
| Secondary outcomes | Time/items to mastery, completion, repeat failures, dropout, learner feedback |
| Randomization | Learner-level, stratified by baseline assessment when available |
| Stop rules | Worse delayed mastery, unacceptable dropout, repeated failure, or outcome-linkage regressions |

Calculate sample size and a minimum detectable effect from the partner's own
baseline variance and expected attrition. Orchid’s basic readiness checks and
the default 30-user policy gate are not a substitute for a powered learning
study.

## Stage the rollout

1. Run an A/A logging check. Verify the candidate list, selected and actually
   rendered exercise, content version, model version, submission, score, and
   delayed outcome can be joined for every learner.
2. Shadow Orchid beside the authored path. Inspect its logged proposal,
   challenge probabilities,
   support, prerequisite decisions, and learner-designer explanations.
3. Randomize treatment and control. Start the reference pilot with its fixed
   empirical path (`kt_backbone="empirical"`); do not make CQL, delayed-gain
   policy learning, or exploration the pilot intervention.
4. Retain the static control for the whole experiment. Do not replace it after
   observing early favorable results.
5. Introduce small exploration only after logging is trustworthy, and only
   among exercises already approved by curriculum rules. Its purpose is later
   policy evaluation, not the primary causal estimate.

## Persist decision evidence

Use the decision loop for treatment decisions and persist the returned record
before showing the exercise. Give each request a stable application-generated
decision ID so a timeout/retry does not change the selected item:

```python
ranked, decision = ranker.recommend_and_log(
    user_id=learner.id,
    candidate_item_ids=eligible_exercise_ids,
    timestamp=attempt_time,
    exploration=0.0,
    policy_version="certification-practice-v1",
    decision_id=request_id,
)

show_exercise(ranked[0].item_id)

# Later, when the exercise is scored:
ranker.observe_decision(
    decision.decision_id,
    outcome=correct,
    timestamp=scored_at,
    outcome_event_id=lms_score_event_id,
)
```

`SQLiteDecisionStore` plus `SQLitePilotLifecycleStore` are sufficient for a
single-host prototype, but the learning product must also persist experiment
assignment, mode transitions, model artifact ID, catalog/eligibility versions,
the exact item version actually rendered and submitted, and a globally unique
outcome-event ID. Log authored-control selections in that same application
schema. Keep the independent delayed assessment in a separate stream and do
not feed it into the live learner state before the primary analysis. See the
[pilot integration contract](04-pilot-integration.md).

## Interpret results carefully

Use a delayed independent outcome for the treatment-vs-control estimate. The
[logged-policy validation guide](../benchmarks/credibility.md) is an additional
support check for a future adaptive policy version; it does not establish a
causal benefit outside the observed population, candidate construction, and
time window.

A positive, adequately powered result supports the narrowly tested curriculum
and learner population. Package that integration only after it can be replayed,
monitored, and compared against the same authored control in the next course.
