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

1. Run an A/A logging check on QA traffic or a separate pre-study cohort.
   Verify that the candidate list, selected and actually rendered exercise,
   content version, model version, submission, score, and delayed outcome can
   be joined for every learner.
2. Shadow Orchid beside the authored path on pre-study traffic. Inspect its
   logged proposal, challenge probabilities, support, prerequisite decisions,
   and learner-designer explanations.
3. Define the efficacy cohort and freeze its enrollment roster before active
   exposure. Activate the reference pilot after A/A and shadow checks. It
   assigns learners once and serves its fixed empirical path
   (`kt_backbone="empirical"`) only to the treatment arm. Keep CQL,
   delayed-gain policy learning, and exploration out of this intervention.
4. Retain the static control for the whole experiment. Do not replace it after
   observing early favorable results.
5. Consider exploration in a separately specified later study, only among
   exercises already approved by curriculum rules. Its purpose is later
   policy evaluation, not the primary causal estimate here.

## Persist decision evidence

Use the reference adapter's `pilot.enroll(...)` to freeze a complete learner
roster before practice begins, including learners who later never request an
exercise. Use `pilot.serve(...)` with a stable product request ID for each
eligible request. Record the returned item and content version at render,
submission, and scoring through the adapter's lifecycle methods. The
[end-to-end workflow](05-pilot-workflow.md) shows this path with the durable
assignment, decision, and lifecycle stores.

Keep the independent delayed assessment in a separate stream and do not feed
it into the live learner state before the primary analysis. The decision-level
`pilot.analysis_frame()` is useful for delivery audits; the [one-course study
guide](06-one-course-study.md) shows the separate frozen roster needed for a
learner-level assigned-arm analysis.

## Interpret results carefully

Use a delayed independent outcome for the treatment-vs-control estimate. The
[logged-policy validation guide](../benchmarks/credibility.md) is an additional
support check for a future adaptive policy version; it does not establish a
causal benefit outside the observed population, candidate construction, and
time window.

A positive, adequately powered result supports the narrowly tested curriculum
and learner population. Package that integration only after it can be replayed,
monitored, and compared against the same authored control in the next course.
