# Run the networking/routing sample study

This example exercises a single-course comparison between the authored routing
sequence and Orchid's frozen empirical ranker. All learners see the same
approved exercise bank. The script uses synthetic practice and retention
scores, so its numeric result says nothing about learning efficacy.

## Run the sample

From a repository checkout with the package installed (for example,
`python -m pip install -e .`):

```bash
python scripts/networking_routing_pilot.py /tmp/networking-routing-pilot
python scripts/analyze_pilot_retention.py \
  /tmp/networking-routing-pilot/enrollment.csv \
  /tmp/networking-routing-pilot/assessments.csv \
  --window-days 3 \
  --analysis-timestamp 1781468800 \
  --delivery-audit /tmp/networking-routing-pilot/delivery-audit.jsonl \
  --output /tmp/networking-routing-pilot/reanalyzed.json
```

The first command writes:

| File | Purpose |
| --- | --- |
| `enrollment.csv` | Frozen roster of all 40 assigned learners, including those who never request practice; `synthetic_data=True` labels the sample |
| `assessments.csv` | Independent, synthetic retained-mastery scores |
| `delivery-audit.jsonl` | One row per practice decision, with arm, actual delivery, and lifecycle evidence |
| `retention-report.json` | Arm-level statistics and a treatment-minus-control difference |

The sample calls `pilot.enroll(...)` for every learner before serving any
exercise and exports `pilot.enrollment_frame()` with the fixed due date. Use
that durable enrollment export for the pre-declared efficacy cohort in a real
study. A learner who never practices or misses the assessment still belongs
to the assigned-arm analysis. The decision-level
`pilot.analysis_frame()` remains an audit export and must not be used as the
randomized-unit denominator.

The analyzer waits until every learner's scheduled assessment window has
closed. `--analysis-timestamp` fixes the cutoff in Unix seconds so the sample
report can be reproduced later; in a real study, record the actual data-lock
cutoff. The cutoff cannot be in the future. The CLI carries
`synthetic_data=True` from the sample enrollment CSV into its JSON output.
That marker labels this example; absence of the marker does not certify that
another input is real.

## Replace the synthetic inputs

Choose one course run and lock the study protocol before assigning learners:

1. Name the course, active-phase efficacy cohort, eligible learner population,
   objective, authored control, model artifact, content version, and exact
   candidate rules. Validate A/A and shadow delivery on QA traffic or a
   pre-study cohort; do not silently mix those learners into the efficacy
   roster.
2. Freeze the randomized enrollment roster at assignment with
   `pilot.enroll(user_id, course_run_id, timestamp, stratum=...)`. Save one row
   per learner with `user_id`, `course_run_id`, `assigned_arm`, `stratum`, and
   `assessment_due_timestamp`; `pilot.enrollment_frame()` supplies the durable
   assignment and enrollment fields. The due date must come from a fixed
   course calendar set before treatment. Do not calculate it from a learner's
   completion date after seeing their assigned arm. Keep learners who never
   request practice or complete an assessment in the roster. `serve()` can
   assign on demand for compatibility, but use explicit `enroll()` in this
   study so nonparticipants have durable course-run enrollment evidence.
3. Use a held-out assessment that is never served as practice or used to fit
   the treatment model. Verify the held-out item IDs against both arms'
   practice candidate sets; a Boolean marker alone does not prove separation.
   Export one row per completed assessment with `assessment_event_id`, `user_id`,
   `course_run_id`, `assessment_form_version`, Unix `timestamp`, normalized
   `score` in `[0, 1]`, and `independent=True`. Keep the form version fixed;
   represent a missing assessment by no row. Import the complete outcome
   export with `pilot.import_delayed_assessments(...)`, including assessed
   learners who never practiced. `pilot.assessment_frame()` can then export
   their durable assessment events.
4. Pre-register the assessment window, sample-size calculation, target effect,
   missing-data handling, and safety stop rules with the learning partner.
   This example uses a ±3-day window around the scheduled due date. Pass the
   chosen half-width with `--window-days` and run after the last due date plus
   that half-width. Its primary composite assigns zero to a missing or late
   assessment. Preserve the full raw assessment export for review.
5. Run the analysis on the frozen roster and inspect the delivery audit before
   interpreting the estimate. Check assigned versus actually delivered arms,
   missing render/submit/score records, fallbacks, and assessment coverage.
   Export the audit with
   `pilot.analysis_frame().to_json("delivery-audit.jsonl", orient="records", lines=True)`;
   pass that file with `--delivery-audit`. Build the roster CSV from the
   declared efficacy cohort in `pilot.enrollment_frame()` and the fixed course
   calendar, and use the complete independent assessment export. Both must
   include the applicable learners with no practice decisions.

The report gives each arm's randomized count, assessment rate, primary
composite mean, and mean score among observed assessments. It also gives the
unadjusted assigned-arm mean difference, a descriptive 95% bootstrap interval
for the composite when each arm has enough observations to resample, and
worst-case bounds obtained by allowing every missing score to be anywhere in
`[0, 1]`. These bounds show how much the retained-mastery conclusion could
move when assessments are missing. The zero-for-missing composite measures
assessment participation as well as mastery; interpret it accordingly. The
point estimate and bootstrap interval do not adjust for the enrollment strata.

When the delivery audit is supplied, the analyzer rejects assignment or
delivery-mode contradictions and reports missing render, submission, and score
events. The audit is a delivery check; the randomized roster still determines
the denominator.

Do not infer a learning effect from this sample, from a decision-level export,
or from the bootstrap interval alone. A real pilot needs its pre-specified
power and success criteria, independent assessment design, operational audit,
and review of missing outcomes. The [reference pilot workflow](05-pilot-workflow.md)
has the serving and recovery details.
