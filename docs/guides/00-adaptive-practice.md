# Adaptive-practice data readiness

Use Orchid first for a bounded learning objective: one course or certification
unit, an existing bank of scored exercises, and a stable set of curriculum
rules. Orchid is a practice-sequencing layer, not an LMS, authoring tool, or
generic content recommender.

## Start with the learning contract

Every historical row is one completed attempt:

| Field | Learning meaning |
| --- | --- |
| `user_id` | Stable learner identifier |
| `item_id` | Stable exercise identifier and content version |
| `outcome` | `1` for correct/completed, `0` for not yet correct |
| `timestamp` | Attempt order in one consistent numeric time unit |

Do not turn unjoined, abandoned, or still-pending attempts into failures. Keep
them out of training until the result is known, and monitor the join rate in
production.

Add real curriculum metadata whenever it exists:

- a stable skill/category for each exercise;
- author-reviewed difficulty, rather than a success-rate proxy;
- prerequisite concepts and the authored candidate filter;
- exercise/content version and an assessment-only flag; and
- an independent delayed outcome, such as a retention quiz or certification
  result.

Current Orchid accepts one category per exercise. Do not expand multi-skill
items into repeated outcome rows: that can leak one answer into several skill
records. Keep the source item event intact until multi-skill support is
explicitly modeled.

## Fit with an exercise catalog

The preferred integration keeps author metadata separate from learner attempts.
Pass one canonical catalog row per exercise; Orchid validates that every
historical exercise appears in it and registers catalog-only exercises for
future serving.

```python
catalog = pd.DataFrame({
    "item_id": ["q-01", "q-02", "q-03"],
    "category_id": ["networking", "networking", "security"],
    "difficulty": [0.30, 0.55, 0.70],
})

ranker = AdaptiveRanker().fit(
    attempts,
    catalog=catalog,
    prerequisite_by_concept={"security": ["networking"]},
)
```

The standard catalog column names are `item_id`, `category_id`, and
`difficulty`; customize them with `catalog_item_col`, `catalog_category_col`,
and `catalog_difficulty_col`. Event-level category and difficulty columns still
work, but catalog metadata is the more durable learning-product contract.

### Validate the authoring catalog before fitting

For a pilot, validate the richer authoring contract before extracting the three
columns used by `fit`. The validator reports every issue at once: duplicate or
conflicting item versions, missing metadata, dangling prerequisites, and
prerequisite cycles.

```python
from orchid_ranker.learning_catalog import validate_learning_catalog

validation = validate_learning_catalog(
    authoring_catalog,
    require_complete_metadata=True,
)
validation.raise_for_errors()
```

Use `content_version`, `course_id`, `module_id`, a skill or category,
`difficulty`, `assessment_only`, and `prerequisites` in the authoring source.
An exercise marked `assessment_only` must be removed by the application's
eligibility query; Orchid ranks only the IDs it receives. The validator is an
import-quality gate, not a replacement for that serving-time rule.

## Inspect the fitted readiness report

```python
from orchid_ranker import AdaptiveRanker

ranker = AdaptiveRanker().fit(
    attempts,
    category_col="skill_id",
    difficulty_col="difficulty",
)

readiness = ranker.learning_readiness()
print(readiness["active_tracer"])
print(readiness["reasons"])
```

The report includes event, learner, exercise, and sequence support; outcome
balance; the presence of skill and difficulty metadata; and concrete next
steps. The default checks are configurable starting points, not universal
sample-size claims.

When support is sparse, Orchid automatically uses the **empirical** learner.
It combines global, exercise, learner, and learner-exercise outcomes with
smoothing and still adapts immediately after each attempt. This is the right
pilot behavior: transparent and useful without pretending that a deep
knowledge tracer has learned a reliable learner representation.

When basic support is adequate, Orchid evaluates the configured knowledge
tracer. That still does not prove it should be served. Compare it with an
authored sequence and the empirical baseline on a strictly chronological
holdout, using calibration (Brier score/ECE) as well as ranking metrics.

## Build eligible exercise sets outside Orchid

Before each recommendation, your application should exclude exercises that are
unavailable, already complete, inappropriate for the learner, blocked by an
authored prerequisite, or reserved for assessment.

```python
eligible_exercise_ids = [
    exercise.id
    for exercise in course.exercises
    if exercise.is_available_to(learner)
    and not learner.has_mastered(exercise)
    and course.prerequisites_met(learner, exercise)
    and not exercise.assessment_only
]

ranked = ranker.recommend(learner.id, eligible_exercise_ids, top_k=3)
```

Orchid orders this exact list; it never invents curriculum eligibility. This
makes the learning designer's rules testable and preserves a safe control path.

## Do not treat immediate correctness as learning impact

The practice outcome helps choose a next exercise. The pilot’s primary metric
should instead be delayed and independent: retained mastery, a post-test on
unserved items, certification pass rate, or time-to-mastery without lower
retention. See [learning-efficacy pilots](03-learning-pilot.md) for the
experimental design.
