# How Orchid works

Orchid is an adaptive-practice engine. It exposes one learning loop:

```text
completed practice attempts → fit → recommend → observe → recommend again
```

## Core learner loop

`AdaptiveRanker` owns the fitted state and live updates.

| Method | Purpose |
|--------|---------|
| `fit(events)` | Learn from chronological completed practice attempts |
| `learning_readiness()` | Explain whether the data supports knowledge tracing |
| `recommend(user_id, candidates)` | Rank a pedagogically eligible exercise set |
| `observe(...)` | Update one learner after the outcome is known |
| `recommend_and_log(...)` | Recommend and create a production decision record |
| `observe_decision(...)` | Link a delayed outcome to that decision |

The object serializes these operations, so a single serving instance does not
expose partially updated learner state or a half-promoted policy.

The default path has no model selector. Orchid starts sparse pilots with a
transparent empirical learner and selects knowledge tracing only after basic
data-support checks pass. These checks are a deployment aid, not a claim that a
model improves learning; validate against an authored sequence on a
chronological holdout and in a controlled pilot.

## The data contract

The minimal event schema is:

- `user_id` — learner identifier
- `item_id` — exercise identifier
- `outcome` — completed/correct (`1`) or not yet correct (`0`)
- `timestamp` — attempt order

`outcome` is a useful adaptation signal, not the product's north-star outcome.
Measure retained mastery with a delayed, independent assessment or
certification result; a click or next-attempt correctness is not evidence of
learning impact.

## Responsibilities

Your application decides which exercises are eligible. This includes
prerequisites, authored sequence rules, assessment holdouts, accommodations,
availability, enrollment, and other hard curriculum constraints.

Orchid decides how to order those eligible exercises from the learner's attempt
history. This separation prevents a statistical ranker from bypassing a hard
curriculum rule.

## Curriculum structure

Skills/categories and author-reviewed difficulty make the adaptive policy more
useful and explainable:

```python
ranker.fit(
    events,
    category_col="skill_id",
    difficulty_col="difficulty",
)
```

They are optional so a pilot can start with existing event data, but they should
be part of a real learning integration. A missing category makes each exercise
its own concept; outcome-derived difficulty is only a proxy for authored
difficulty. Orchid does not infer pedagogical prerequisites for you.

## Adaptation

`observe` updates the selected learner's state immediately:

```python
ranker.observe(
    user_id=user_id,
    item_id=item_id,
    outcome=outcome,
    timestamp=timestamp,
)
```

The next `recommend` call sees that update. Periodic offline refits can still be
used to incorporate larger batches of new history.

`AdaptiveRanker` is the package-root entry point for this learner loop. The
optional `orchid_ranker.pilot` module adds a deliberately explicit reference
adapter for a controlled study; it is not needed for a local recommendation
prototype. See [Adaptive-practice data readiness](guides/00-adaptive-practice.md)
before a pilot and [learning-efficacy pilots](guides/03-learning-pilot.md)
before making an outcome claim.
