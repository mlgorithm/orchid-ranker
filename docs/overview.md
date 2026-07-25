# How Orchid works

Orchid exposes one adaptive recommendation loop:

```text
historical outcomes → fit → recommend → observe → recommend again
```

## One public object

`AdaptiveRanker` owns the fitted state and the live updates.

| Method | Purpose |
|--------|---------|
| `fit(events)` | Learn from chronological historical outcomes |
| `recommend(user_id, candidates)` | Rank an eligible candidate set |
| `observe(...)` | Update one user after the outcome is known |
| `recommend_and_log(...)` | Recommend and create a production decision record |
| `observe_decision(...)` | Link a delayed outcome to that decision |

The default path has no model selector. Orchid chooses and composes the
internal behavior needed for adaptive ranking.

## The data contract

The minimal event schema is:

- `user_id`
- `item_id`
- `outcome`
- `timestamp`

`outcome` is binary: `1` means the desired result happened and `0` means it did
not. It should express the objective the system is actually intended to
improve. A click is not a useful substitute for success unless clicking really
is the objective.

## Responsibilities

Your application decides which items are eligible. This includes availability,
safety, legal restrictions, enrollment, inventory, locale, age, and other hard
constraints.

Orchid decides how to order the eligible items from the user's outcome history.

This separation prevents a statistical ranker from accidentally bypassing a
hard application rule.

## Optional structure

Categories and difficulty can help when they have real domain meaning:

```python
ranker.fit(
    events,
    category_col="skill_id",
    difficulty_col="difficulty",
)
```

They are optional extensions to the same object, not different models or
products.

## Adaptation

`observe` updates the selected user's state immediately:

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

Orchid has one supported entry point: `AdaptiveRanker`.
