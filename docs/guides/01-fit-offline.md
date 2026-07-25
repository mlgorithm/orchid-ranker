# Fit historical outcomes

This guide covers the one required preparation step: fitting
`AdaptiveRanker` from chronological outcomes.

## Install

```bash
pip install orchid-ranker
```

## Required data

```python
import pandas as pd

events = pd.DataFrame({
    "user_id":   ["a", "a", "a", "b", "b", "b"],
    "item_id":   [101, 102, 201, 101, 102, 201],
    "outcome":   [1,   1,   0,   1,   0,   0],
    "timestamp": [1,   2,   3,   1,   2,   3],
})
```

The requirements are:

- user and item identifiers must be stable;
- outcomes must be binary `0` or `1`;
- timestamps must be non-negative; and
- timestamps must preserve event order within each user.

Missing outcomes should not be silently converted to failures. Keep incomplete
interactions out of the fitting table until their outcome is known.

## Fit

```python
from orchid_ranker import AdaptiveRanker

ranker = AdaptiveRanker().fit(events)
```

Orchid chooses the internal adaptive policy. There is no model-selection step
in the supported workflow.

## Custom column names

You can identify existing columns without renaming the source table:

```python
ranker.fit(
    events,
    user_col="account",
    item_col="task",
    outcome_col="success",
    timestamp_col="event_time",
)
```

## Optional category and difficulty

```python
ranker.fit(
    events,
    category_col="skill_id",
    difficulty_col="difficulty",
)
```

Use these only when they already have a defensible domain meaning. Categories
can connect related items. Difficulty can distinguish trivial success from an
appropriate challenge. Neither is required.

## Verify the fit

```python
assert ranker.is_fitted

recommendations = ranker.recommend(
    user_id="a",
    candidate_item_ids=[101, 102, 201],
    top_k=2,
)
```

Evaluate on a chronological holdout rather than a random row split. Future
interactions from a user must not leak into the training side of an evaluation.

## Refreshing

Use `observe` for immediate per-user updates. Refit periodically when enough
new history accumulates or when monitoring shows material drift.
