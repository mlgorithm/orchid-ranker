# API

The supported package-root API is:

```python
from orchid_ranker import AdaptiveRanker
```

## `AdaptiveRanker()`

Create an unfitted adaptive recommender.

```python
ranker = AdaptiveRanker()
```

The defaults select the internal adaptive policy. Most applications should not
pass model or training options.

## `fit`

```python
ranker.fit(
    events,
    *,
    user_col="user_id",
    item_col="item_id",
    outcome_col="outcome",
    timestamp_col="timestamp",
    category_col=None,
    difficulty_col=None,
)
```

Fits the ranker and returns the same object.

By default, Orchid expects `user_id`, `item_id`, `outcome`, and `timestamp`.
Pass the column arguments only when your source table uses different names.

## `recommend`

```python
recommendations = ranker.recommend(
    user_id,
    candidate_item_ids,
    *,
    top_k=10,
)
```

Returns ranked recommendation objects. The fields intended for ordinary
application use are:

| Field | Meaning |
|-------|---------|
| `item_id` | Recommended item identifier |
| `score` | Relative adaptive ranking score |
| `outcome_probability` | Estimated probability of a positive outcome |

Scores are meaningful for ordering candidates from the same request; do not
interpret them as globally calibrated business values.

`candidate_item_ids=[]` returns `[]`. It is never interpreted as “all known
items.” Omit candidates only when an explicitly configured catalog fallback or
candidate generator is intended.

## `observe`

```python
ranker.observe(
    user_id=user_id,
    item_id=item_id,
    outcome=outcome,
    timestamp=timestamp,
)
```

Updates the user's state from one completed interaction. Outcomes must be
exactly binary `0` or `1`; fractional values are rejected. Timestamps are
finite, non-negative numeric values in one application-defined unit.

## `register_items`

```python
ranker.register_items(catalog)
```

Registers catalog items that were absent from fitting history. Registered items
can be served and observed immediately with a conservative OOV representation;
refit to learn item-specific parameters from their accumulated outcomes.

## `recommend_and_log`

```python
recommendations, decision = ranker.recommend_and_log(
    user_id,
    candidate_item_ids,
    timestamp=timestamp,
    top_k=10,
    exploration=0.0,
)
```

Performs a recommendation and creates an immutable decision record containing
the candidate set, chosen item, scores, probabilities, propensity, policy
version, and context needed for later evaluation. The default policy version is
derived from the fitted model and configuration.

When exploration is nonzero, persist this record before returning the
recommendation.

## `observe_decision`

```python
linked_outcome = ranker.observe_decision(
    decision_id,
    outcome=outcome,
    timestamp=timestamp,
)
```

Links a delayed outcome to an earlier decision and updates the live user state.
A decision accepts only one linked outcome.

## `is_fitted`

```python
ranker.is_fitted
```

Returns `True` after `fit` succeeds.
