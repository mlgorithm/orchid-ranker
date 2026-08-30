# API

The supported package-root API is:

```python
from orchid_ranker import AdaptiveRanker
```

## `AdaptiveRanker()`

Create an unfitted adaptive-practice recommender.

```python
ranker = AdaptiveRanker()
```

The defaults select the internal adaptive policy. Sparse pilots automatically
start with an empirical learner; most applications should not pass model or
training options.

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
    catalog=None,
    prerequisite_by_concept=None,
)
```

Fits the ranker and returns the same object. For learning products, rows should
be completed practice attempts and `outcome` should be correctness/completion,
not a click.

By default, Orchid expects `user_id`, `item_id`, `outcome`, and `timestamp`.
Pass the column arguments only when your source table uses different names.

For a learning product, prefer a separate catalog with canonical `item_id`,
`category_id`, and `difficulty` fields. Orchid validates historical coverage,
uses catalog metadata for the fit, and registers catalog-only exercises for
future recommendations. Pass an authored prerequisite map through
`prerequisite_by_concept`. See [Adaptive-practice data readiness](guides/00-adaptive-practice.md).

## `learning_readiness`

```python
report = ranker.learning_readiness()
```

Returns the data-support assessment selected during `fit`. It includes learner,
exercise, and sequence support; outcome balance; skill/difficulty metadata
coverage; whether basic knowledge-tracing checks passed; the active tracer; and
recommended next steps. The thresholds are configurable starting checks, not a
guarantee of model quality. See [Adaptive-practice data readiness](guides/00-adaptive-practice.md).

## `recommend`

```python
recommendations = ranker.recommend(
    user_id,
    candidate_item_ids,
    *,
    top_k=10,
)
```

Returns ranked exercise recommendation objects. The fields intended for ordinary
application use are:

| Field | Meaning |
|-------|---------|
| `item_id` | Recommended item identifier |
| `score` | Relative adaptive ranking score |
| `outcome_probability` | Estimated probability of a positive outcome |

Scores are meaningful for ordering candidates from the same request; do not
interpret them as globally calibrated learning values. The probability predicts
the next practice outcome; it is not a retained-mastery estimate.

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

Updates the learner's state from one completed attempt. Outcomes must be
exactly binary `0` or `1`; fractional values are rejected. Timestamps are
finite, non-negative numeric values in one application-defined unit.

## `register_items`

```python
ranker.register_items(catalog)
```

Registers catalog items that were absent from fitting history. Registered items
can be served and observed immediately with a learned global OOV prior; refit
to learn item-specific parameters from their accumulated outcomes.

## `recommend_and_log`

```python
recommendations, decision = ranker.recommend_and_log(
    user_id,
    candidate_item_ids,
    timestamp=timestamp,
    top_k=10,
    exploration=0.0,
    decision_id=request_id,
)
```

Performs a recommendation and creates an immutable decision record containing
the candidate set, chosen item, scores, probabilities, propensity, policy
version, and context needed for later evaluation. The record also retains the
base adaptive scores so a future CQL promotion can evaluate the exact deployed
blend. The default policy version is derived from the fitted model's learned
state and deployed overlay.

When exploration is nonzero, persist this record before returning the
recommendation.

Pass an application-generated `decision_id` to make a delivery retry
idempotent. Retrying the same request returns the original decision and action;
using the ID for different request inputs raises `ValueError`. Construct the
ranker with a `decision_store` (for example `SQLiteDecisionStore`) when
decision/outcome records must survive a process restart. The store does not
persist fitted model or learner state.

Use optional `decision_metadata` for a JSON-compatible application context that
must travel with the immutable decision, such as a catalog version, experiment
arm, or externally managed model-artifact ID. It is included in idempotency
checking and cannot be replaced by a retry.

Items without local feedback support are rejected by default. Use
`allow_unsupported_feedback=True` only if an external system is responsible for
the entire feedback path.

## `fit_policy`

```python
ranker.fit_policy(
    earlier_completed_decisions,
    evaluation_decisions=later_completed_decisions,
)
```

Optionally fit and promote a conservative CQL overlay. Promotion requires a
strictly future, duplicate-resistant holdout with at least 30 events and 30
users by default, plus user-cluster-bootstrap rollout evidence. A passing
candidate is served and evaluated as the exact adaptive-base+CQL blend, not as
standalone CQL.

## `observe_decision`

```python
linked_outcome = ranker.observe_decision(
    decision_id,
    outcome=outcome,
    timestamp=timestamp,
    outcome_event_id=score_event_id,
)
```

Links a delayed outcome to an earlier decision and updates the live user state.
A decision accepts only one linked outcome. A retry with the same outcome
payload returns that outcome; a conflicting second outcome raises `ValueError`.
Pass the LMS's unique `outcome_event_id` when available; it is retained with the
immutable linked outcome.

## `is_fitted`

```python
ranker.is_fitted
```

Returns `True` after `fit` succeeds.
