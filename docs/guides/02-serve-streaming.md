# Production serving

Keep one fitted `AdaptiveRanker` available to the serving process. For each
request:

1. construct the eligible candidate set;
2. recommend and persist the decision;
3. return the selected item; and
4. link the outcome when it arrives.

## Recommend and log

```python
ranked, decision = ranker.recommend_and_log(
    user_id="user-42",
    candidate_item_ids=["step-01", "step-02", "step-03"],
    timestamp=123456,
    top_k=3,
    exploration=0.05,
)
```

The chosen item is first in `ranked`. The decision is deeply immutable and
records the exact candidate set, scores, action probabilities, chosen
propensity, resolved policy, and deployment-specific policy version.

Persist the decision in an append-only store before returning the response.
Without that record, randomized traffic cannot be evaluated reliably.

## Link the outcome

```python
linked = ranker.observe_decision(
    decision.decision_id,
    outcome=1,
    timestamp=123500,
)
```

The decision ID prevents an outcome from being attached to the wrong
recommendation. Duplicate outcomes and timestamps earlier than the decision
are rejected.

If a request does not need an immutable decision record, the ordinary update
is:

```python
ranker.observe(
    user_id="user-42",
    item_id="step-02",
    outcome=1,
    timestamp=123500,
)
```

## Candidate safety

Candidate eligibility belongs outside the ranker. Apply hard rules before
calling Orchid:

```python
eligible = [
    item
    for item in catalog
    if item.available and item.allowed_for(user)
]

ranked = ranker.recommend(user.id, [item.id for item in eligible])
```

Do not expect a statistical ranking score to enforce legal, safety, inventory,
or access-control rules.

An empty eligible list is a safe no-op: `recommend(user.id, [])` returns `[]`.
It never falls back to the training catalog. Passing `None` also returns no
items unless you explicitly enable `allow_catalog_fallback=True` or attach a
candidate generator.

## Timestamps and outcomes

Use one consistent numeric time unit (for example, Unix seconds or a monotonic
event sequence). Orchid preserves fractional timestamps and rejects negative,
non-finite, or non-numeric values. Outcomes are exactly `0` or `1`; values such
as `0.9` are rejected rather than silently rounded.

## New catalog items

Register items before serving them if they were not present in fitting history:

```python
ranker.register_items(catalog)
```

`fit_semantic_items(catalog)` registers the same catalog automatically. New
items use a learned global OOV prior until the next offline refit, but they can
be recommended, logged, observed, and included in that refit. An item returned
by a separately attached semantic encoder is not automatically registered.
Such a recommendation has `feedback_supported=False` and
`recommend_and_log()` rejects it by default. Register the item before serving
it, or use `allow_unsupported_feedback=True` only when an external system owns
the complete feedback path.

## Exploration

Start with `exploration=0.0`. Introduce a small nonzero rate only after:

- decisions are persisted successfully;
- outcomes join back by decision ID;
- candidate sets are complete and exact;
- the fallback behavior has been reviewed; and
- monitoring can detect coverage or outcome regressions.

`min_item_support` is an explicit per-call safety floor. With exploration on,
it overrides the configured default exactly; set it to `0` only when newly
registered items are intentionally eligible for exploration.

## Offline-policy promotion

`fit_policy()` is optional. It promotes a CQL overlay only after evaluating the
same blended adaptive-base+CQL action rule that Orchid serves:

```python
ranker.fit_policy(
    earlier_completed_decisions,
    evaluation_decisions=later_completed_decisions,
)
```

The evaluation window must be strictly later, disjoint by both decision ID and
event content, and contain at least 30 events from 30 users by default. Orchid
uses a user-cluster bootstrap by default and preserves the unblended base scores
in every new decision record so the future evaluation can replay the actual
deployment rule. A successful promotion is logged as a distinct `hybrid+cql`
policy name and learned-state version.

## Monitor

```python
report = ranker.shadow_report()
print(report.to_dict())
```

Review decision volume, outcome coverage, candidate coverage, exploration,
propensities, calibration, and drift. Offline estimates can reject a weak
policy, but they do not replace a controlled live rollout.

## Operational checklist

- Preserve event order for every user.
- Make decision writes idempotent.
- Reject duplicate linked outcomes.
- Version deployments and retain a known fallback.
- Monitor missing and delayed outcomes.
- Refit on chronological windows rather than random row splits.
- Treat user identifiers and outcome histories as sensitive data.

## Concurrency

One `AdaptiveRanker` serializes its fit, recommendation, observation, decision
logging, and policy-promotion operations. This is safe for a modest serving
process and guarantees a request never sees a partially updated model. For
high-throughput systems, use one fitted ranker per worker and promote a freshly
fitted process or snapshot atomically at the application boundary.

The same `AdaptiveRanker` object is used from first fit through production
feedback; there is no separate serving model to configure.
