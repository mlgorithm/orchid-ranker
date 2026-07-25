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

The chosen item is first in `ranked`. The decision records the exact candidate
set, scores, action probabilities, chosen propensity, context, and policy
version.

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

## Exploration

Start with `exploration=0.0`. Introduce a small nonzero rate only after:

- decisions are persisted successfully;
- outcomes join back by decision ID;
- candidate sets are complete and exact;
- the fallback behavior has been reviewed; and
- monitoring can detect coverage or outcome regressions.

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

The same `AdaptiveRanker` object is used from first fit through production
feedback; there is no separate serving model to configure.
