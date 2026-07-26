# Quickstart

Orchid has one workflow:

1. Fit historical outcomes.
2. Recommend from eligible candidates.
3. Observe the result.

## Install

```bash
python -m pip install orchid-ranker
```

## Prepare four columns

| Column | Meaning |
|--------|---------|
| `user_id` | The person, account, or agent receiving the item |
| `item_id` | The exercise, step, task, or content identifier |
| `outcome` | Binary result: `1` for success, `0` otherwise |
| `timestamp` | A non-negative time or sequence number |

```python
import pandas as pd

history = pd.DataFrame({
    "user_id":   ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
    "item_id":   [101, 102, 201, 101, 102, 201, 101, 102, 201],
    "outcome":   [1,   1,   0,   1,   0,   0,   1,   1,   1],
    "timestamp": [1,   2,   3,   1,   2,   3,   1,   2,   3],
})
```

Rows represent completed interactions with meaningful outcomes, not
impressions. Timestamps must preserve each user's event order.

## Fit, recommend, observe

```python
from orchid_ranker import AdaptiveRanker

ranker = AdaptiveRanker().fit(history)

ranked = ranker.recommend(
    user_id="a",
    candidate_item_ids=[101, 102, 201],
    top_k=2,
)

for recommendation in ranked:
    print(
        recommendation.item_id,
        recommendation.score,
        recommendation.outcome_probability,
    )

ranker.observe(
    user_id="a",
    item_id=ranked[0].item_id,
    outcome=1,
    timestamp=4,
)
```

The new outcome updates that user immediately. Call `recommend` again to get
the adapted ranking.

Your application must construct the candidate set using its hard constraints.
Pass only items that are available, safe, licensed, or otherwise eligible.

## Optional information

If you already have a meaningful category or difficulty column, identify it
while fitting:

```python
ranker.fit(
    history,
    category_col="skill_id",
    difficulty_col="difficulty",
)
```

These fields are optional. Do not invent them merely to use Orchid.

## Production logging

When you need an immutable decision record:

```python
ranked, decision = ranker.recommend_and_log(
    user_id="a",
    candidate_item_ids=[101, 102, 201],
    timestamp=5,
    exploration=0.05,
)

ranker.observe_decision(
    decision.decision_id,
    outcome=1,
    timestamp=6,
)
```

Persist the decision before returning the recommendation. See
[Production serving](guides/02-serve-streaming.md) for the operational
contract.

## Common issues

- No recommendations: an empty eligible list returns `[]`; make sure the list
  contains fitted or registered catalog items.
- An outcome is rejected: values must be exactly binary `0` or `1`.
- Results do not adapt: call `observe` after completed interactions.
- Timestamp errors: use finite, non-negative numeric values in chronological order.
