# Quickstart

Orchid has one adaptive-practice workflow:

1. Fit completed learner attempts.
2. Inspect data readiness.
3. Recommend from pedagogically eligible exercises.
4. Observe the scored result.

## Install

```bash
python -m pip install orchid-ranker
```

## Prepare four columns

| Column | Meaning |
|--------|---------|
| `user_id` | The learner receiving practice |
| `item_id` | The exercise identifier (and ideally content version) |
| `outcome` | Binary result: `1` for correct/completed, `0` for not yet correct |
| `timestamp` | A non-negative attempt time or sequence number |

```python
import pandas as pd

history = pd.DataFrame({
    "user_id":   ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
    "item_id":   [101, 102, 201, 101, 102, 201, 101, 102, 201],
    "outcome":   [1,   1,   0,   1,   0,   0,   1,   1,   1],
    "timestamp": [1,   2,   3,   1,   2,   3,   1,   2,   3],
})
```

Rows represent completed practice attempts, not impressions. Timestamps must
preserve each learner's event order.

## Fit, recommend, observe

```python
from orchid_ranker import AdaptiveRanker

ranker = AdaptiveRanker().fit(history)
readiness = ranker.learning_readiness()
print(readiness["active_tracer"])

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

The new outcome updates that learner immediately. Call `recommend` again to
get the adapted ranking. On sparse pilot data, `active_tracer` is `empirical`:
a transparent, smoothed learner/exercise baseline. Orchid moves to knowledge
tracing only after its configurable data-support checks pass.

Your application must construct the candidate set using authored curriculum
rules. Pass only exercises that are available, appropriate, prerequisite-ready,
and not reserved for assessment.

## Optional information

If you already have a real skill/category or author-reviewed difficulty column,
identify it while fitting:

```python
ranker.fit(
    history,
    category_col="skill_id",
    difficulty_col="difficulty",
)
```

These fields are optional for a pilot. Do not invent them merely to use Orchid;
an outcome-derived difficulty is a proxy, not an instructional-design fact.

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
contract. For a real learning pilot, also persist an independent delayed
assessment outcome; see [Run a learning-efficacy pilot](guides/03-learning-pilot.md).

## Common issues

- No recommendations: an empty eligible list returns `[]`; make sure the list
  contains fitted or registered catalog items.
- An outcome is rejected: values must be exactly binary `0` or `1`.
- Results do not adapt: call `observe` after completed interactions.
- Timestamp errors: use finite, non-negative numeric values in chronological order.
