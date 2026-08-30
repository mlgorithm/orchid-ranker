# Orchid Ranker

Orchid is an adaptive-practice engine for learning products.

It ranks an eligible set of exercises, observes the learner's result, and
adapts what it recommends next. The public interface is one class:
`AdaptiveRanker`.

## Install

```bash
pip install orchid-ranker
```

## Complete example

```python
import pandas as pd
from orchid_ranker import AdaptiveRanker

history = pd.DataFrame({
    "user_id":   ["a", "a", "a", "b", "b", "b"],
    "item_id":   [101, 102, 201, 101, 102, 201],
    "outcome":   [1,   1,   0,   1,   0,   0],
    "timestamp": [1,   2,   3,   1,   2,   3],
})

ranker = AdaptiveRanker().fit(history)
ranked = ranker.recommend("a", [101, 102, 201], top_k=2)

ranker.observe(
    user_id="a",
    item_id=ranked[0].item_id,
    outcome=1,
    timestamp=4,
)
```

Orchid chooses the appropriate starting learner: a transparent empirical
baseline for sparse pilots and knowledge tracing only after data-support checks
pass. Users do not select from a model catalog.

```python
print(ranker.learning_readiness())
```

## Where it fits

Orchid works best when:

- learners complete multiple scored practice attempts;
- an existing curriculum supplies multiple pedagogically valid next exercises;
- the next exercise should react to prior attempts; and
- success can be measured later through retained mastery, a post-test, or a
  certification outcome.

It is not a generic content feed, product recommender, LMS, or curriculum
authoring system. Your learning product owns the content, eligibility rules,
and learner experience; Orchid supplies adaptive practice sequencing and the
decision evidence for a controlled evaluation.

## Choose your path

| If you want to… | Start here |
| --- | --- |
| Try the learner loop on historical attempts | [Quickstart](quickstart.md) |
| Check whether your data and catalog are ready | [Adaptive-practice data readiness](guides/00-adaptive-practice.md) |
| Add durable decisions and delayed outcomes to an existing product | [Production serving](guides/02-serve-streaming.md) |
| Run an evidence-oriented controlled pilot | [Pilot workflow](guides/05-pilot-workflow.md) |
| Look up a class or method | [API reference](api_reference.md) |
| Check compatibility guarantees | [API support policy](api-support-policy.md) |

Start with the [quickstart](quickstart.md), then read
[how Orchid works](overview.md), the small [API reference](api_reference.md),
[adaptive-practice data readiness](guides/00-adaptive-practice.md), or
[how to run a learning-efficacy pilot](guides/03-learning-pilot.md). The
[pilot integration contract](guides/04-pilot-integration.md) defines the
handoff between Orchid and a learning platform. The
[end-to-end reference-pilot workflow](guides/05-pilot-workflow.md) turns that
contract into a runnable sequence. The
[product roadmap](roadmap.md) describes the path from a single-course pilot to
an evidence-backed integration. The [design-partner council](design-partner-council.md)
keeps that roadmap grounded in simulated customer review without claiming any
real-company affiliation.
