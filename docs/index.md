# Orchid Ranker

Orchid is an outcome-driven adaptive recommender.

It ranks an eligible candidate set, observes the result, and adapts what it
recommends next. The public interface is one class: `AdaptiveRanker`.

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

Orchid chooses its internal adaptive policy. Users do not select from a model
catalog.

## Where it fits

Orchid works when:

- interactions happen in a sequence;
- each completed interaction produces a meaningful outcome;
- the next recommendation should react to prior outcomes; and
- the application can provide an eligible candidate set.

Adaptive practice is one use case, not the product boundary. The same loop can
support onboarding, training, coaching, games, and other outcome-bearing
sequences.

Start with the [quickstart](quickstart.md), then read
[how Orchid works](overview.md), the small [API reference](api_reference.md),
or [how to validate a rollout](benchmarks/credibility.md).
