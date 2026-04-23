# Orchid Ranker

**Progression-aware recommendation toolkit** for products where users move
through structured content, build competence, or deepen product adoption over
time.

## Why Orchid Ranker?

Orchid Ranker is purpose-built for long-term user value in progression-native
domains. Unlike generic recommender systems, it ships with:

- **A single high-level API** for batch recommenders: `OrchidRecommender`
- **Progression-native metrics** for competence, category coverage, and stretch fit
- **Streaming adaptation** through `as_streaming()` for neural recommenders
- **Safety guardrails** that can fall back to a frozen baseline
- **Privacy and audit hooks** for regulated deployments

## Best-fit use cases

Orchid is strongest when users move through a meaningful progression and the
system should adapt to that path safely. Common fits include:

- workforce learning and certification
- product onboarding and activation
- rehabilitation and coaching
- structured educational content

These are better fits than generic discovery or pure CTR ranking.

## Quick Start

Install the ML extra from PyPI for the recommender API:

```bash
pip install "orchid-ranker[ml]"
```

The base `orchid-ranker` package remains available for torch-free progression
utilities such as metrics, tracing, and dependency graphs.

Create your first recommender:

```python
import pandas as pd
from orchid_ranker import OrchidRecommender

interactions = pd.DataFrame(
    {
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 11, 10, 12],
        "rating": [1.0, 0.0, 1.0, 1.0],
    }
)

recommender = OrchidRecommender.from_interactions(
    interactions,
    strategy="als",
    rating_col="rating",
    epochs=3,
)
recommendations = recommender.recommend(user_id=1, top_k=3)
```

Restrict ranking to a candidate pool when your service already has eligible
items:

```python
recommendations = recommender.recommend(
    user_id=1,
    top_k=3,
    candidate_item_ids=[10, 12, 13],
)
```

## Next Steps

- [**Quickstart**](quickstart.md) - install, fit, recommend, evaluate
- [**Guide 1: Fit offline**](guides/01-fit-offline.md) - batch recommender workflow
- [**Guide 2: Serve streaming**](guides/02-serve-streaming.md) - live adaptation
- [**Guide 3: Operate safely**](guides/03-operate-safely.md) - guardrails and fallback
- [**Usage scenarios**](scenarios.md) - practical recipes for common deployments
- [**Why Orchid**](why-orchid.md) - product thesis and evidence
- [**Competitor comparison**](competitors.md) - when to use Orchid vs other stacks

## Features at a Glance

| Feature | Benefit |
|---------|---------|
| 10 strategies | Flexible batch ranking behind one API |
| Candidate-pool ranking | Rank the eligible items your service passes in |
| Streaming bridge | Adapt per user without retraining the batch model |
| Progression metrics | Evaluate growth, coverage, sequence adherence, and stretch fit |
| Safety fallback | Serve a frozen baseline when adaptive metrics degrade |

---

**Get started now** with the [Quickstart](quickstart.md).
