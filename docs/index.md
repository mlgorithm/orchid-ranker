# Orchid Ranker

**Adaptive-learning engine** for systems that choose the next best
exercise, lesson, task, or review item as learners improve, struggle, recover,
or move through structured content.

## Why Orchid Ranker?

Orchid Ranker is purpose-built for products where progress matters more than
the next click. Unlike generic recommender systems, it ships with:

- **A single high-level API** for adaptive learning: `AdaptiveLearningEngine`
- **Learner-state tracking** through AKT/SAKT-style knowledge tracing
- **Prerequisite-aware progression** through concept and difficulty metadata
- **Progression-native metrics** for competence, category coverage, and stretch fit
- **Live adaptation** through `observe()` after each learner outcome
- **Offline policy evaluation** through IPS, SNIPS, and doubly robust estimates
- **Safety guardrails** that can fall back to a reviewed learning policy
- **Privacy and audit hooks** for regulated deployments

## Quick Start

Install from PyPI:

```bash
pip install 'orchid-ranker[adaptive]'
```

Create your first adaptive-learning recommender:

```python
import pandas as pd
from orchid_ranker import AdaptiveLearningEngine

events = pd.DataFrame(
    {
        "user_id": [1, 1, 2, 2],
        "item_id": [101, 201, 101, 202],
        "correct": [1, 0, 1, 1],
        "concept": ["number-sense", "fractions", "number-sense", "fractions"],
        "difficulty": [0.20, 0.45, 0.20, 0.50],
    }
)

recommender = AdaptiveLearningEngine(policy="auto", epochs=1).fit(
    events,
    correct_col="correct",
    concept_col="concept",
    item_difficulty_col="difficulty",
    prerequisite_by_concept={"fractions": ["number-sense"]},
)
ranked = recommender.rank(user_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
recommender.observe(user_id=1, item_id=ranked[0].item_id, correct=True)
```

## Next Steps

- [**Quickstart**](quickstart.md) - install, fit, recommend, evaluate
- [**Algorithm roadmap**](algorithm-roadmap.md) - modern KT, semantic exercise recommendation, and policy-learning direction
- [**Guide 1: Fit offline**](guides/01-fit-offline.md) - adaptive-learning fit workflow
- [**Offline policy evaluation**](offline-policy-evaluation.md) - evaluate adaptive policies before rollout
- [**Guide 2: Serve adaptive recommendations**](guides/02-serve-streaming.md) - live learner-state adaptation
- [**Guide 3: Operate safely**](guides/03-operate-safely.md) - guardrails and fallback
- [**Usage scenarios**](scenarios.md) - practical recipes for common deployments
- [**Use-case examples**](examples.md) - compliance, language review, rehab, and safe rollout examples

## Features at a Glance

| Feature | Benefit |
|---------|---------|
| Learner-state tracking | Estimate competence from outcomes |
| Prerequisite graphs | Build valid next-step candidate pools |
| Scenario selector | Choose adaptive learning, safe rollout, regulated, or cold-start workflows |
| Candidate-pool ranking | Rank only the eligible learning items your service passes in |
| Progression metrics | Evaluate growth, coverage, sequence adherence, and stretch fit |
| Offline policy evaluation | Estimate adaptive-policy value from logged propensities |
| Safety fallback | Serve a reviewed learning policy when adaptive metrics degrade |

---

Start with the [quickstart](quickstart.md),
then run the
[adaptive learning quickstart](https://github.com/mlgorithm/orchid-ranker/blob/main/examples/adaptive_learning_quickstart.py).
