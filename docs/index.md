# Orchid Ranker

**Adaptive educational recommender toolkit** for building intelligent learning systems that grow with your students.

## Why Orchid Ranker?

Orchid Ranker is purpose-built for education. Unlike generic recommender systems, it ships with:

- **9 adaptive strategies** optimized for learning outcomes
- **Education-native metrics** (knowledge tracing, curriculum alignment)
- **Differential privacy** for student data protection
- **Agentic simulation** to test curricula at scale
- **Enterprise deployment** patterns and capacity planning

## Quick Start

Install from PyPI:

```bash
pip install orchid-ranker
```

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

recommender = OrchidRecommender(strategy="als", epochs=3)
recommender.fit(interactions, rating_col="rating")
recommendations = recommender.recommend(user_id=1, top_k=3)
```

## Next Steps

- [**Quick Start Tutorial**](quickstart.md) – 5-minute walkthrough
- [**Architecture Overview**](overview.md) – Package layout and major components
- [**Progression & Tracing**](tutorial_pc_eb.md) – Mastery tracking and forgetting curves
- [**Differential Privacy**](tutorial_dp.md) – Privacy-preserving training controls
- [**SafeSwitch**](tutorial_safe_mode.md) – Safe deployment guardrails
- [**Observability**](tutorial_observability.md) – Metrics, dashboards, and readiness
- [**Enterprise Deployment**](deployment.md) – Production considerations

## Features at a Glance

| Feature | Benefit |
|---------|---------|
| 9 strategies | Flexible, education-optimized algorithms |
| DP support | Privacy-preserving recommendations |
| Simulation | Test before deploying |
| Knowledge graphs | Model dependencies between concepts |
| Evaluation tools | Validate recommendations systematically |

---

**Get started now** with the [Quick Start Tutorial](quickstart.md).
