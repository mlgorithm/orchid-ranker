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
from orchid_ranker import OrchidRecommender

recommender = OrchidRecommender(strategy="knowledge_tracing")
recommendations = recommender.recommend(student_id="s1", context={})
```

## Next Steps

- [**Installation Guide**](getting-started/installation.md) – Set up your environment
- [**Quick Start Tutorial**](getting-started/quickstart.md) – 5-minute walkthrough
- [**Knowledge Tracing**](tutorials/knowledge_tracing.md) – Model student mastery
- [**Curriculum Design**](tutorials/curriculum.md) – Build adaptive pathways
- [**Agentic Simulation**](tutorials/agentic.md) – Validate strategies at scale
- [**Differential Privacy**](tutorials/privacy.md) – Protect student data
- [**Enterprise Deployment**](tutorials/deployment.md) – Production considerations

## Features at a Glance

| Feature | Benefit |
|---------|---------|
| 9 strategies | Flexible, education-optimized algorithms |
| DP support | Privacy-preserving recommendations |
| Simulation | Test before deploying |
| Knowledge graphs | Model dependencies between concepts |
| Evaluation tools | Validate recommendations systematically |

---

**Get started now** with the [Installation Guide](getting-started/installation.md).
