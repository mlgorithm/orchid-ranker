# Competitor comparison

Orchid Ranker is not trying to be the biggest recommender model zoo or the
fastest GPU retrieval stack. Its best fit is narrower: products where the user
has a trajectory and the ranking system should adapt safely as that trajectory
changes.

Use Orchid when recommendations should improve a user's long-term outcome:
learning progress, onboarding depth, rehab adherence, training completion,
fitness progression, expertise-driven shopping, or curated content discovery.

## The short version

| Use case | Best first choice |
|----------|-------------------|
| You need adaptive learning with learner state, prerequisites, progression reward, and live outcome updates | Orchid `AdaptiveLearningRecommender` |
| You need fast generic collaborative filtering on implicit feedback | `implicit` |
| You need dozens of research algorithms and standard benchmark protocols | RecBole |
| You need GPU-scale feature engineering, training, and Triton serving | NVIDIA Merlin |
| You need hybrid matrix factorization with user/item metadata | LightFM |
| You want to build custom Keras retrieval/ranking models | TensorFlow Recommenders |
| You want notebooks and examples for many recommendation approaches | Microsoft Recommenders |
| You want a standalone recommender service with REST APIs and a dashboard | Gorse |

## What Orchid is optimized for

Orchid has four opinionated capabilities:

1. **Learner-state-aware ranking.** It estimates correctness and competence
   from historical and live learner outcomes.
2. **Progression-aware ranking.** It can evaluate whether users are moving
   through useful content, categories, or sophistication levels instead of only
   clicking the next item.
3. **Live adaptation.** `observe()` lets the next recommendation change after
   a response, completion, or failure.
4. **Operational safety.** Progression monitors, guardrails, and frozen
   baseline fallback make adaptive behavior easier to review and operate.

The core user journey is:

```python
from orchid_ranker import AdaptiveLearningRecommender

rec = AdaptiveLearningRecommender(policy="auto").fit(
    outcomes,
    correct_col="correct",
    concept_col="concept",
    item_difficulty_col="difficulty",
    prerequisite_by_concept={"fractions": ["number-sense"]},
)

ranked = rec.rank(user_id=42, candidate_item_ids=candidates, top_k=5)
rec.observe(user_id=42, item_id=ranked[0].item_id, correct=True)
```

## How Orchid differs

| Project | Public positioning | How Orchid differs |
|---------|--------------------|--------------------|
| [RecBole](https://recbole.io/docs/) | PyTorch framework for reproducing and developing recommendation models, with broad algorithm and dataset coverage. | Orchid has fewer algorithms, but gives a product-oriented progression loop: fit, adapt, monitor, and fall back safely. |
| [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin/nvtabular) | GPU-accelerated recommender pipeline components for feature engineering, training, and production inference. | Orchid is not a GPU infrastructure stack. It is a Python runtime for progression-aware ranking and adaptive safety. |
| [implicit](https://github.com/benfred/implicit) | Fast collaborative filtering for implicit feedback datasets. | `implicit` is stronger for plain collaborative filtering. Orchid's reason to exist is adaptive learning: learner state, prerequisites, progression reward, live `observe()`, OPE, and guardrails. |
| [LightFM](https://making.lyst.com/lightfm/docs/) | Hybrid recommendation algorithms for implicit and explicit feedback, with user and item metadata. | Orchid's wedge is user trajectory and operational safety, not only hybrid matrix factorization. |
| [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) | Keras-based library for building recommender system models across data preparation, modeling, training, evaluation, and deployment. | Orchid sits at a higher product layer: a ready recommender API plus progression-specific serving and monitoring behavior. |
| [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) | Jupyter notebook examples and best practices for recommendation systems. | Orchid is an importable runtime with a narrow progression thesis, not primarily an examples repository. |
| [Gorse](https://gorse.io/) | Open-source recommender system engine with APIs, database integrations, dashboard, and online evaluation. | Orchid is a Python library for embedding progression-aware recommendation inside an existing product or service. |

## Where not to use Orchid

Choose another stack when:

- You need a catalog of 50+ academic models for paper reproduction.
- You need distributed GPU training and serving as the main problem.
- Your only target metric is CTR, watch time, or ad revenue.
- Your users do not have a meaningful progression path.
- You need a full recommender service with storage, APIs, and dashboard out of
  the box.

## Defensible claim

The strongest public claim is:

> Orchid Ranker is an adaptive-learning recommender stack for products where
> recommendations should make the user better, not merely more engaged. It
> combines learner-state tracing, prerequisite-aware ranking, progression
> reward, live outcome updates, OPE, and safe fallback patterns.

That claim is clearer, narrower, and easier to prove than saying Orchid is a
general replacement for RecBole, Merlin, implicit, LightFM, TensorFlow
Recommenders, Microsoft Recommenders, or Gorse.

For implementation recipes after choosing Orchid, see [Usage scenarios](scenarios.md).
