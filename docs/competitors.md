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
| You need a quick Python recommender with a simple fit/recommend API | Orchid `OrchidRecommender` or `implicit` |
| You need progression metrics, live adaptation, and safe fallback | Orchid |
| You need dozens of research algorithms and standard benchmark protocols | RecBole |
| You need GPU-scale feature engineering, training, and Triton serving | NVIDIA Merlin |
| You need hybrid matrix factorization with user/item metadata | LightFM |
| You want to build custom Keras retrieval/ranking models | TensorFlow Recommenders |
| You want notebooks and examples for many recommendation approaches | Microsoft Recommenders |
| You want a standalone recommender service with REST APIs and a dashboard | Gorse |

## What Orchid is optimized for

Orchid has three opinionated capabilities:

1. **Progression-aware ranking.** It can evaluate whether users are moving
   through useful content, categories, or sophistication levels instead of only
   clicking the next item.
2. **Streaming adaptation.** A fitted neural recommender can be promoted with
   `as_streaming()` so recent outcomes affect the next rank call.
3. **Operational safety.** Progression monitors, guardrails, and frozen
   baseline fallback make adaptive behavior easier to review and operate.

The core user journey is:

```python
from orchid_ranker import OrchidRecommender

rec = OrchidRecommender.from_interactions(df, strategy="als", rating_col="rating")
batch_recs = rec.recommend(user_id=42, top_k=10, candidate_item_ids=candidates)

adaptive = OrchidRecommender.from_interactions(
    df,
    strategy="neural_mf",
    rating_col="rating",
).as_streaming()

adaptive.observe(user_id=42, item_id=7, correct=True, category="onboarding")
live_recs = adaptive.rank(user_id=42, candidate_item_ids=candidates, top_k=10)
```

## How Orchid differs

| Project | Public positioning | How Orchid differs |
|---------|--------------------|--------------------|
| [RecBole](https://recbole.io/docs/) | PyTorch framework for reproducing and developing recommendation models, with broad algorithm and dataset coverage. | Orchid has fewer algorithms, but gives a product-oriented progression loop: fit, adapt, monitor, and fall back safely. |
| [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin/nvtabular) | GPU-accelerated recommender pipeline components for feature engineering, training, and production inference. | Orchid is not a GPU infrastructure stack. It is a Python runtime for progression-aware ranking and adaptive safety. |
| [implicit](https://github.com/benfred/implicit) | Fast collaborative filtering for implicit feedback datasets. | Orchid includes simple recommender strategies, but adds candidate-pool ranking, streaming adaptation, progression metrics, and safety patterns. |
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

> Orchid Ranker is a progression-aware recommender library for products where
> user outcomes matter more than the next click. It combines a simple
> fit/recommend API with streaming adaptation, progression metrics, and safe
> fallback patterns.

That claim is clearer, narrower, and easier to prove than saying Orchid is a
general replacement for RecBole, Merlin, implicit, LightFM, TensorFlow
Recommenders, Microsoft Recommenders, or Gorse.

For implementation recipes after choosing Orchid, see [Usage scenarios](scenarios.md).
