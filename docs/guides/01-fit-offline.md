# Guide 1: Fit the batch fallback

`OrchidRecommender` is Orchid's batch fallback for ordinary user-item data. Use
it when you do not yet have learning concepts, difficulty, prerequisites, or
live outcome updates. For the primary adaptive-learning workflow, start with
`AdaptiveLearningEngine` in the [Quickstart](../quickstart.md). This guide
walks through the minimal batch path: load interactions, fit, recommend,
predict, and save.

## Install

```bash
pip install 'orchid-ranker[ml]'
```

## Load interactions

Your data needs three columns: a user identifier, an item identifier, and
(optionally) a numeric rating. If no rating column is provided, all
interactions are treated as implicit positive feedback.

```python
import pandas as pd

df = pd.read_csv("interactions.csv")  # user_id, item_id, rating
print(df.head())
```

## Fit

```python
from orchid_ranker import OrchidRecommender

rec = OrchidRecommender(strategy="legacy_binary_mf")
rec.fit(df, user_col="user_id", item_col="item_id", rating_col="rating")
```

`"legacy_binary_mf"` is a backward-compatible binary-MF baseline trained with BCE and
sampled missing-item negatives when `rating_col` is omitted. Pass
`strategy="auto"` to let Orchid pick -- it selects `explicit_mf` when it
detects a range of rating values and `legacy_binary_mf` for binary signals. Install
`orchid-ranker[implicit]` and use `strategy="implicit_als"` when you need a
true alternating-least-squares solver.

## Recommend and predict

```python
# Top-k recommendations for a user
recs = rec.recommend(user_id=42, top_k=10)
for r in recs:
    print(r.item_id, r.score)

# Pointwise score for a specific (user, item) pair
score = rec.predict(user_id=42, item_id=7)
```

## One-liner with `from_interactions`

```python
rec = OrchidRecommender.from_interactions(df, strategy="legacy_binary_mf",
                                          user_col="user_id",
                                          item_col="item_id",
                                          rating_col="rating")
```

## Save and reload

```python
rec.save("model.pt")

rec2 = OrchidRecommender.load("model.pt")
assert rec2.predict(user_id=42, item_id=7) == score
```

## Browse available strategies

```python
print(OrchidRecommender.available_strategies())
```

Output lists the legacy strategy set (`legacy_binary_mf`, `als`, `auto`, `explicit_mf`, `implicit_als`,
`implicit_bpr`, `linucb`, `neural_mf`, `popularity`, `random`, `user_knn`)
with one-line descriptions.

---

You can stop here and have a working fallback model. For adaptive learning
with learner state and live outcomes, use
`examples/adaptive_learning_quickstart.py`. For generic real-time adaptation
without learning metadata, continue to
[Guide 2: Serve a streaming recommender](02-serve-streaming.md).
