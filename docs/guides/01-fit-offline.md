# Guide 1: Fit a batch recommender

Orchid Ranker ships a single entry point -- `OrchidRecommender` -- that wraps
ten ranking strategies behind one API. This guide walks through the minimal
path: load interactions, fit, recommend, predict, and save. No streaming, no
monitoring, just a working batch recommender you can query from a notebook or a
cron job.

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

rec = OrchidRecommender(strategy="als")
rec.fit(df, user_col="user_id", item_col="item_id", rating_col="rating")
```

`"als"` is a good default for implicit or sparse data. Pass `strategy="auto"`
to let Orchid pick -- it selects `explicit_mf` when it detects a range of
rating values and `als` for binary signals.

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
rec = OrchidRecommender.from_interactions(df, strategy="als",
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

Output lists all ten strategies (`als`, `auto`, `explicit_mf`, `implicit_als`,
`implicit_bpr`, `linucb`, `neural_mf`, `popularity`, `random`, `user_knn`)
with one-line descriptions.

---

You can stop here and have a working batch recommender. For real-time
adaptation, continue to [Guide 2: Serve a streaming recommender](02-serve-streaming.md).
