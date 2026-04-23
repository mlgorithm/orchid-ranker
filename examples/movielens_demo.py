#!/usr/bin/env python3
"""MovieLens-style demo -- fit on synthetic ratings, get recommendations.

Run with: python examples/movielens_demo.py
Requires: pip install orchid-ranker[ml]
"""
import numpy as np
import pandas as pd

from orchid_ranker import OrchidRecommender

# --- Synthetic MovieLens-like dataset ---
rng = np.random.default_rng(42)
n_users, n_items, n_interactions = 200, 100, 5000

interactions = pd.DataFrame({
    "user_id": rng.integers(0, n_users, n_interactions),
    "item_id": rng.integers(0, n_items, n_interactions),
    "rating": rng.choice(
        [1, 2, 3, 4, 5], n_interactions, p=[0.10, 0.15, 0.25, 0.30, 0.20],
    ),
})

print(f"Dataset: {n_users} users, {n_items} items, {n_interactions} interactions")
print(f"Rating distribution:\n{interactions['rating'].value_counts().sort_index()}\n")

# --- Fit with ALS (alternating least squares) ---
rec = OrchidRecommender.from_interactions(
    interactions, strategy="als", user_col="user_id", item_col="item_id",
)
print(f"Fitted: {rec}\n")

# --- Top-5 recommendations for user 0 ---
recs = rec.recommend(user_id=0, top_k=5)
print("Top-5 recommendations for user 0:")
for r in recs:
    print(f"  item {r.item_id}: score {r.score:.4f}")

# --- Point prediction ---
score = rec.predict(user_id=0, item_id=recs[0].item_id)
print(f"\nPredicted score for top pick: {score:.4f}")

# --- Frozen baseline fallback ---
fallback = rec.baseline_rank(user_id=0, top_k=5)
print(f"Baseline fallback items: {[r.item_id for r in fallback]}")
