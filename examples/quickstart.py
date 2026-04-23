#!/usr/bin/env python3
"""Orchid Ranker quickstart — fit once, serve forever, operate safely.

Run with: python examples/quickstart.py
"""
import numpy as np
import pandas as pd

from orchid_ranker import OrchidRecommender

# --- Generate sample interactions (replace with your own data) ---
rng = np.random.RandomState(42)
n_users, n_items, n_interactions = 12, 50, 300
interactions = pd.DataFrame({
    "user_id": rng.randint(0, n_users, n_interactions),
    "item_id": rng.randint(0, n_items, n_interactions),
})

# 1. FIT — one-shot training on historical data.
rec = OrchidRecommender.from_interactions(
    interactions,
    strategy="als",
    epochs=1,
    embedding_dim=8,
)

# 2. RECOMMEND — batch recommendations.
top5 = rec.recommend(user_id=0, top_k=5)
print(f"Top-5 for user 0: {[r.item_id for r in top5]}")

# 3. BASELINE RANK — the frozen fallback for when a guardrail halts.
fallback = rec.baseline_rank(user_id=0, top_k=5)
print(f"Baseline fallback: {[r.item_id for r in fallback]}")

# 4. EVALUATE — check model quality.
score = rec.predict(user_id=0, item_id=top5[0].item_id)
print(f"Predicted score for top recommendation: {score:.4f}")

print("\nQuickstart complete! See docs/guides/ for streaming and safety setup.")
