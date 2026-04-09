"""Generate a small, deterministic benchmark fixture for regression testing.

Creates a synthetic educational interaction dataset with known properties:
- 200 users, 300 items, ~8000 interactions
- Seed-locked for exact reproducibility
- Includes both implicit and explicit signals
"""
import numpy as np
import pandas as pd
import os

SEED = 42
NUM_USERS = 200
NUM_ITEMS = 300
NUM_INTERACTIONS = 8000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate():
    rng = np.random.RandomState(SEED)

    user_ids = rng.randint(0, NUM_USERS, size=NUM_INTERACTIONS)
    item_ids = rng.randint(0, NUM_ITEMS, size=NUM_INTERACTIONS)

    # Simulate latent skill alignment: users with higher IDs prefer harder items
    user_skill = user_ids / NUM_USERS  # 0..1
    item_difficulty = item_ids / NUM_ITEMS  # 0..1
    alignment = 1.0 - np.abs(user_skill - item_difficulty)

    # Generate ratings with noise
    ratings = np.clip(alignment * 5.0 + rng.normal(0, 0.8, NUM_INTERACTIONS), 1.0, 5.0)
    ratings = np.round(ratings, 1)

    # Binary labels (implicit feedback): rating >= 3.5
    labels = (ratings >= 3.5).astype(float)

    df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings,
        "label": labels,
    })

    # Deterministic train/test split (80/20 by index)
    n = len(df)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    # Also generate item features for LinUCB (random but deterministic)
    item_features = rng.randn(NUM_ITEMS, 8).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, "item_features.npy"), item_features)

    print(f"Generated fixture: {len(train_df)} train, {len(test_df)} test, "
          f"{NUM_USERS} users, {NUM_ITEMS} items")
    print(f"  Train positive rate: {train_df['label'].mean():.3f}")
    print(f"  Test positive rate:  {test_df['label'].mean():.3f}")


if __name__ == "__main__":
    generate()
