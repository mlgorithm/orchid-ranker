# Orchid Ranker Serialization Module

## Overview

The `serialization` module provides save/load functionality for OrchidRecommender and TwoTowerRecommender models. Models can be persisted to disk and restored in their fitted state.

## Features

- **OrchidRecommender**: Save strategy, user/item mappings, fitted model state, and item features
- **TwoTowerRecommender**: Save neural model weights and configuration
- **Automatic versioning**: Checkpoints include version string for forward compatibility
- **Comprehensive state preservation**: Restores models to exact fitted state
- **Logging integration**: All save/load operations logged for debugging

## API

### Module-level Functions

#### `save_model(model, path)`

Save any OrchidRecommender or TwoTowerRecommender to disk.

**Parameters:**
- `model`: OrchidRecommender or TwoTowerRecommender instance
- `path`: str or Path destination

**Raises:**
- `ValueError`: If model type unsupported or not fitted
- `RuntimeError`: If write fails

**Example:**
```python
from orchid_ranker import save_model, OrchidRecommender
import pandas as pd

# Create and fit model
rec = OrchidRecommender(strategy="als")
interactions = pd.DataFrame({
    "user_id": [1, 1, 2, 2],
    "item_id": [10, 20, 20, 30],
})
rec.fit(interactions)

# Save to disk
save_model(rec, "checkpoints/model.pt")
```

#### `load_model(path)`

Load a previously saved model.

**Parameters:**
- `path`: str or Path to checkpoint file

**Returns:**
- Restored OrchidRecommender or TwoTowerRecommender in fitted state

**Raises:**
- `FileNotFoundError`: If checkpoint not found
- `RuntimeError`: If load/restoration fails

**Example:**
```python
from orchid_ranker import load_model

rec = load_model("checkpoints/model.pt")
predictions = rec.predict(user_id=1, item_id=10)
```

### OrchidRecommender Convenience Methods

#### `save(path)` (instance method)

Delegates to `save_model()`.

**Example:**
```python
rec = OrchidRecommender(strategy="als")
rec.fit(interactions_df)
rec.save("model.pt")
```

#### `load(path)` (classmethod)

Delegates to `load_model()`.

**Example:**
```python
rec = OrchidRecommender.load("model.pt")
```

## Supported Strategies

The serialization module supports all OrchidRecommender strategies:
- `als` (Alternating Least Squares)
- `explicit_mf` (Explicit Matrix Factorization)
- `neural_mf` (Neural Matrix Factorization)
- `popularity` (Popularity-based)
- `random` (Random baseline)
- `linucb` (LinUCB contextual bandit)
- `implicit_als` (Implicit ALS)
- `implicit_bpr` (Implicit BPR)
- `user_knn` (User K-Nearest Neighbors)

## Checkpoint Format

Checkpoints are saved using `torch.save()` (pickle-based) with the structure:

```python
{
    "version": "1.0",                    # Forward compatibility
    "model_type": "OrchidRecommender",   # or "TwoTowerRecommender"
    "state": {
        # OrchidRecommender specific:
        "strategy": "als",
        "strategy_kwargs": {...},
        "device": "cpu",
        "user_map": {user_id: idx, ...},     # User ID to index mapping
        "item_map": {item_id: idx, ...},     # Item ID to index mapping
        "seen_items": {user_idx: {...}, ...}, # Items seen by each user
        "baseline_type": "ALSBaseline",
        "baseline_state_dict": {...},   # For neural models with state_dict
        # or
        "baseline_object": baseline,    # For non-neural models
        "item_features": np.ndarray,    # Only for linucb strategy
    }
}
```

## Internal State Preserved

For **OrchidRecommender**:
- Strategy name and configuration (strategy_kwargs)
- User/item bidirectional mappings (_user2idx, _idx2user, _item2idx, _idx2item)
- Per-user seen items for filtering (_seen_items)
- Item features (for linucb)
- Fitted baseline model:
  - For neural models: PyTorch state_dict
  - For non-neural models: Full baseline object

For **TwoTowerRecommender**:
- Model architecture parameters (num_users, num_items, hidden, emb_dim, state_dim, etc.)
- Neural network weights (state_dict)
- Device placement

## Usage Examples

### Save and Load OrchidRecommender

```python
from orchid_ranker import OrchidRecommender, save_model, load_model
import pandas as pd

# Create synthetic interactions
interactions = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3],
    "item_id": [10, 20, 20, 30, 10, 30],
    "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 2.0],
})

# Train model
rec = OrchidRecommender(strategy="explicit_mf", factors=32)
rec.fit(interactions, rating_col="rating")

# Save
rec.save("my_model.pt")
# or save_model(rec, "my_model.pt")

# Later, load and use
loaded_rec = OrchidRecommender.load("my_model.pt")
# or loaded_rec = load_model("my_model.pt")

predictions = loaded_rec.predict(user_id=1, item_id=10)
recommendations = loaded_rec.recommend(user_id=1, top_k=5)
```

### Save and Load TwoTowerRecommender

```python
from orchid_ranker.agents.recommender_agent import TwoTowerRecommender
from orchid_ranker import save_model, load_model

# Create and train model
model = TwoTowerRecommender(
    num_users=100,
    num_items=50,
    user_dim=20,
    item_dim=20,
)

# ... training code ...

# Save
save_model(model, "two_tower.pt")

# Later, load
loaded_model = load_model("two_tower.pt")
```

### Error Handling

```python
from orchid_ranker import load_model

try:
    rec = load_model("non_existent.pt")
except FileNotFoundError as e:
    print(f"Checkpoint not found: {e}")

try:
    rec = OrchidRecommender(strategy="als")
    rec.save("model.pt")  # Not fitted yet
except RuntimeError as e:
    print(f"Cannot save unfitted model: {e}")
```

## Forward Compatibility

Checkpoints include a version string (`"1.0"`) for future compatibility. If a newer version of the library loads an older checkpoint, it may warn about version mismatch but will attempt to load.

## Notes

- Device placement is preserved (CPU/CUDA)
- All mappings are preserved exactly, enabling exact reproduction of predictions
- Seen items are preserved for correct filtering in `recommend(filter_seen=True)`
- For linucb strategy, item features are saved with the model
