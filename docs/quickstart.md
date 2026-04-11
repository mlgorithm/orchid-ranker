# Quickstart Guide

This walkthrough installs Orchid Ranker from PyPI, fits a simple recommender, and runs the evaluation CLI. No external services are required.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install orchid-ranker
```

For agentic simulations or plots add extras, e.g. `pip install orchid-ranker[agentic,viz]`.

## 2. Fit & Recommend (3 steps)

```python
import pandas as pd
from orchid_ranker import OrchidRecommender

interactions = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3],
    "item_id": [10, 11, 10, 13, 12, 14],
    "label":   [1, 0, 1, 1, 0, 1],
})

rec = OrchidRecommender(strategy="als", epochs=3)
rec.fit(interactions, rating_col="label")
print(rec.recommend(user_id=1, top_k=3))
```

See `examples/quickstart.py` for a runnable script.

## 3. CLI Evaluation

```bash
orchid-evaluate \
  --train examples/data/quickstart_train.csv \
  --test examples/data/quickstart_test.csv \
  --strategy "als,epochs=3" \
  --top-k 5
```

Sample CSVs are auto-generated when you run `examples/quickstart.py`. The CLI outputs Precision@5, Recall@5, MAP@10, and NDCG@10.

## 4. Optional Extras

- Prometheus metrics: `orchid_ranker.start_metrics_server()`.
- Differential privacy: pass `dp_cfg` with engine `"opacus"` or `"per_sample"` (see `docs/privacy.md`).
- Agentic simulations: install `[agentic]` extra and see the examples in `examples/`.

## Next Steps

- Browse `docs/overview.md` for module map.
- Deployment recipes live in `docs/deployment.md`.
