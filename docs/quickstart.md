# Quickstart Guide

This walkthrough installs Orchid Ranker from PyPI and fits the
adaptive-learning recommender path: learner state, difficulty-aware ranking,
prerequisites, and live re-ranking after an outcome.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install 'orchid-ranker[ml]'
```

The base `pip install orchid-ranker` package is for torch-free progression
utilities. `AdaptiveLearningRecommender` uses PyTorch-backed tracing, so install
the `ml` extra for the primary workflow.

## 2. Fit, Rank, Observe

```python
import pandas as pd
from orchid_ranker import AdaptiveLearningRecommender

events = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3],
    "item_id": [101, 201, 101, 202, 101, 201],
    "correct": [1, 0, 1, 1, 0, 1],
    "concept": ["number-sense", "fractions", "number-sense", "fractions", "number-sense", "fractions"],
    "difficulty": [0.20, 0.45, 0.20, 0.50, 0.20, 0.45],
})

rec = AdaptiveLearningRecommender(policy="auto", epochs=1).fit(
    events,
    correct_col="correct",
    concept_col="concept",
    item_difficulty_col="difficulty",
    prerequisite_by_concept={"fractions": ["number-sense"]},
)

ranked = rec.rank(user_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
rec.observe(user_id=1, item_id=ranked[0].item_id, correct=True)
print(ranked)
```

See `examples/adaptive_learning_quickstart.py` for the full adaptive-learning
script and `examples/quickstart.py` for the generic batch recommender fallback.
See `examples/scenario_selection.py` when you want Orchid to choose a workflow
from product and data signals. See `examples/knowledge_tracing_quickstart.py`
for predicted-correctness ranking from learner sequences,
`examples/akt_quickstart.py` for difficulty-aware tracing, and
`examples/kt_policy_quickstart.py` for KT-guided next-item ranking.

## 3. Batch fallback

Use `OrchidRecommender.from_interactions(...)` when you only have ordinary
user-item interactions and no learning concepts, difficulty, or prerequisites.

## 4. CLI Evaluation

```bash
python examples/quickstart.py

orchid-evaluate \
  --train examples/data/quickstart_train.csv \
  --test examples/data/quickstart_test.csv \
  --strategy "als,epochs=3" \
  --top-k 5
```

Sample CSVs are auto-generated when you run `examples/quickstart.py`. The CLI outputs Precision@5, Recall@5, MAP@10, and NDCG@10.

## 5. Optional Extras

- Prometheus metrics: `orchid_ranker.start_metrics_server()`.
- Differential privacy: pass `dp_cfg` with engine `"opacus"` or `"per_sample"` (see `docs/privacy.md`).
- Agentic simulations: install `[agentic]` extra and see the examples in `examples/`.

## Next Steps

- Read `docs/adaptive-learning-positioning.md` to understand the business fit.
- Browse `docs/overview.md` for module map.
- Browse `docs/scenarios.md` for practical deployment recipes.
- Browse `docs/algorithm-roadmap.md` for planned KT and policy-learning algorithms.
- Use `docs/guides/01-fit-offline.md` for batch usage.
- Use `docs/guides/02-serve-streaming.md` when you need live adaptation.
