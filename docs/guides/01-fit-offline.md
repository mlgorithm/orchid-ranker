# Guide 1: Fit Offline

Fit Orchid as an adaptive-learning ranker: learner outcomes in, staged
learner-state and progression policy out. This is the default path for new
projects.

## Install

```bash
pip install 'orchid-ranker[adaptive]'
```

## Load learner events

At minimum, provide learner ID, item ID, correctness, and timestamp. Concept
and difficulty metadata make the policy materially more useful.

```python
import pandas as pd

events = pd.read_csv("learner_events.csv")
catalog = pd.read_csv("exercise_catalog.csv")

training = events.merge(catalog, on="item_id", how="left")
```

Expected columns:

| Column | Meaning |
|--------|---------|
| `learner_id` | Learner or user identifier |
| `item_id` | Exercise / lesson / task identifier |
| `correct` | Binary or thresholdable outcome |
| `ts` | Monotonic timestamp for chronological splits and time-aware KT |
| `concept_id` | Skill/concept label for competence and prerequisites |
| `difficulty` | Item difficulty in `[0, 1]` when available |

## Fit

```python
from orchid_ranker import AdaptiveRanker

ranker = AdaptiveRanker(
    kt_backbone="saint+",
    policy="auto",
    epochs=2,
    d_model=32,
).fit_kt(
    training,
    learner_col="learner_id",
    item_col="item_id",
    correct_col="correct",
    timestamp_col="ts",
    concept_col="concept_id",
    item_difficulty_col="difficulty",
)
```

Use `kt_backbone="sakt"` for a compact attention baseline, `"dkt"` or
`"dkvmn"` for sequence ablations, and `"akt"` / `"saint+"` when you have
difficulty or timestamp signal.

## Recommend And Observe

```python
candidates = catalog["item_id"].tolist()
ranked = ranker.recommend("learner-42", candidates, top_k=5)

ranker.observe(
    learner_id="learner-42",
    ts=123456,
    item_id=ranked[0].item_id,
    concept_id=None,
    correct=1,
)
```

The next call to `recommend(...)` uses the updated learner state.

## Add Semantic Cold Start

```python
ranker.fit_semantic_items(
    catalog,
    item_col="item_id",
    text_col="item_text",
    metadata_cols=["concept_id", "difficulty"],
)

ranked = ranker.recommend(
    "learner-42",
    top_k=5,
    item_query_text="fraction addition with unlike denominators",
)
```

Semantic candidates outside the KT training universe can still be returned with
`policy="semantic_cold_start"` when catalog metadata is available.

## Logged-Policy Learning

When you have logged candidate sets, chosen actions, propensities, and rewards:

```python
policy_report = ranker.fit_policy(logged_decisions, algo="cql")
ope = ranker.ope_report(logged_decisions)
```

Use OPE and rollout gates before serving a learned policy to live learners.

Continue to [Guide 2: Serve Adaptive Recommendations](02-serve-streaming.md)
for online serving patterns.
