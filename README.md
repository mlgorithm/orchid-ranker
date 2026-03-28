# Orchid Ranker

**Adaptive educational recommender toolkit** for personalized learning at scale.

Orchid Ranker combines a modular orchestration engine, realistic learner simulators, and a plug-and-play recommender API (similar to Surprise) into a single library purpose-built for educational technology. It grew out of large-scale tutoring experiments and now powers adaptive item selection, knowledge tracing, curriculum sequencing, and offline evaluation for learning platforms.

[![Python 3.9–3.13](https://img.shields.io/badge/python-3.9%E2%80%933.13-blue.svg)](https://www.python.org/)
[![PyTorch 1.13–2.9](https://img.shields.io/badge/pytorch-1.13%E2%80%932.9-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.1-brightgreen.svg)](CHANGELOG.md)

---

## Why Orchid Ranker?

Most recommender libraries target e-commerce or media. Orchid Ranker is designed from the ground up for **education**, where the goal isn't just relevance — it's learning. The library provides Bayesian knowledge tracing, prerequisite-aware curriculum sequencing, Zone of Proximal Development (ZPD) targeting, forgetting curve modeling, and educational evaluation metrics alongside traditional recommendation algorithms.

**Key differentiators:**

- **9 recommendation strategies** from popularity baselines to contextual bandits and neural models, all behind a unified `OrchidRecommender` API
- **Knowledge tracing** with Bayesian Knowledge Tracing (BKT), mastery tracking, and Ebbinghaus forgetting curves
- **Curriculum intelligence** via prerequisite graphs (DAG) with cycle detection, topological ordering, and ZPD-aware recommendations
- **Learner simulation** with realistic student agents modeling knowledge, fatigue, trust, and engagement dynamics
- **Educational metrics** including learning gain, knowledge coverage, curriculum adherence, and difficulty appropriateness
- **Privacy by design** with differential privacy (DP-SGD), RBAC, and audit logging
- **Model lifecycle** with serialization, cross-validation, grid/random search, and train/test splitting
- **Enterprise ready** with Prometheus observability, Snowflake/BigQuery/S3 connectors, MLflow tracking, Docker/Helm/Terraform deployment

---

## Installation

```bash
pip install orchid-ranker
```

Or install from source:

```bash
git clone https://github.com/farhad-vadiee/orchid-ranker.git
cd orchid-ranker
pip install -e .
```

Optional extras for specific use cases:

```bash
pip install orchid-ranker[agentic]       # Experiment helpers + Opacus DP
pip install orchid-ranker[viz]           # Matplotlib plotting
pip install orchid-ranker[connectors]    # Snowflake, BigQuery, S3, MLflow
pip install orchid-ranker[observability] # Prometheus metrics
pip install orchid-ranker[benchmarks]    # Competitor baselines (implicit, reclab)
pip install orchid-ranker[dev]           # pytest, mypy, ruff, build tools
```

---

## Quick Start

### 1. Fit and recommend in 5 lines

```python
import pandas as pd
from orchid_ranker import OrchidRecommender

interactions = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3, 3],
    "item_id": [10, 12, 10, 11, 12, 13, 14],
    "rating":  [1,  1,  1,  0,  1,  0,  1],
})

rec = OrchidRecommender(strategy="als", epochs=5)
rec.fit(interactions, rating_col="rating")
print(rec.recommend(user_id=1, top_k=5))
```

### 2. Compare strategies side by side

```python
from orchid_ranker import compare_models

results = compare_models(
    interactions,
    strategies=["popularity", "als", "user_knn", "linucb"],
    k=3,
)
print(results)  # DataFrame with metrics per strategy
```

### 3. Track student mastery

```python
from orchid_ranker import BayesianKnowledgeTracing

bkt = BayesianKnowledgeTracing(p_init=0.1, p_transit=0.1, p_slip=0.1, p_guess=0.2)
for correct in [True, True, False, True, True, True]:
    bkt.update(correct=correct)

print(f"P(mastery): {bkt.p_known():.3f}")
print(f"Mastered: {bkt.is_mastered()}")
```

### 4. Build a prerequisite-aware curriculum

```python
from orchid_ranker import PrerequisiteGraph, CurriculumRecommender

graph = PrerequisiteGraph()
graph.add_edge("algebra", "calculus")
graph.add_edge("algebra", "statistics")
graph.add_edge("calculus", "differential_equations")

cr = CurriculumRecommender(graph=graph, difficulty_map={
    "algebra": 0.3, "calculus": 0.6, "statistics": 0.5, "differential_equations": 0.8
})

# Student has mastered algebra — what should they learn next?
next_items = cr.recommend(student_mastery={"algebra"}, n=3)
print(next_items)  # ['calculus', 'statistics'] — respects prerequisites
```

### 5. Save and load models

```python
from orchid_ranker import save_model, load_model

save_model(rec, "my_model.orchid")
loaded = load_model("my_model.orchid")
print(loaded.recommend(user_id=1, top_k=3))
```

### 6. Hyperparameter tuning

```python
from orchid_ranker import GridSearchCV

gs = GridSearchCV(
    strategy="als",
    param_grid={"epochs": [5, 10, 20], "emb_dim": [32, 64]},
    cv=3,
    scoring="ndcg@10",
)
gs.fit(interactions, rating_col="rating")
print(f"Best params: {gs.best_params_}")
print(f"Best score: {gs.best_score_:.4f}")
```

---

## Recommendation Strategies

All strategies are accessible through `OrchidRecommender(strategy=...)`:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `als` | Alternating Least Squares matrix factorization | Sparse implicit feedback |
| `explicit_mf` | FunkSVD-style SGD factorization | Explicit 1-5 star ratings |
| `neural_mf` | Deep neural matrix factorization (BCE/BPR/softmax) | Non-linear interaction patterns |
| `linucb` | Linear Upper Confidence Bound contextual bandit | Exploration/exploitation with features |
| `user_knn` | User-based collaborative filtering | Smaller catalogs, interpretable |
| `popularity` | Items ranked by mean acceptance rate | Cold-start fallback |
| `random` | Uniform random sampling | Sanity-check baseline |
| `implicit_als` | Weighted ALS via the `implicit` library | Large-scale implicit feedback |
| `implicit_bpr` | Bayesian Personalized Ranking via `implicit` | Pairwise ranking optimization |

List all strategies programmatically:

```python
from orchid_ranker import STRATEGY_GUIDE
for name, desc in STRATEGY_GUIDE.items():
    print(f"  {name}: {desc}")
```

---

## Knowledge Tracing & Mastery

### Bayesian Knowledge Tracing (BKT)

Estimates the probability a learner has mastered a skill based on their response history:

```python
from orchid_ranker import BayesianKnowledgeTracing

bkt = BayesianKnowledgeTracing(
    p_init=0.1,       # Prior probability of knowing the skill
    p_transit=0.1,     # Probability of learning after each attempt
    p_slip=0.1,        # Probability of a careless error when knowing
    p_guess=0.2,       # Probability of guessing correctly when not knowing
    mastery_threshold=0.95,
)

bkt.update(correct=True)
print(bkt.p_known())      # Current mastery probability
print(bkt.is_mastered())   # True if p_known > threshold
bkt.reset()                # Reset to prior
```

### Multi-Skill Mastery Tracking

```python
from orchid_ranker import MasteryTracker

tracker = MasteryTracker(
    skills=["algebra", "geometry", "calculus"],
    default_params={"p_init": 0.1, "p_transit": 0.15},
)

tracker.update("algebra", correct=True)
tracker.update("algebra", correct=True)
print(tracker.mastered_skills())   # Skills above mastery threshold
print(tracker.recommend_next())    # Next skill to study
```

### Forgetting Curve

Models memory decay using the Ebbinghaus exponential forgetting model:

```python
from orchid_ranker import ForgettingCurve

fc = ForgettingCurve(initial_strength=1.0, strength_gain_on_review=0.5)
fc.review()  # Record a review event

retention = fc.retention_at(time_since_last_review=3600)  # After 1 hour
needs_review = fc.should_review(threshold=0.5)
```

---

## Curriculum Sequencing

### Prerequisite Graph

Model skill dependencies as a directed acyclic graph:

```python
from orchid_ranker import PrerequisiteGraph

graph = PrerequisiteGraph()
graph.add_edge("fractions", "algebra")
graph.add_edge("algebra", "calculus")
graph.add_edge("algebra", "linear_algebra")

# Query the graph
print(graph.topological_order())                    # Valid learning sequence
print(graph.prerequisites_for("calculus"))           # {'algebra'}
print(graph.all_prerequisites_for("calculus"))       # {'fractions', 'algebra'}
print(graph.available_skills(mastered={"fractions"}))# ['algebra']
print(graph.is_ready("calculus", mastered={"fractions", "algebra"}))  # True

# Find the learning path to a target skill
path = graph.learning_path("calculus", mastered={"fractions"})
print(path)  # ['algebra', 'calculus']

# Cycle detection is automatic
graph.add_edge("calculus", "fractions")  # Raises ValueError
```

### Curriculum Recommender

Recommends the next items to study, respecting prerequisites and targeting the learner's Zone of Proximal Development:

```python
from orchid_ranker import CurriculumRecommender

cr = CurriculumRecommender(
    graph=graph,
    difficulty_map={"fractions": 0.2, "algebra": 0.5, "calculus": 0.8, "linear_algebra": 0.7},
)

recommendations = cr.recommend(student_mastery={"fractions"}, n=3)
```

---

## Model Selection & Evaluation

### Train/Test Split

```python
from orchid_ranker import train_test_split

train, test = train_test_split(interactions, test_size=0.2, by_user=True, random_state=42)
```

### Cross-Validation

```python
from orchid_ranker import cross_validate

results = cross_validate(interactions, strategy="als", k=5, strategy_kwargs={"epochs": 10})
print(results)  # Dict with mean/std for each metric across folds
```

### Hyperparameter Tuning

```python
from orchid_ranker import GridSearchCV, RandomSearchCV

# Exhaustive search
gs = GridSearchCV(strategy="als", param_grid={"epochs": [5, 10], "emb_dim": [32, 64]}, cv=3)
gs.fit(interactions, rating_col="rating")

# Random search (faster for large grids)
rs = RandomSearchCV(
    strategy="als",
    param_distributions={"epochs": [5, 10, 20], "emb_dim": [16, 32, 64, 128]},
    n_iter=6, cv=3, random_state=42,
)
rs.fit(interactions, rating_col="rating")
```

### Educational Metrics

```python
from orchid_ranker import learning_gain, knowledge_coverage, difficulty_appropriateness

gain = learning_gain(pre_score=0.4, post_score=0.8)       # Normalized learning gain
coverage = knowledge_coverage(mastered={"a", "b"}, total_skills={"a", "b", "c", "d"})
appropriateness = difficulty_appropriateness(
    recommended_difficulties=[0.5, 0.55, 0.6],
    student_ability=0.5,
    zpd_width=0.25,
)
```

---

## Learner Simulation

The `StudentAgent` simulates realistic learner behavior with configurable knowledge, fatigue, engagement, and trust dynamics:

```python
from orchid_ranker import StudentAgent, StudentAgentFactory

student = StudentAgentFactory.create(
    user_id=1,
    knowledge_mode="scalar",  # or "IRT", "MIRT", "ZPD", "ContextualZPD"
    seed=42,
)

response = student.accept(
    item_id=10,
    difficulty=0.5,
    correct=True,
    dwell_time=30.0,
    feedback="positive",
)
print(f"Knowledge: {student.get_knowledge():.2f}")
print(f"Fatigue: {student.get_fatigue():.2f}")
print(f"Engagement: {student.get_engagement():.2f}")
```

---

## Adaptive Experiments

Run full adaptive vs. baseline experiments with the orchestration engine:

```python
from orchid_ranker.experiments import RankingExperiment

runner = RankingExperiment("configs/my_dataset.yaml", dataset="my_dataset", cohort_size=16)
summary = runner.run_many(
    strategies=["adaptive", "fixed", "linucb", "als", "popularity"],
    dp_enabled=False,
)
print(summary)
```

### Differential Privacy

Enable per-sample DP-SGD with gradient clipping and Gaussian noise:

```python
from orchid_ranker import get_dp_config

# Built-in privacy presets
dp_cfg = get_dp_config("eps_1")  # epsilon=1.0 preset
# Presets: "off", "eps_2", "eps_1", "eps_05", "eps_02"

# Or configure manually
dp_cfg = {
    "enabled": True,
    "engine": "per_sample",
    "noise_multiplier": 1.0,
    "sample_rate": 0.02,
    "max_grad": 1.0,
    "delta": 1e-5,
}
```

---

## Enterprise Features

### Observability

```python
from orchid_ranker import start_metrics_server, record_training, export_metrics

start_metrics_server(port=8000)  # Prometheus /metrics endpoint
record_training(strategy="als", dataset="ednet", metric_name="ndcg@10", value=0.42)
```

### Connectors

```python
from orchid_ranker import SnowflakeConnector, BigQueryConnector, S3StreamConnector, MLflowTracker

# Pull training data from Snowflake
sf = SnowflakeConnector(account="...", user="...", password="...", warehouse="...", database="...")

# Track experiments in MLflow
tracker = MLflowTracker(tracking_uri="http://localhost:5000", experiment_name="orchid-v2")
```

### Security

```python
from orchid_ranker import AccessControl, AuditLogger, DEFAULT_POLICY

ac = AccessControl(policy=DEFAULT_POLICY)
ac.check_permission(role="writer", action="train_model")  # True

logger = AuditLogger(log_path="audit.jsonl")
logger.log(event="model_trained", user="admin", details={"strategy": "als"})
```

### Deployment

- **Docker**: `docker build -t orchid-ranker .`
- **Kubernetes**: `helm install orchid ./deploy/helm/orchid-ranker`
- **Terraform**: See `deploy/terraform/README.md`

---

## Visualization

```python
from orchid_ranker.visualization import (
    plot_user_activity,
    plot_item_difficulty,
    plot_learning_curve,
    plot_knowledge_trajectory,
    plot_acceptance_heatmap,
)

plot_user_activity(interactions_df, top_n=25)
plot_item_difficulty(items_df)
plot_learning_curve(round_summary_df, metric="mean_accuracy")
```

---

## CLI

Evaluate strategies from the command line:

```bash
orchid-evaluate \
    --train data/train.csv \
    --test data/test.csv \
    --strategy "als,epochs=5" \
    --strategy "user_knn,k=25" \
    --strategy "popularity"
```

---

## Benchmarks

Compare against established libraries:

```bash
# vs. Surprise SVD
python benchmarks/compare_surprise.py --train train.csv --test test.csv --rating-col rating

# vs. implicit ALS/BPR (multi-seed)
python benchmarks/eval_implicit.py --seeds 11 13 17 --top-users 400 --top-items 800 --k 10

# Adaptive vs. fixed (synthetic learners)
python benchmarks/run_agentic_adaptive.py --rounds 80 --users 16 --items 64 --top-k 6

# MovieLens 100K
python benchmarks/run_agentic_ml100k.py --rounds 80 --top-users 400 --top-items 800 --top-k 6
```

---

## Dataset Format

Provide five CSV files and a YAML configuration:

- `train.csv`, `val.csv`, `test.csv` — each with `u`, `i`, `label` columns (plus optional `timestamp`, `correct`, `accept`)
- `side_information_users.csv` — per-learner features
- `side_information_items.csv` — per-item features

```yaml
run:
  dataset: my_dataset

datasets:
  my_dataset:
    paths:
      base_dir: data/my-dataset-processed
      train: train.csv
      val: val.csv
      test: test.csv
      side_information_users: side_information_users.csv
      side_information_items: side_information_items.csv
    interactions:
      timestamp: true
    users:
      categorical: [cohort, gender]
      numeric: [mean_accuracy, activity_span_days]
    items:
      categorical: [module, topic]
      numeric: [difficulty, recent_clicks_4w]
```

---

## Support Matrix

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.9 - 3.13 |
| PyTorch | 1.13 - 2.9 |
| OS | Ubuntu 22.04, macOS 14+, Windows Server 2022 |

See `docs/api_support_policy.md` for full versioning commitments.

---

## Project Layout

```
orchid-ranker/
  src/orchid_ranker/
    __init__.py           # Public API (74 symbols)
    recommender.py        # OrchidRecommender high-level API
    baselines.py          # 9 strategy implementations
    knowledge_tracing.py  # BKT, MasteryTracker, ForgettingCurve
    curriculum.py         # PrerequisiteGraph, CurriculumRecommender
    evaluation.py         # Ranking + educational metrics
    model_selection.py    # Cross-validation, train/test split
    tuning.py             # GridSearchCV, RandomSearchCV
    serialization.py      # Model save/load
    dp.py                 # Differential privacy presets
    observability.py      # Prometheus metrics
    agents/               # StudentAgent, TwoTowerRecommender, orchestrator
    connectors/           # Snowflake, BigQuery, S3, MLflow
    safety/               # SafeSwitch DR controller
    security/             # RBAC, audit logging
    visualization/        # Matplotlib plotting helpers
    data/                 # DatasetLoader
    experiments/          # RankingExperiment runner
  tests/                  # 440+ tests including stress tests
  benchmarks/             # Competitor comparisons
  configs/                # Dataset YAML configurations
  deploy/                 # Docker, Helm, Terraform
  docs/                   # Tutorials, security, compliance
  examples/               # Quickstart scripts and notebooks
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| `docs/quickstart.md` | Getting started tutorial |
| `docs/overview.md` | Architecture overview |
| `docs/tutorial_data_ingestion.md` | Dataset schema & ingestion |
| `docs/tutorial_dp.md` | Differential privacy deep dive |
| `docs/tutorial_safe_mode.md` | SafeSwitch walkthrough |
| `docs/tutorial_observability.md` | Monitoring & metrics |
| `docs/performance_playbook.md` | Performance tuning |
| `docs/security.md` | Security overview |
| `docs/enterprise_runbook.md` | Production checklists |
| `docs/benchmarking.md` | Benchmark CLI recipes |
| `docs/api_reference.md` | Complete API reference |

---

## Contributing

```bash
git clone https://github.com/farhad-vadiee/orchid-ranker.git
cd orchid-ranker
pip install -e ".[dev]"
python -m pytest tests/
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use Orchid Ranker in your research, please cite:

```bibtex
@software{vadiee2025orchid,
  author = {Vadiee, Farhad},
  title = {Orchid Ranker: Adaptive Educational Recommender Toolkit},
  version = {0.2.1},
  year = {2025},
  url = {https://github.com/farhad-vadiee/orchid-ranker}
}
```
