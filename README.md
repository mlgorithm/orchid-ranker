# Orchid Ranker

Orchid Ranker is an adaptive educational recommender toolkit that pairs
rich dataset preprocessing pipelines with a modular slate orchestration
engine, learner simulators, and a plug-and-play recommender class you can
drop straight into your product (much like `surprise`’s algorithms).
The toolkit grew out of experiments with the EdNet and OULAD datasets and now bundles:

- preprocessing utilities (`orchid_ranker.preprocessing`) that transform
  raw learner interaction logs into feature-rich CSV bundles;
- agent modules (`orchid_ranker.agents`) implementing student
  simulators, recommender policies, and the multi-user orchestration
  loop;
- ready-to-use baseline recommenders plus an `OrchidRecommender`
  high-level API that feels similar to `surprise` (now including an
  explicit MF/FunkSVD-style baseline); and
- utilities for running offline experiments similar to those used in the
  associated LAK'26 studies.

## Installation

```bash
pip install .
```

or, once published, simply:

```bash
pip install orchid-ranker
```

You can opt into extra functionality when installing from PyPI:

```bash
pip install orchid-ranker[agentic,viz,preprocess,benchmarks]
```

- `agentic` brings in optional experiment helpers.
- `viz` adds plotting dependencies.
- `preprocess` installs CLI preprocessing extras.

## Enterprise readiness

- See `docs/enterprise_runbook.md` for release checklists, CI smoke tests, monitoring guidance, and rollout strategy.
- To sanity-check the SafeSwitch non-regression gate on every release, run `./scripts/ci_safe_smoke.sh` (which wraps the ML-100K safe smoke scenario).
- For a hands-on tutorial showing how to run the adaptive vs fixed benchmark, safe smoke run, and inspect telemetry, see `docs/tutorial_safe_mode.md`.
- Prefer notebooks? Open `docs/tutorials/safe_mode.ipynb` to run the same workflow interactively (baseline run, safe smoke script, telemetry visualization).
- `benchmarks` installs optional competitor libraries (`implicit`, `reclab`).
 - `agentic` now bundles Opacus for production-grade DP accounting.

### Support policy

See `docs/api_support_policy.md` for the officially supported runtime matrix and versioning commitments.

| Component | Supported versions | Notes |
|-----------|-------------------|-------|
| Python    | 3.9 – 3.13         | Verified in CI across CPython builds |
| PyTorch   | 1.13 – 2.9         | Primary focus on 2.x for GPU optimisations |
| OS        | Ubuntu 22.04, macOS 14+, Windows Server 2022 | Windows coverage targets CPU paths |

### Quickstart & Deployment

- Follow `docs/quickstart.md` or run `python examples/quickstart.py` to generate sample data and train your first model.
- Work through the full walkthrough in `docs/tutorials/library_walkthrough.md` for DP, observability, and simulation extras.

### Deployment quickstart

- Build the container image with `docker build -t orchid-ranker .` (see `Dockerfile`).
- Kubernetes users can install via `helm install orchid ./deploy/helm/orchid-ranker` and configure audit forwarding/metrics through the supplied values.
- Terraform users should reference the module guidance in `deploy/terraform/README.md` or wrap the Helm chart in their own release module.

## Quick start

### As a plug-and-play recommender

```python
import pandas as pd
from orchid_ranker import OrchidRecommender, Recommendation, configure_logging

# Optional: wire library logs into your observability stack
configure_logging(level="INFO")

# 1. Interaction log (implicit labels or ratings)
interactions = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3, 3],
    "item_id": [10, 12, 10, 11, 12, 13, 14],
    "label":   [1,  1,  1,  0,  1,  0,  1],
})

# Optional: item-side features (align with sorted item ids)
item_side = pd.DataFrame({
    "item_id": [10, 11, 12, 13, 14],
    "difficulty": [0.35, 0.80, 0.55, 0.25, 0.60],
    "popularity": [120, 45, 85, 30, 65],
})
item_features = (
    item_side[["difficulty", "popularity"]]
    .to_numpy(dtype="float32")
)

# 2. Instantiate a Surprise-style recommender (choose any strategy)
strategies = [
    ("linucb", {"alpha": 1.5, "item_features": item_features}),
    ("als", {"epochs": 5}),
    ("explicit_mf", {"epochs": 20, "emb_dim": 64}),
    ("user_knn", {"k": 25}),
    ("popularity", {}),
    ("random", {}),
]

for strategy, kwargs in strategies:
    rec = OrchidRecommender(
        strategy=strategy,
        validate_inputs=True,
        **{k: v for k, v in kwargs.items() if k not in {"item_features"}},
    )
    rec.fit(
        interactions,
        rating_col="label",
        item_features=kwargs.get("item_features"),
    )
    print(f"{strategy.title()} recommendations:", rec.recommend(user_id=1, top_k=5))

# Predict a specific score if needed
als_rec = OrchidRecommender(strategy="als", epochs=5).fit(interactions, rating_col="label")
print("ALS predicted relevance:", als_rec.predict(user_id=1, item_id=10))
```

### Running adaptive vs. baseline experiments

```python
from orchid_ranker.preprocessing import preprocess_ednet
from orchid_ranker.experiments import RankingExperiment

# 1) Preprocess your raw EdNet dump
preprocess_ednet(
    base_path="/path/to/raw/u-files",
    content_path="/path/to/content",
    output_path="./data/ednet-processed",
)

# 2) Run a quick comparison with the experiment driver
runner = RankingExperiment("configs/ednet.yaml", dataset="ednet", cohort_size=16)
summary = runner.run_many(["adaptive", "fixed", "linucb", "als"], dp_enabled=False)
print(summary)
```

See the `experiments/` directory for end-to-end experiment scripts and the
`runs/` folder for generated reports.

### Apples-to-apples benchmarks

- Explicit ratings (Orchid vs Surprise SVD):

```bash
PYTHONPATH=src python benchmarks/compare_surprise.py \
  --train tmp/ml100k_explicit/train.csv \
  --test tmp/ml100k_explicit/test.csv \
  --rating-col rating \
  --orchid-strategy explicit_mf \
  --orchid-epochs 20 \
  --orchid-emb 64
```

- Implicit top‑K (binary, filter_seen, multi-seed):

```bash
PYTHONPATH=src python benchmarks/eval_implicit.py --seeds 11 13 17 --top-users 400 --top-items 800 --k 10
```

### Agentic (adaptive) benchmarks

- Fixed vs Adaptive (synthetic) with warmup/replay:

```bash
PYTHONPATH=src python benchmarks/run_agentic_adaptive.py --rounds 80 --users 16 --items 64 --top-k 6
```

- Optional Funk-guided candidates/distillation for adaptive:

```bash
PYTHONPATH=src python benchmarks/run_agentic_adaptive.py --rounds 80 --users 16 --items 64 --top-k 6 \
  --funk-candidates --funk-pool 48
```

- MovieLens 100K fixed vs adaptive (MF-derived features):

```bash
PYTHONPATH=src python benchmarks/run_agentic_ml100k.py \
  --rounds 80 --top-users 400 --top-items 800 --top-k 6 --dim 32 --funk-candidates
```
Add `--quick` to run a much lighter configuration (smaller user/item subsets and rounds).

- sklearn digits variant (CPU-safe):

```bash
PYTHONPATH=src python benchmarks/run_agentic_sklearn_digits.py --rounds 40 --users 64 --top-k 5 --dim 8
```

## Dataset format at a glance

You can plug in **any** dataset as long as you provide five CSV files and
a short YAML schema:

- `train.csv`, `val.csv`, `test.csv` each with at least `u`, `i`, `label`
  (optionally `timestamp`, `correct`, `accept`, etc.).
- `side_information_users.csv` describing per-learner features.
- `side_information_items.csv` describing per-item features.
- `configs/<your-dataset>.yaml` declaring which columns are categorical
  vs numeric and where the CSVs live.

Minimal YAML example:

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

As long as these files exist, `orchid_ranker.data.DatasetLoader` handles
all encoding automatically.

## Visualising your data

The `orchid_ranker.visualization` module provides lightweight helpers:

```python
from orchid_ranker.visualization import (
    plot_user_activity,
    plot_item_difficulty,
    plot_learning_curve,
)

plot_user_activity(interactions_df, top_n=25)
plot_item_difficulty(items_df)
plot_learning_curve(round_summary_df, metric="mean_accuracy")
```

Each function returns a Matplotlib axes so you can further customise the
plot before saving it.


You can also toggle differential privacy quickly via presets:

```python
from orchid_ranker.dp import get_dp_config
from orchid_ranker.experiments import RankingExperiment

runner = RankingExperiment("configs/ednet.yaml", dataset="ednet")
summary = runner.run_many(["adaptive", "fixed"], dp_params=get_dp_config("eps_05"))
```

For lower-level control, instantiate `TwoTowerRecommender` with a `dp_cfg`
payload. The default engine (`"per_sample"`) applies DP-SGD with per-example
clipping:

```python
dp_cfg = {
    "enabled": True,
    "engine": "per_sample",
    "noise_multiplier": 1.0,
    "sample_rate": 0.02,
    "max_grad": 1.0,
    "delta": 1e-5,
}
model = TwoTowerRecommender(..., dp_cfg=dp_cfg)
```


### Built-in baseline modes

`RankingExperiment` understands the following non-adaptive policies out of the box:

| Mode        | Description                                  |
|-------------|----------------------------------------------|
| `fixed`     | Two-tower recommender without online updates |
| `popularity`| Mean acceptance per item                     |
| `random`    | Uniform slate sampling                       |
| `als`       | Matrix-factorization baseline (trained once) |
| `implicit_als` | Weighted implicit ALS via the `implicit` package |
| `implicit_bpr` | Bayesian Personalized Ranking optimiser (`implicit`) |
| `neural_mf` | Shallow neural matrix factorisation with MLP head |
| `user_knn`  | User-based collaborative filtering           |
| `linucb`    | Linear contextual UCB over item features     |

Run them via `runner.run_many([...])` and everyone will report the same summary metrics.


To replicate the full battery of adaptive vs fixed comparisons on EdNet and OULAD run:

```bash
python experiments-sac/run_all.py
```

Results are written under the `runs/` folder (summary CSVs plus per-round metrics).

## Automated checks

- Quick pytest suite: `python -m pytest tests/`
- Smoke orchestrator run: `python benchmarks/run_agentic_smoke.py`
- Compare with Surprise (optional dependency):

  ```bash
  python benchmarks/compare_surprise.py \
      --train path/to/train.csv \
      --test path/to/test.csv \
      --rating-col label
  ```
- Compare with `implicit` (optional dependency):

  ```bash
  python benchmarks/compare_implicit.py \
      --train path/to/train.csv \
      --test path/to/test.csv \
      --rating-col label
  ```
- Compare with ReCLaB TopPop (optional dependency):

  ```bash
  python benchmarks/compare_reclab.py --env topics-static-v1-small
  ```
- Evaluate Orchid strategies and simple sweeps via the CLI:

  ```bash
  orchid-evaluate \
      --train data/train.csv \
      --test data/test.csv \
      --strategy "als,epochs=5" \
      --strategy "implicit_als,factors=64,iterations=10"
  ```

## Operational readiness

- Input validation is enabled by default; pass `validate_inputs=False` for
  best-effort casting when integrating with legacy pipelines.
- Use `configure_logging(level="INFO")` to emit structured logs compatible
  with enterprise observability platforms.
- Enforce role-based access on CLI preprocessors with `--role` and capture DP
  audit trails via the `AuditLogger` exposed in `orchid_ranker.security`.
- Review `docs/security.md` and `docs/compliance/` for SBOM guidance, incident
  response playbook, and retention policies when preparing enterprise rollouts.
- Expose Prometheus metrics via `orchid_ranker.start_metrics_server()` or export with `orchid_ranker.export_metrics()`. Integrate Snowflake/BigQuery/S3 data sources and MLflow tracking via `orchid_ranker.connectors` classes.
- For onboarding/support workflows, see `docs/customer_success/` (playbooks, SLAs, pilot plan) and the seeded notebooks under `examples/notebooks/`.
- See `docs/benchmarking.md` for CLI recipes that compare Orchid against
  Surprise, implicit, and ReCLaB baselines.
- Leverage `orchid_ranker.evaluation` (Precision@K, MAP@10, NDCG@10, calibration)
  for notebook-based analysis or custom pipelines.


## Project layout

- `docs/` – concise reference guides for privacy and API overviews.
- `examples/` – runnable scripts demonstrating common workflows.
- `tests/` – pytest-based smoke tests covering the public API.
- `src/orchid_ranker/contrib/` – legacy or experimental components kept for
  backwards compatibility.
