# Benchmarking Guide

Use the provided helpers to keep regression checks reproducible:

## Pytest smoke suite

```bash
python -m pytest tests/
```

Covers popular recommender strategies (ALS, LinUCB), cold-start behaviour, and
basic DP-accountant sanity checks.

## Agentic smoke run

```bash
python benchmarks/run_agentic_smoke.py --rounds 2 --users 3 --items 12
```

Creates synthetic users/items, wires them through `MultiUserOrchestrator`, and
emits JSONL logs under `runs/`.

## Surprise comparison (optional)

```bash
pip install surprise
python benchmarks/compare_surprise.py \
    --train data/train.csv \
    --test data/test.csv \
    --rating-col label
```

Prints RMSE for Orchid Ranker's ALS baseline alongside Surprise's SVD.

## Apples-to-apples — Explicit ratings (Orchid vs Surprise)

Use the updated comparison script to run Orchid's new explicit MF baseline against Surprise SVD on the same split:

```bash
# Prepare a quick ML-100K explicit split
PYTHONPATH=src python - <<'PY'
from surprise import Dataset
import numpy as np, pandas as pd
from pathlib import Path
ml = Dataset.load_builtin('ml-100k', prompt=False)
raw = ml.raw_ratings
df = pd.DataFrame(raw, columns=['user_id','item_id','rating','timestamp']).astype({'user_id':int,'item_id':int,'rating':float})
msk = np.random.default_rng(123).random(len(df)) < 0.8
train, test = df[msk], df[~msk]
u, i = set(train.user_id), set(train.item_id)
test = test[test.user_id.isin(u) & test.item_id.isin(i)]
out = Path('tmp/ml100k_explicit'); out.mkdir(parents=True, exist_ok=True)
train[['user_id','item_id','rating']].to_csv(out/'train.csv', index=False)
test[['user_id','item_id','rating']].to_csv(out/'test.csv', index=False)
print('wrote', out)
PY

# Compare: Orchid explicit MF vs Surprise SVD
PYTHONPATH=src python benchmarks/compare_surprise.py \
  --train tmp/ml100k_explicit/train.csv \
  --test tmp/ml100k_explicit/test.csv \
  --rating-col rating \
  --orchid-strategy explicit_mf \
  --orchid-epochs 20 \
  --orchid-emb 64
```

This prints RMSE for both models. In our local run, `explicit_mf` beat SVD with a modest config.

## Apples-to-apples — Implicit top-K (binary, filter_seen)

Run the multi-seed implicit benchmark that enforces a shared candidate set and filter_seen for all models:

```bash
PYTHONPATH=src python benchmarks/eval_implicit.py --seeds 11 13 17 --top-users 400 --top-items 800 --k 10
```

It reports mean±std for P@10/Recall@10/NDCG@10 across:
- Orchid `implicit_als`, `implicit_bpr`, and a couple of `neural_mf` (BPR) configs
- Surprise SVD used as a ranking proxy (trained on binary labels)
- Popularity

Outputs are persisted to `tmp/reports/implicit_results.{json,csv}` for CI consumption.

## CI integration (explicit + implicit)

- Add a simple CI step to run both apples-to-apples checks and persist results:

```yaml
- name: Apples-to-apples explicit (ML-100K)
  run: |
    PYTHONPATH=src python benchmarks/compare_surprise.py \
      --train tmp/ml100k_explicit/train.csv \
      --test tmp/ml100k_explicit/test.csv \
      --rating-col rating \
      --orchid-strategy explicit_mf \
      --orchid-epochs 20 \
      --orchid-emb 64 \
      --output tmp/reports/explicit_results.json

- name: Apples-to-apples implicit (ML-100K)
  run: |
    PYTHONPATH=src python benchmarks/eval_implicit.py --seeds 11 13 17 --top-users 400 --top-items 800 --k 10

- name: Archive benchmark artifacts
  uses: actions/upload-artifact@v4
  with:
    name: orchid-benchmarks
    path: |
      tmp/reports/explicit_results.*
      tmp/reports/implicit_results.*
```

You can add a small checker to compare JSON metrics against a known baseline and fail the workflow if deltas exceed thresholds.

## Implicit comparison

```bash
pip install implicit
python benchmarks/compare_implicit.py \
    --train data/train.csv \
    --test data/test.csv \
    --rating-col label
```

Reports RMSE for Orchid ALS and the `implicit` ALS implementation on the same
splits.

## ReCLaB dense benchmark

```bash
pip install reclab
python benchmarks/compare_reclab.py --env topics-static-v1-small
```

Generates train/test splits from the selected ReCLaB environment and compares
Orchid ALS with ReCLaB's TopPop recommender.

## Orchid evaluation CLI

```bash
orchid-evaluate \
    --train data/train.csv \
    --test data/test.csv \
    --strategy "als,epochs=5" \
    --strategy "implicit_als,factors=64"
```

Runs the built-in metrics suite (Precision@5, Recall@5, MAP@10, NDCG@10) and
supports basic hyper-parameter sweeps via repeated `--strategy` flags.

## Performance profiling

```bash
python benchmarks/profile_strategies.py --strategies als,neural_mf,user_knn --users 400 --items 600 --interactions 20000
```

Generates synthetic data, times each strategy’s `fit()` and `recommend()` calls, and prints a latency summary (optionally exporting JSON for CI dashboards).

### CI integration

- Run the profiler with `--output` to persist metrics (e.g. `python benchmarks/profile_strategies.py --output ci/perf.json`).
- Commit a baseline JSON and compare within CI (GitHub Actions, GitLab) to flag regressions above an agreed threshold.
- Example GitHub Actions snippet:

  ```yaml
  - name: Profile strategies
    run: python benchmarks/profile_strategies.py --strategies als,linucb,neural_mf --users 400 --items 600 --interactions 20000 --output perf.json
  - name: Check regression
    run: python scripts/check_perf_regression.py perf.json ci/baseline_perf.json --max-delta 0.1
  ```

Supply your own comparison script (`check_perf_regression.py`) to assert that latency increases remain within tolerance.

## Agentic benchmarks

Synthetic fixed vs adaptive:
```bash
PYTHONPATH=src python benchmarks/run_agentic_adaptive.py --rounds 80 --users 16 --items 64 --top-k 6
```
Optional Funk guidance:
```bash
PYTHONPATH=src python benchmarks/run_agentic_adaptive.py --rounds 80 --users 16 --items 64 --top-k 6 --funk-candidates --funk-pool 48
```

MovieLens 100K fixed vs adaptive:
```bash
PYTHONPATH=src python benchmarks/run_agentic_ml100k.py --rounds 80 --top-users 400 --top-items 800 --top-k 6 --dim 32 --funk-candidates
```
Add `--funk-distill` to include the Funk auxiliary loss or omit `--funk-candidates`
to run without MF-guided recall. Use `--quick` for a lightweight smoke run (25 rounds,
smaller user/item subsets).

### System-level tuning helpers

- Use `--timing-log runs/foo/timing.jsonl --timing-rounds 5` on `run_agentic_ml100k.py` (or `run_agentic_smoke.py`) to capture per-phase timings without the heavy Torch profiler. The orchestrator writes one JSON line per sampled round with `candidate_sampling`, `tower_infer`, `decide`, `student_interact`, `train_step`, and `warmup_sync` durations so you can diff runs in CI.
- Enable `--native-score` on any agentic benchmark or smoke test to route the tower dot-products through the optional C++ `fast_score` kernel when the build environment supports PyTorch extensions. When the extension cannot be built, the flag silently falls back to pure PyTorch matmul and the run still succeeds.
- For micro-level sanity checks (including CI), run `PYTHONPATH=src python benchmarks/bench_infer.py --iters 20 --users 64 --items 512 --candidates 128 --dim 32 --native-score`. The script reports milliseconds per inference call on the selected device (CPU, CUDA, or Apple MPS) and honours `--torch-compile` if you want to compare eager vs compiled towers.

#### Quick safe-mode MovieLens sanity run

If you just need to verify the SafeSwitch-DR gate on ML-100K without waiting for the full benchmark, use the helper script:

```bash
./scripts/run_ml100k_safe_smoke.sh [runs/ml100k-safe-smoke]
```

It runs the adaptive policy only (fixed is skipped) on a 60×120 slice for 5 rounds with `--safe-eb --safe-eb-dr`, and writes both the standard JSONL metrics and per-phase timings to the provided log directory. This configuration completes well under two minutes even on CPU/MPS laptops and is suitable for CI gates.
