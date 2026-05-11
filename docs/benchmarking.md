# Benchmarking Guide

Use the provided helpers to keep regression checks reproducible:

## Pytest smoke suite

```bash
python -m pytest tests/
```

Covers popular recommender strategies (ALS, LinUCB), cold-start behaviour, and
basic DP-accountant sanity checks.

## Adaptive-learning benchmark path

For the primary Orchid claim, start with knowledge tracing and policy OPE on an
adaptive-learning dataset:

```bash
PYTHONPATH=src python benchmarks/adaptive_efficiency_benchmark.py \
    --data data/assistments_kt/interactions.csv \
    --models akt \
    --seeds 11 17 23 \
    --timestamp-col timestamp \
    --item-difficulty-col difficulty \
    --concept-col skill_id \
    --policy-targets 0.90 \
    --policy-rewards progression delayed_gain \
    --include-kt-value-policy \
    --candidate-size 20 \
    --max-events 10000 \
    --epochs 1 \
    --output benchmarks/results_adaptive_efficiency_assistments.json
```

This produces one artifact with KT prediction quality, policy uplift, and
runtime. Generic recommender benchmarks are still useful regression checks, but
they are not Orchid's main product claim.

## Agentic smoke run

```bash
python benchmarks/run_agentic_smoke.py --rounds 2 --users 3 --items 12
```

Creates synthetic users/items, wires them through `MultiUserOrchestrator`, and
emits JSONL logs under `runs/`.

## Generic fallback comparison: Surprise (optional)

```bash
pip install surprise
python benchmarks/compare_surprise.py \
    --train data/train.csv \
    --test data/test.csv \
    --rating-col label
```

Prints RMSE for Orchid Ranker's ALS baseline alongside Surprise's SVD.

## Generic fallback comparison: explicit ratings

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

## Generic fallback comparison: implicit top-K

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

## Generic fallback comparison: implicit library

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

## Knowledge tracing benchmark

Run the experimental SAKT-style tracer on an EdNet/ASSISTments-style interaction
CSV with chronological per-user holdout:

```bash
PYTHONPATH=src python benchmarks/kt_sakt_benchmark.py \
    --data data/ednet_interactions.csv \
    --model akt \
    --user-col user_id \
    --item-col question_id \
    --correct-col correct \
    --timestamp-col timestamp \
    --item-difficulty-col difficulty \
    --epochs 5 \
    --output benchmarks/results_kt_sakt.json
```

The benchmark predicts each held-out response before observing it, then updates
the learner history with the held-out outcome. This replay protocol prevents
future answers from leaking into earlier predictions. Reported metrics include
accuracy, AUC, Brier score, log loss, and expected calibration error for SAKT
or AKT-inspired tracing and an item-mean correctness baseline. Use
`--model sakt` when no item difficulty column is available.

For ASSISTments raw data, first normalize the dataset into Orchid's KT schema:

```bash
PYTHONPATH=src python benchmarks/assistments/preprocess.py \
    --interactions data/assistments/raw/skill_builder_data.csv \
    --format classic \
    --output data/assistments_kt/interactions.csv
```

See [ASSISTments KT Benchmark](benchmarks/assistments-kt.md) for classic
ASSISTments, ASSISTments 2017, and FoundationalASSIST preprocessing details.

For EdNet KT1 data, normalize either a single denormalized CSV or a directory
of per-user KT1 CSV files:

```bash
PYTHONPATH=src python benchmarks/ednet/preprocess.py \
    --interactions data/ednet/KT1 \
    --questions data/ednet/contents/questions.csv \
    --max-files 10000 \
    --output data/ednet_kt/interactions.csv
```

The EdNet preprocessor supports raw KT1 rows with `user_answer` plus
`questions.csv` metadata, or denormalized rows that already include
`is_correct`.

## KT policy OPE benchmark

After a KT tracer predicts correctness, evaluate whether a next-item policy
would beat a baseline under logged-action replay assumptions:

```bash
PYTHONPATH=src python benchmarks/kt_policy_ope_benchmark.py \
    --data data/assistments_kt/interactions.csv \
    --model akt \
    --timestamp-col timestamp \
    --item-difficulty-col difficulty \
    --candidate-size 20 \
    --max-events 10000 \
    --seeds 11 17 23 \
    --epochs 1 \
    --output benchmarks/results_kt_policy_ope_assistments_akt_sweep.json
```

The public ASSISTments path uses a synthetic-uniform candidate logging
assumption because the raw logs do not include true propensities. It reports
IPS, SNIPS, direct-method, and doubly robust estimates through
`orchid_ranker.ope`. If your production or experiment logs include the true
probability assigned to the shown item, pass `--logging-propensity-col` to use
that real logging probability instead of the synthetic-uniform assumption.

To evaluate delayed learning-gain proxy rewards, pass
`--reward-mode delayed_gain` with a concept column:

```bash
PYTHONPATH=src python benchmarks/kt_policy_ope_benchmark.py \
    --data data/assistments_kt/interactions.csv \
    --model akt \
    --timestamp-col timestamp \
    --item-difficulty-col difficulty \
    --concept-col skill_id \
    --policy progression \
    --reward-mode delayed_gain \
    --delayed-gain-window 5 \
    --candidate-size 20 \
    --max-events 10000 \
    --seeds 11 17 23 \
    --epochs 1 \
    --output benchmarks/results_kt_policy_ope_assistments_delayed_gain_sweep.json
```

Delayed-gain mode uses SNIPS until Orchid has a calibrated direct model for
future learning gain.
See [KT Policy OPE Benchmark](benchmarks/kt-policy-ope.md) for the current
result and caveats.

## Adaptive efficiency benchmark

Use the consolidated adaptive-efficiency benchmark when you want one artifact
that combines KT prediction quality, policy OPE, and wall-clock throughput:

```bash
PYTHONPATH=src python benchmarks/adaptive_efficiency_benchmark.py \
    --data data/assistments_kt/interactions.csv \
    --models sakt akt \
    --seeds 11 \
    --timestamp-col timestamp \
    --item-difficulty-col difficulty \
    --concept-col skill_id \
    --policy-targets 0.70 0.90 0.95 \
    --policy-rewards correctness progression delayed_gain \
    --include-kt-value-policy \
    --candidate-size 20 \
    --max-events 10000 \
    --epochs 1 \
    --output benchmarks/results_adaptive_efficiency_assistments.json
```

The checked-in ASSISTments run reports AKT as the best one-epoch tracer
(AUC 0.7312, accuracy 0.7001) and the progression policy at
`target_correct=0.70` as the strongest policy slice (+0.3148 uplift on the
progression reward, 569 replay events/sec). The delayed-gain-aware policy is
now the best delayed same-skill gain slice in the single-seed grid
(`target_correct=0.95`, +0.0037 uplift), but the three-seed sweep is still only
break-even (+0.0010 uplift, CI [-0.0081, 0.0100]). Treat this as a compact
engineering benchmark, not as a live learning-effect claim: the current
artifact uses synthetic candidate propensities.

The learned support-constrained delayed-gain policy is available with
`--policy support_delayed_gain`. Its first three-seed ASSISTments run improved
target coverage from 0.0194 to 0.0787, but the direct reward model failed the
doubly robust check (DR uplift -0.0560). Keep it as an experimental diagnostic
until the reward model is calibrated.

Use the delayed-gain model diagnostic benchmark when DR disagrees with SNIPS:

```bash
PYTHONPATH=src python benchmarks/delayed_gain_model_benchmark.py \
    --data data/assistments_kt/interactions.csv \
    --model akt \
    --timestamp-col timestamp \
    --item-difficulty-col difficulty \
    --concept-col skill_id \
    --target-correct 0.95 \
    --reward-model-weightings uniform support_inverse \
    --reward-model-cross-fit-folds 3 \
    --epochs 1 \
    --output benchmarks/results_delayed_gain_model_assistments_target095.json
```

The current diagnostic shows that both uniform and support-inverse direct
models overpredict target-matched delayed-gain rewards by about +0.085. That is
why the support-constrained policy remains experimental even though ordinary
validation RMSE is around 0.098.

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

- Use `--timing-log runs/foo/timing.jsonl --timing-rounds 5` on `run_agentic_ml100k.py` (or `run_agentic_smoke.py`) to capture per-phase timings without the heavy Torch profiler. The orchestrator writes one JSON line per sampled round with `candidate_sampling`, `tower_infer`, `decide`, `user_interact`, `train_step`, and `warmup_sync` durations so you can diff runs in CI.
- Enable `--native-score` on any agentic benchmark or smoke test to route the tower dot-products through the optional C++ `fast_score` kernel when the build environment supports PyTorch extensions. When the extension cannot be built, the flag silently falls back to pure PyTorch matmul and the run still succeeds.
- For micro-level sanity checks (including CI), run `PYTHONPATH=src python benchmarks/bench_infer.py --iters 20 --users 64 --items 512 --candidates 128 --dim 32 --native-score`. The script reports milliseconds per inference call on the selected device (CPU, CUDA, or Apple MPS) and honours `--torch-compile` if you want to compare eager vs compiled towers.

#### Quick safe-mode MovieLens sanity run

If you just need to verify the SafeSwitch-DR gate on ML-100K without waiting for the full benchmark, use the helper script:

```bash
./scripts/run_ml100k_safe_smoke.sh [runs/ml100k-safe-smoke]
```

It runs the adaptive policy only (fixed is skipped) on a 60×120 slice for 5 rounds with `--safe-eb --safe-eb-dr`, and writes both the standard JSONL metrics and per-phase timings to the provided log directory. This configuration completes well under two minutes even on CPU/MPS laptops and is suitable for CI gates.

### Interpreting SafeSwitch telemetry

Every `round_summary` entry in the JSONL log contains a `safe_gate` block when `--safe-eb` is on:

```json
{
  "type": "round_summary",
  "round": 5,
  "safe_gate": {
    "serve_policy": "teacher",
    "p_used": 0.05,
    "t": 5,
    "mean": -10.1,
    "rad": 5.0,
    "lcb": -15.1,
    "acc_lcb": -2.3,
    "p": 0.05
  },
  ...
}
```

- `serve_policy`: which policy actually served the slate (`teacher` or `adaptive`).
- `p_used`: mix probability for the just-completed round.
- `lcb`: DR uplift lower confidence bound; once this turns positive, the gate ramps `p` up.
- `acc_lcb`: acceptance-rate lower bound (per user); if it dips below the configured floor, the gate falls back to the teacher (p=0).
- `p`: current probability that the next round will use the adaptive policy.

Plotting `lcb`, `acc_lcb`, and `p` over time makes it easy to demonstrate the non-regression guarantee.
