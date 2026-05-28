# Benchmarking Guide

Benchmark Orchid on the problem it claims to solve: adaptive-learning
recommendation from learner state, concept/difficulty metadata, progression
reward, and safe rollout evidence.

## Pytest Smoke Suite

```bash
python -m pytest tests/
```

The unit suite covers KT tracers, adaptive ranking, progression metrics, OPE,
guardrails, security helpers, and operational integrations.

For release evidence, also run the suite with deprecation warnings visible.
Unexpected Orchid deprecation warnings mean a test, example, or internal path is
still using old compatibility names instead of the stable adaptive-learning API.

## Credibility Artifact

For the primary Orchid claim, generate both JSON and Markdown:

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
    --output benchmarks/results_adaptive_efficiency_assistments.json \
    --report-md benchmarks/results_adaptive_efficiency_assistments.md
```

This produces one machine-readable artifact and one reviewer-friendly report
with KT prediction quality, policy uplift, support diagnostics, and runtime.
Start with the [benchmark credibility protocol](benchmarks/credibility.md) for
release, PR-review, or buyer-evaluation evidence.

A benchmark result is only publishable when the report states the dataset,
split, seeds, policy, reward definition, target-policy support, and decision
label. A positive uplift without support diagnostics is an engineering signal,
not a product claim.

## Knowledge-Tracing Benchmark

Run a SAKT/AKT-style tracer on an EdNet/ASSISTments-style interaction CSV with
chronological per-user holdout:

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

The replay predicts each held-out response before observing it, then updates the
learner history with that held-out outcome. This prevents future answers from
leaking into earlier predictions. Metrics include accuracy, AUC, Brier score,
log loss, and expected calibration error.

For ASSISTments raw data:

```bash
PYTHONPATH=src python benchmarks/assistments/preprocess.py \
    --interactions data/assistments/raw/skill_builder_data.csv \
    --format classic \
    --output data/assistments_kt/interactions.csv
```

For EdNet KT1 data:

```bash
PYTHONPATH=src python benchmarks/ednet/preprocess.py \
    --interactions data/ednet/KT1 \
    --questions data/ednet/contents/questions.csv \
    --max-files 10000 \
    --output data/ednet_kt/interactions.csv
```

## KT Policy OPE Benchmark

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
assumption because the raw logs do not include true propensities. Production or
experiment logs should pass a real `--logging-propensity-col`.

To evaluate delayed learning-gain proxy rewards:

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

## Adaptive Efficiency Benchmark

Use the consolidated benchmark when you want one artifact that combines KT
prediction quality, policy OPE, and wall-clock throughput:

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
    --output benchmarks/results_adaptive_efficiency_assistments.json \
    --report-md benchmarks/results_adaptive_efficiency_assistments.md
```

The current checked-in artifact uses synthetic candidate propensities. Treat it
as an engineering benchmark unless your run uses real logged propensities and a
validated reward definition.

## Reward-Model Diagnostics

Use the delayed-gain diagnostic benchmark when DR disagrees with SNIPS:

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

Keep support-constrained delayed-gain policies experimental until the direct
reward model is calibrated.

## Agentic Adaptive Smoke Runs

Synthetic fixed-vs-adaptive simulation:

```bash
PYTHONPATH=src python benchmarks/run_agentic_adaptive.py --rounds 80 --users 16 --items 64 --top-k 6
```

Lightweight orchestrator smoke:

```bash
PYTHONPATH=src python benchmarks/run_agentic_smoke.py --rounds 2 --users 3 --items 12
```

Use these to test operational wiring, not to claim general recommender-system
superiority.
