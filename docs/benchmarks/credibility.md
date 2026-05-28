# Benchmark credibility protocol

This page defines the benchmark standard for Orchid's central claim:

> Orchid is a production adaptive-learning recommender for choosing the next
> learning item safely, with learner state, prerequisites, KT or IRT signals,
> offline policy evidence, and explicit rollout gates.

The benchmark goal is not to prove that Orchid is the best generic recommender
library. It is to make the adaptive-learning claim reproducible and falsifiable.

## One-command report

Use the consolidated benchmark when you want a single JSON artifact plus a
reviewer-friendly Markdown report:

```bash
PYTHONPATH=src python benchmarks/adaptive_efficiency_benchmark.py \
  --data data/assistments_kt/interactions.csv \
  --models akt sakt \
  --seeds 11 17 23 \
  --timestamp-col timestamp \
  --item-difficulty-col difficulty \
  --concept-col skill_id \
  --policy-targets 0.70 0.90 0.95 \
  --policy-rewards progression delayed_gain \
  --candidate-size 20 \
  --max-events 10000 \
  --max-seq-len 50 \
  --d-model 32 \
  --n-heads 4 \
  --epochs 1 \
  --batch-size 512 \
  --device cpu \
  --output benchmarks/results_adaptive_efficiency_assistments.json \
  --report-md benchmarks/results_adaptive_efficiency_assistments.md
```

The Markdown report records:

- the data contract: rows, users, items, split, model list, seeds, device, and runtime;
- KT prediction quality against the item-mean baseline;
- policy OPE value against the random candidate baseline;
- match rate and effective sample size, so weak support is visible;
- a conservative decision label: `candidate for canary`, `research only`, or
  `do not ship`.

The report only emits a canary-style decision when the run has at least three
seeds and target-policy effective sample size of at least 100. Lower-support
runs remain `research only`, even if the point estimate is positive.

## Smoke run

For CI or local wiring checks, use the tiny fixture. Do not cite this as a
benchmark result.

```bash
PYTHONPATH=src python benchmarks/adaptive_efficiency_benchmark.py \
  --data benchmarks/fixtures/assistments_tiny_raw.csv \
  --models akt \
  --seeds 3 \
  --user-col user_id \
  --item-col problem_id \
  --correct-col correct \
  --timestamp-col order_id \
  --concept-col skill_id \
  --policy-targets 0.70 \
  --policy-rewards progression \
  --candidate-size 2 \
  --max-events 8 \
  --max-seq-len 3 \
  --d-model 16 \
  --n-heads 2 \
  --epochs 1 \
  --batch-size 4 \
  --device cpu \
  --output /tmp/orchid-smoke.json \
  --report-md /tmp/orchid-smoke.md
```

## Evidence ladder

| Level | Evidence | Meaning |
|-------|----------|---------|
| L0 | Unit tests and import sweeps | Code paths run, but no quality claim. |
| L1 | Tiny fixture smoke report | Benchmark wiring works. Not a performance claim. |
| L2 | Public dataset, time-ordered split, fixed seeds | Reproducible offline evidence. |
| L3 | Public dataset plus competitor baselines | Shows where Orchid is or is not better than adjacent libraries. |
| L4 | Production logs with true logging propensities | Causal OPE becomes defensible. |
| L5 | Live A/B or canary with guardrails | Production outcome claim. |

The current public ASSISTments and specialty results are L2 to L3 depending on
the module. They are useful for release gating and product direction. They are
not substitutes for L4 or L5 evidence on a customer's own logs.

## 9.5/10 adaptive-learning bar

A high score for Orchid is not based on the number of recommender algorithms it
contains. It is based on whether the adaptive-learning workflow is coherent,
measurable, and safe to operate.

| Dimension | 9.5/10 expectation |
|-----------|--------------------|
| Public API focus | New docs and examples lead with `AdaptiveRanker`, `AdaptiveLearningEngine`, KT, OPE, progression policy, and guardrails. |
| KT quality | Time-ordered replay reports correctness quality, calibration, and item-mean baselines. |
| Next-item policy value | Progression or delayed-gain policy reports include uplift, confidence intervals, match rate, and ESS. |
| Learning metadata | Benchmarks use concepts, difficulty, prerequisites, or explicit cold-start metadata instead of plain user-item clicks. |
| Safety | Candidate policies pass OPE gates before canary and have a reviewed fallback policy. |
| Operations | Tests, lint, type checks, docs, package build, and import-contract checks pass with no unexpected deprecation warnings. |
| Claim discipline | Generic top-K recommendation, CTR, music/movie, and ad-ranking claims are absent. |

## Competitor interpretation

| Library | What it should beat Orchid at | What Orchid must prove instead |
|---------|-------------------------------|--------------------------------|
| [RecBole](https://recbole.io/docs/) | Broad recommender-model coverage, dataset tooling, and academic benchmark breadth. | A narrower adaptive-learning workflow that combines KT, progression reward, OPE, and safe serving. |
| [implicit](https://benfred.github.io/implicit/) | Fast implicit-feedback ALS/BPR and nearest-neighbor collaborative filtering. | Better next-learning-item decisions when item difficulty, prerequisites, and learner state matter. |
| [LightFM](https://making.lyst.com/lightfm/docs/) | Simple hybrid matrix factorization with user and item metadata and WARP/BPR losses. | Live progression-aware adaptation after each learner response. |
| [pyKT](https://github.com/pykt-team/pykt-toolkit) | Deep KT model benchmarking and KT architecture breadth. | Turning KT predictions into ranked next actions with offline rollout evidence. |

This comparison is intentionally asymmetric. Orchid should not claim to be a
larger model zoo than RecBole, a faster ALS implementation than `implicit`, a
simpler hybrid ranker than LightFM, or a broader KT benchmark than pyKT.

## Pass criteria for a 9.5/10 claim

A release can make a strong production-readiness claim only when:

1. `ruff`, `mypy`, docs, package build, import sweep, and full tests pass.
2. The benchmark report is regenerated from public data with fixed seeds.
3. JSON and Markdown artifacts are committed together.
4. Every positive result includes support diagnostics: match rate, ESS, and CI.
5. Every unsupported or negative result is documented as such.
6. Delayed-gain or progression claims are separated from immediate-correctness claims.
7. The docs and examples do not route users toward deleted generic recommender APIs.

This standard is deliberately strict. It makes Orchid look less like a grab bag
of algorithms and more like a library with a falsifiable product thesis.
