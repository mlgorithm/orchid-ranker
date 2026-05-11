# ASSISTments KT Benchmark

This benchmark path evaluates Orchid's experimental knowledge-tracing models on
ASSISTments-style data using chronological replay.

## Supported Raw Datasets

The preprocessor supports:

- classic ASSISTments 2009/2012-style interaction CSVs with columns such as
  `user_id`, `problem_id`, `correct`, `order_id`, and optional `skill_id`.
- [FoundationalASSIST](https://huggingface.co/datasets/ASSISTments/FoundationalASSIST)-style files with `Interactions.csv` columns such as
  `user_xid`, `problem_id`, `discrete_score`, and `end_time`, plus optional
  `Skills.csv`.

[ASSISTments 2017](https://sites.google.com/view/assistmentsdatamining/dataset)
access is free but requires signing up through the official dataset page. The
classic [2009-2010 ASSISTments data](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)
page documents the skill-builder format and columns. FoundationalASSIST is
available through Hugging Face under a non-commercial research license. After
accepting the relevant terms, download the raw CSVs locally and run the
preprocessing step below.

## Preprocess

Classic ASSISTments:

```bash
PYTHONPATH=src python benchmarks/assistments/preprocess.py \
  --interactions data/assistments/raw/skill_builder_data.csv \
  --format classic \
  --output data/assistments_kt/interactions.csv
```

FoundationalASSIST:

```bash
PYTHONPATH=src python benchmarks/assistments/preprocess.py \
  --interactions data/foundationalassist/Data/Interactions.csv \
  --skills data/foundationalassist/Data/Skills.csv \
  --format foundational \
  --output data/assistments_kt/interactions.csv
```

The output schema is:

```text
user_id,item_id,correct,timestamp,difficulty[,skill_id,skill_name]
```

`difficulty` is computed as `1 - item_accuracy` on the processed data.

## Run

```bash
PYTHONPATH=src python benchmarks/kt_sakt_benchmark.py \
  --data data/assistments_kt/interactions.csv \
  --model sakt \
  --timestamp-col timestamp \
  --epochs 5 \
  --output benchmarks/results_kt_assistments_sakt.json

PYTHONPATH=src python benchmarks/kt_sakt_benchmark.py \
  --data data/assistments_kt/interactions.csv \
  --model akt \
  --timestamp-col timestamp \
  --item-difficulty-col difficulty \
  --epochs 5 \
  --output benchmarks/results_kt_assistments_akt.json
```

The evaluator predicts each held-out response before observing it, then updates
the learner history. This prevents future answers from leaking into earlier
predictions.

## Tiny Fixture Smoke

The repository includes a tiny fixture only to verify the preprocessing and CLI
pipeline:

```bash
PYTHONPATH=src python benchmarks/assistments/preprocess.py \
  --interactions benchmarks/fixtures/assistments_tiny_raw.csv \
  --format classic \
  --output /tmp/assistments_tiny_kt.csv

PYTHONPATH=src python benchmarks/kt_sakt_benchmark.py \
  --data /tmp/assistments_tiny_kt.csv \
  --model akt \
  --timestamp-col timestamp \
  --item-difficulty-col difficulty \
  --epochs 1 \
  --d-model 16 \
  --n-heads 2
```

Do not cite the tiny fixture as a benchmark result. It only checks wiring.
The checked-in smoke output is
`benchmarks/results_kt_assistments_smoke.json`.

## Status

## Current Result

The first full-data run uses the classic ASSISTments 2009 skill-builder CSV
downloaded through `benchmarks/assistments/download.py`. Preprocessing with
`--min-user-events 5 --min-item-events 5` produced:

| Split | Events | Users | Items |
|-------|-------:|------:|------:|
| Train | 302,649 | 3,770 | 17,648 |
| Test | 77,114 | 3,768 | 14,770 |

Configuration:

```bash
PYTHONPATH=src python benchmarks/kt_sakt_benchmark.py \
  --data data/assistments_kt/interactions.csv \
  --model akt \
  --timestamp-col timestamp \
  --item-difficulty-col difficulty \
  --test-fraction 0.2 \
  --max-seq-len 50 \
  --d-model 32 \
  --n-heads 4 \
  --epochs 1 \
  --batch-size 512 \
  --device cpu \
  --output benchmarks/results_kt_assistments_akt.json
```

| Model | Accuracy | AUC | Brier | Log loss | ECE |
|-------|---------:|----:|------:|---------:|----:|
| Item mean | 0.6778 | 0.6934 | 0.2098 | 0.6143 | 0.0485 |
| SAKT, 1 epoch | 0.6571 | 0.6339 | 0.2187 | 0.6280 | **0.0067** |
| AKT-inspired, 1 epoch | **0.7052** | **0.7355** | **0.1945** | **0.5727** | 0.0106 |

The AKT-inspired tracer beats the item-mean baseline on accuracy, AUC, Brier
score, log loss, and calibration. The SAKT smoke run is better calibrated than
item mean but underperforms on ranking/error metrics at this one-epoch setting.

Artifacts:

- `benchmarks/results_kt_assistments_sakt.json`
- `benchmarks/results_kt_assistments_akt.json`
- `benchmarks/results_kt_assistments_smoke.json`
- `benchmarks/results_kt_policy_ope_assistments_akt.json`
- `benchmarks/results_kt_policy_ope_assistments_akt_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_akt_target09.json`
- `benchmarks/results_kt_policy_ope_assistments_akt_target09_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_progression_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_delayed_gain_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_delayed_gain_target09_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_delayed_gain_target095_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_delayed_gain_policy_target095_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_support_delayed_gain_target095_sweep.json`
- `benchmarks/results_adaptive_efficiency_assistments.json`
- `benchmarks/results_kt_policy_ope_smoke.json`

This is a real public-dataset result, but it is still an early one-epoch CPU
run. Before making a stronger claim, run multiple seeds and tune model size,
sequence length, and epochs.

## Policy OPE Slice

The first hardened logged-policy OPE slice uses the same ASSISTments 2009
split, the AKT-inspired tracer, `KTValuePolicy`, candidate sets of size 20, and
10,000 held-out replay events per seed. Because ASSISTments does not expose
true platform propensities, this benchmark uses a synthetic-uniform candidate
logging assumption and compares against a random-uniform candidate baseline.

| Metric | Value |
|--------|------:|
| Runs | 3 |
| Random baseline DR value, mean | 0.7027 |
| KTValuePolicy DR value, mean | 0.6999 |
| DR uplift, mean | -0.0028 |
| Across-seed uplift CI | [-0.0364, 0.0307] |
| Target match rate, mean | 0.0487 |
| Target effective sample size, mean | 487 |

This is not a positive policy-lift result. Cite it as an OPE pipeline proof
point and as evidence that the current heuristic policy needs improvement
before live rollout.

A target-correctness sensitivity run with `--target-correct 0.90` reports mean
DR uplift of +0.1880 over the same random-uniform candidate baseline. That is a
correctness-reward result, not a learning-gain result; it probably favors easier
items and should be used to motivate progression-gain rewards rather than to
claim a superior teaching policy.

The first progression-reward run uses `ProgressionValuePolicy` with
`--reward-mode progression` and `--concept-col skill_id`. Across seeds 11, 17,
and 23, the AKT-backed policy reports mean doubly robust uplift of +0.3212 over
the random-uniform candidate baseline on the progression reward. This is the
first positive policy result on the adaptive-learning objective, but it still
uses synthetic candidate propensities and has low target-policy coverage
(mean match rate 0.0198, mean ESS 198), so it should be cited as directional
offline evidence rather than a live-learning outcome claim.

The delayed same-skill gain run is stricter. With `--reward-mode delayed_gain`,
the default `target_correct=0.70` policy reports mean SNIPS uplift of -0.0305
against the same random-uniform candidate baseline. Tuning to
`target_correct=0.90` improves the result to -0.0211, and
`target_correct=0.95` gives -0.0208, but neither beats random. This is the
current progression-policy gap.

`DelayedGainValuePolicy` uses training-only same-concept delayed-gain priors in
the ranker. At `target_correct=0.95`, the three-seed delayed-gain sweep improves
to mean SNIPS uplift +0.0010 with CI [-0.0081, 0.0100]. Orchid is now roughly
break-even on the strict delayed-gain proxy, but this is not yet a decisive
positive delayed learning-gain result.

The learned `SupportConstrainedDelayedGainPolicy` increases target coverage
from 0.0194 to 0.0787 and ESS from 133 to 539. However, the direct reward model
fails the stricter doubly robust check: mean DR uplift is -0.0560 with CI
[-0.0806, -0.0314]. Treat this as a useful negative result: support improved,
but the direct delayed-gain model needs calibration before it should drive the
headline policy.

## Adaptive Efficiency Snapshot

The consolidated adaptive-efficiency artifact runs SAKT, AKT, the correctness
policy, the progression policy, and delayed-gain policy slices in one command:

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

| Metric | Result |
|--------|-------:|
| Total benchmark time | 306.4 sec |
| AKT AUC / accuracy | 0.7312 / 0.7001 |
| SAKT AUC / accuracy | 0.6369 / 0.6610 |
| AKT replay speed | 3,588 events/sec |
| SAKT replay speed | 2,545 events/sec |
| Best progression-policy uplift | +0.3148 |
| Best delayed-gain uplift | +0.0037 |

This benchmark is useful for checking end-to-end engineering efficiency and
policy direction after algorithm changes. It is intentionally not stronger than
the multi-seed policy artifacts above, because this snapshot currently uses one
seed.
