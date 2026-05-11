# KT Policy OPE Benchmark

This benchmark evaluates a KT-guided next-item policy with offline policy
evaluation. It is different from the KT prediction benchmark:

- KT prediction asks: "Can the tracer predict the next correctness label?"
- Policy OPE asks: "Would the policy's chosen next item improve reward versus
  a baseline under logged-action replay assumptions?"

## Assumption

Most public adaptive-learning datasets, including classic ASSISTments logs, do
not include the platform's true action propensities. The benchmark therefore
uses a documented synthetic assumption:

1. For each held-out logged event, build a candidate set containing the logged
   item plus sampled distractors from train-known items.
2. Treat the logged action as if it was drawn uniformly from that candidate set.
3. Evaluate `KTValuePolicy` against a random-uniform candidate baseline with
   `orchid_ranker.ope.compare_logged_policies`.

This is useful for regression testing and directional policy comparison. It is
not a causal deployment claim. If production logs contain true propensities,
use `orchid_ranker.ope` directly with those propensities.

## Run

```bash
PYTHONPATH=src python benchmarks/kt_policy_ope_benchmark.py \
  --data data/assistments_kt/interactions.csv \
  --model akt \
  --timestamp-col timestamp \
  --item-difficulty-col difficulty \
  --test-fraction 0.2 \
  --candidate-size 20 \
  --max-events 10000 \
  --max-seq-len 50 \
  --d-model 32 \
  --n-heads 4 \
  --epochs 1 \
  --batch-size 512 \
  --device cpu \
  --output benchmarks/results_kt_policy_ope_assistments_akt.json
```

## Current ASSISTments Result

The first hardened run uses the classic ASSISTments 2009 split from the KT
benchmark, caps OPE replay at 10,000 held-out events per run, and repeats the
AKT-backed policy replay for seeds 11, 17, and 23.

| Metric | Value |
|--------|------:|
| Runs | 3 |
| OPE events per run | 10,000 |
| Candidate size | 20 |
| Mean target match rate | 0.0487 |
| Mean target effective sample size | 487 |
| Mean random baseline DR value | 0.7027 |
| Mean KTValuePolicy DR value | 0.6999 |
| Mean DR uplift | -0.0028 |
| Across-seed uplift CI | [-0.0364, 0.0307] |

Interpretation: the policy-OPE pipeline works, but the default stretch policy
does not yet beat a random-uniform candidate baseline under this synthetic
replay. This is a useful negative result: Orchid can now detect when a new
adaptive policy is not ready to ship.

## Target-Correctness Sensitivity

Changing only `--target-correct` from `0.70` to `0.90` produces a strong
correctness uplift on the same three seeds:

| Metric | Value |
|--------|------:|
| Mean random baseline DR value | 0.7027 |
| Mean KTValuePolicy DR value | 0.8908 |
| Mean DR uplift | +0.1880 |
| Across-seed uplift CI | [0.1830, 0.1930] |
| Mean target match rate | 0.0453 |
| Mean target effective sample size | 453 |

This is not automatically a better adaptive-learning policy. It likely favors
easier items because the reward is immediate correctness. The result is useful
because it exposes the next product question clearly: Orchid needs progression
or learning-gain rewards, not just correctness rewards, before claiming
pedagogical policy lift.

## Progression Reward Policy

`ProgressionValuePolicy` changes the optimization target from immediate
correctness to a transparent progression reward. The reward combines predicted
correctness, target-correctness fit, item difficulty, learner competence,
stretch-zone fit, mastery-gain potential, and easy/hard/repetition penalties.

On the same ASSISTments 2009 split, with AKT-backed predictions, candidate
sets of size 20, 10,000 replay events, and seeds 11, 17, and 23:

| Metric | Value |
|--------|------:|
| Runs | 3 |
| Mean random baseline DR value | 0.6220 |
| Mean ProgressionValuePolicy DR value | 0.9431 |
| Mean DR uplift | +0.3212 |
| Across-seed uplift CI | [0.3146, 0.3278] |
| Mean target match rate | 0.0198 |
| Mean target effective sample size | 198 |

Interpretation: this is the first positive policy result on the progression
objective, but it is still an offline replay under synthetic candidate
propensities. The low target match rate means it should be treated as evidence
for reward-design direction, not as a causal live-learning claim.

## Delayed Same-Skill Gain

The harder benchmark asks whether the selected item is followed by better
future correctness on the same skill. Use `--reward-mode delayed_gain`:

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

The delayed-gain reward is:

```text
clip(0.5 + 0.5 * (future same-skill correctness - train prior same-skill correctness), 0, 1)
```

The progression-policy delayed-gain runs use SNIPS because they do not fit a
direct delayed-gain reward model.

| Policy setting | Target value | Baseline value | Uplift | Across-seed uplift CI | Match rate | ESS |
|----------------|-------------:|---------------:|-------:|----------------------:|-----------:|----:|
| `target_correct=0.70` | 0.5172 | 0.5477 | -0.0305 | [-0.0520, -0.0089] | 0.0139 | 95 |
| `target_correct=0.90` | 0.5266 | 0.5477 | -0.0211 | [-0.0349, -0.0073] | 0.0147 | 101 |
| `target_correct=0.95` | 0.5268 | 0.5477 | -0.0208 | [-0.0361, -0.0056] | 0.0144 | 98 |

Interpretation: tuning toward safer correctness reduces the loss, but the
current progression policy still does not beat random candidates on delayed
same-skill gain. This is the benchmark that should drive the next policy
iteration.

`DelayedGainValuePolicy` adds training-only same-concept delayed-gain priors to
the ranker. On the same ASSISTments split, with `target_correct=0.95`, seeds
11, 17, and 23:

| Policy setting | Target value | Baseline value | Uplift | Across-seed uplift CI | Match rate | ESS |
|----------------|-------------:|---------------:|-------:|----------------------:|-----------:|----:|
| `DelayedGainValuePolicy`, `target_correct=0.95` | 0.5486 | 0.5477 | +0.0010 | [-0.0081, 0.0100] | 0.0194 | 133 |

Interpretation: this is a material improvement over the progression policy's
negative delayed-gain result, but it is not a decisive win. The point estimate
is slightly positive while the confidence interval still crosses zero.

`SupportConstrainedDelayedGainPolicy` adds a direct delayed-gain reward model
and a support penalty for low-coverage actions. This improves replay coverage,
but the first direct-model result is negative under doubly robust OPE:

| Policy setting | Estimator | Target value | Baseline value | Uplift | Across-seed uplift CI | Match rate | ESS |
|----------------|-----------|-------------:|---------------:|-------:|----------------------:|-----------:|----:|
| `SupportConstrainedDelayedGainPolicy`, `target_correct=0.95` | DR | 0.4867 | 0.5427 | -0.0560 | [-0.0806, -0.0314] | 0.0787 | 539 |

The same run has mean SNIPS uplift +0.0021, so the policy is not obviously
worse under pure replay. The DR failure is a calibration warning: the direct
reward model is not reliable enough to use as the delayed-gain value model yet.

## Delayed-Gain Reward-Model Diagnostics

`benchmarks/delayed_gain_model_benchmark.py` diagnoses the direct reward model
separately from aggregate policy value. It reports validation and cross-fitted
model error, decile lift, and calibration on target-policy matched actions.

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

On the seed-11 ASSISTments diagnostic, ordinary validation metrics look good,
but target-action calibration fails:

| Reward-model weighting | Validation RMSE | Cross-fit RMSE | Target-match n | Target-match bias | Target-match RMSE |
|------------------------|----------------:|---------------:|---------------:|------------------:|------------------:|
| `uniform` | 0.0981 | 0.0987 | 617 | +0.0849 | 0.2059 |
| `support_inverse` | 0.0981 | 0.0987 | 636 | +0.0848 | 0.2064 |

Interpretation: the model is calibrated on the logged training distribution but
overpredicts reward for the actions selected by the learned support-constrained
policy. A one-seed weighted OPE run confirms the issue: support-inverse direct
model training produced DR uplift -0.1186, while target SNIPS stayed much closer
to neutral. The next algorithmic step is not a bigger ranker; it is a value
model trained with true target/logging action weights or a more conservative
lower-confidence value objective.

## Adaptive Efficiency Snapshot

`benchmarks/adaptive_efficiency_benchmark.py` runs the KT quality benchmark and
the policy grid in one pass, then records wall-clock throughput. The current
ASSISTments artifact uses seed 11, candidate sets of size 20, 10,000 replay
events per policy slice, and one CPU training epoch.

| Slice | Setting | Value |
|-------|---------|------:|
| Best KT tracer | AKT-inspired | AUC 0.7312 |
| Best KT tracer | AKT-inspired | Accuracy 0.7001 |
| AKT replay speed | 77,114 held-out events | 3,588 events/sec |
| SAKT replay speed | 77,114 held-out events | 2,545 events/sec |
| Best policy uplift | Progression reward, `target_correct=0.70` | +0.3148 |
| Best delayed-gain uplift | Delayed-gain policy, `target_correct=0.95` | +0.0037 |
| Total benchmark time | KT plus policy grid | 306.4 sec |

Interpretation: the consolidated run confirms that AKT is the stronger tracer
for this configuration and that the progression proxy is computationally
practical. It also shows the main research gap more precisely: delayed
same-skill gain is now around break-even, but not yet a statistically stable
positive result.

Artifacts:

- `benchmarks/results_adaptive_efficiency_assistments.json`
- `benchmarks/results_kt_policy_ope_assistments_akt_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_akt_target09_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_progression_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_delayed_gain_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_delayed_gain_target09_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_delayed_gain_target095_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_delayed_gain_policy_target095_sweep.json`
- `benchmarks/results_kt_policy_ope_assistments_support_delayed_gain_target095_sweep.json`
- `benchmarks/results_delayed_gain_model_assistments_target095.json`
- `benchmarks/results_kt_policy_ope_assistments_support_delayed_gain_weighted_seed11.json`
- `benchmarks/results_kt_policy_ope_assistments_akt.json`
- `benchmarks/results_kt_policy_ope_assistments_akt_target09.json`
- `benchmarks/results_kt_policy_ope_smoke.json`
