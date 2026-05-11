# Progression Policy

Correctness alone is not the objective in adaptive learning. A policy that
maximizes immediate correctness can look strong by recommending easier items.
Orchid's progression policy scores candidates with a transparent reward that
combines:

- predicted correctness
- item difficulty
- learner competence for the item concept
- stretch-zone fit
- target-correctness band fit
- mastery-gain potential
- repetition and too-easy penalties

## Use

```python
from orchid_ranker.learning_policy import ProgressionValuePolicy

policy = ProgressionValuePolicy(
    tracer,
    difficulty_by_item={10: 0.25, 20: 0.65, 30: 0.90},
    concept_by_item={10: "fractions", 20: "fractions", 30: "ratios"},
)

ranked = policy.rank("learner-7", [10, 20, 30], top_k=3)
policy.observe("learner-7", ranked[0].item_id, correct=True)
```

The policy is not a black-box RL algorithm. It is intentionally decomposed so
operators can inspect why an item won:

```python
top = ranked[0]
print(top.expected_reward)
print(top.reward.stretch_fit)
print(top.reward.mastery_gain)
print(top.reward.easy_penalty)
```

## Benchmark

Use the policy-OPE benchmark with progression reward:

```bash
PYTHONPATH=src python benchmarks/kt_policy_ope_benchmark.py \
  --data data/assistments_kt/interactions.csv \
  --model akt \
  --timestamp-col timestamp \
  --item-difficulty-col difficulty \
  --concept-col skill_id \
  --policy progression \
  --reward-mode progression \
  --candidate-size 20 \
  --max-events 10000 \
  --seeds 11 17 23 \
  --epochs 1 \
  --output benchmarks/results_kt_policy_ope_assistments_progression_sweep.json
```

Treat this reward as a first operational proxy, not a final pedagogical truth.
The strongest version will use real pre/post mastery deltas or delayed learning
gain when those labels are available.

## Current ASSISTments Result

The first ASSISTments progression-reward replay uses AKT-backed predictions,
candidate sets of size 20, 10,000 replay events per seed, and seeds 11, 17, and
23.

| Metric | Value |
|--------|------:|
| Mean random baseline DR value | 0.6220 |
| Mean ProgressionValuePolicy DR value | 0.9431 |
| Mean DR uplift | +0.3212 |
| Across-seed uplift CI | [0.3146, 0.3278] |
| Mean target match rate | 0.0198 |
| Mean target effective sample size | 198 |

This is a positive offline result on the progression reward objective. It is
not yet a causal live-learning claim because the benchmark uses synthetic
candidate propensities and low target-policy coverage.

## Delayed-Gain Check

The stricter check uses `--reward-mode delayed_gain`, which rewards future
same-skill correctness improvement rather than the immediate progression proxy.
On ASSISTments, the current policy is not yet positive on that objective:

| Setting | Uplift |
|---------|-------:|
| `target_correct=0.70` | -0.0305 |
| `target_correct=0.90` | -0.0211 |
| `target_correct=0.95` | -0.0208 |

This is useful negative evidence. The next policy iteration should optimize
same-skill consolidation and delayed gain directly instead of only scoring
single-step stretch/progression value.
