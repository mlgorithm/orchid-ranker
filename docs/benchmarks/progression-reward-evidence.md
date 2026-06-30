# Progression Reward Evidence

The progression policy ranks items by a hand-designed reward
(`orchid_ranker.progression_reward.expected_progression_reward`) whose weights
encode pedagogical priors: zone-of-proximal-development mastery gain, stretch
fit, and easy/hard/repetition penalties. Those weights are the core of Orchid's
claim that it optimizes *learning progress* rather than immediate correctness.

This benchmark asks two questions that claim depends on:

1. Does the predicted progression reward correlate with **realized** learning
   gain, *beyond* what naive correctness and current ability already explain?
2. Which individual reward terms carry that signal, and which are dead or
   counterproductive?

It is different from the other benchmarks:

- KT prediction asks "can the tracer predict the next correctness label?"
- Policy OPE asks "would the policy's chosen item beat a baseline under logged
  replay?"
- **This** asks "is the reward function itself measuring something real?"

## Realized gain (observational proxy)

For each held-out decision, realized gain is the normalized same-concept
improvement over a future window:

```
gain = future_same_concept_accuracy - prior_same_concept_accuracy
realized = clip(0.5 + 0.5 * gain, 0, 1)        # monotone in gain
```

computed by `attach_delayed_gain_rewards`. Decisions with no future
same-concept attempts are dropped.

Public tutoring logs rarely include true platform propensities, so this is an
**observational** proxy, not a causal estimate. To reduce confounding we never
report a bare correlation alone:

- **Overall Spearman** of predicted reward vs realized gain (context only).
- **Partial Spearman** of predicted reward vs realized gain holding `p_correct`
  and `competence` fixed (rank-residual method). This is the headline number:
  it isolates whether the reward's pedagogical *shape* adds signal beyond
  correctness and current ability.
- **Within-competence-bin stratified buckets**: inside each competence quartile,
  bucket by predicted reward and report mean realized gain. Monotonic-increasing
  buckets within strata = robust to the competence confound.

Both headline correlations carry a bootstrap 95% CI.

## Per-term diagnosis

Each `ProgressionRewardBreakdown` term (`mastery_gain`, `stretch_fit`,
`difficulty_bonus`, `easy_penalty`, `hard_penalty`, `repetition_penalty`,
`expected_outcome_value`) is correlated with realized gain, raw and partial.
Each term is annotated with the sign it carries in the reward sum; a term whose
realized-gain correlation has the *wrong* sign relative to that is pulling the
reward away from learning gain and is a candidate for retuning.

## Ablations and verdict

The predicted reward must beat correctness-only (`p_correct`) and random at
predicting realized gain. The run reports a verdict:

> **evidenced** iff the partial Spearman CI excludes 0 (lower bound > 0) **and**
> the overall Spearman is at least the correctness-only Spearman.

If not evidenced, the per-term table indicates which weights to retune. A
"not evidenced" result is a real finding, not a benchmark failure.

## Run

CI smoke (seconds, synthetic data) is covered by
`tests/test_progression_reward_evidence.py`. For a credible artifact, run on
full chronological ASSISTments-style logs:

```bash
PYTHONPATH=src python benchmarks/progression_reward_evidence.py \
  --data data/assistments_kt/interactions.csv \
  --concept-col skill_id \
  --item-difficulty-col difficulty \
  --model akt \
  --test-fraction 0.2 \
  --future-window 5 \
  --competence-bins 4 \
  --reward-buckets 5 \
  --bootstrap 300 \
  --seeds 11 17 23 \
  --output benchmarks/results_progression_reward_evidence_assistments.json
```

## Output

JSON with a per-seed `runs` block and an aggregated `summary`:

- `overall_spearman` / `partial_spearman_given_pcorrect_competence` — estimate + bootstrap CI
- `ablations` — correctness-only, competence-only, random Spearman
- `per_term` — raw and partial correlation per reward term, with `sign_in_reward`
- `stratified` — per-competence-bin bucket means + within-stratum monotonicity
- `verdict` — `evidenced` flag and the rule applied
- `summary` — seed-averaged headline numbers and `evidenced_fraction`

## Measured results (ASSISTments 2009, faithful AKT, 3 seeds)

Run with `--model akt` on the full ASSISTments 2009 skill-builder data (~398k
interactions; ~5,171 scored decisions per seed). These are the honest current
numbers — they do **not** support the progression-reward claim:

| competence_blend | overall ρ | partial ρ (CI) | corr-only ρ | evidenced |
|---|---|---|---|---|
| 0.0 (default) | −0.019 | **−0.077** ([−0.10, −0.05]) | 0.003 | 0 / 3 |
| 0.5 | −0.060 | −0.104 | 0.003 | 0 / 3 |

The partial Spearman is small but **robustly negative** (CI excludes 0 across
all seeds): after controlling for predicted correctness and competence, the
reward's extra structure mildly *anti*-predicts realized same-concept gain.
Blending tracer competence in (`competence_blend=0.5`) makes it worse, so the
shipped default stays 0.0.

Caveats: realized gain is a noisy observational proxy and |ρ| ≈ 0.08 is small.
But the result is stable and signed, and it **reversed** an earlier weakly
positive number measured against a less-faithful, below-baseline KT model — so
the safe reading is "no credible evidence the current reward predicts learning
gain." The reward weights need a redesign grounded in a stronger outcome signal
(ideally an online/counterfactual experiment), not just retuning.

Related: on this data the lightweight in-repo neural tracers underperform the
item-mean baseline (AUC: item-mean 0.69, AKT 0.66, SAKT 0.60) at 3 epochs /
d_model 64 — they learn and rank AKT > SAKT as expected, but are undertrained
relative to published KT results and a trivial baseline.

## Interpreting results

- **Evidenced, positive partial Spearman:** the reward's pedagogical shape adds
  signal beyond correctness — the weights are defensible.
- **Positive overall but ~0 partial:** the correlation is driven by `p_correct`
  / `competence`; the stretch/ZPD terms are not pulling weight. Retune.
- **Negative per-term partial for a `+` term (or positive for a `-` penalty):**
  that term is counterproductive on this data.
