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
interactions; ~5,171 scored decisions per seed).

**The result is dominated by tracer quality** — this is the headline lesson.
The same benchmark gives opposite signs depending on how well the tracer is
trained:

| tracer | KT AUC | competence_blend | overall ρ | partial ρ (CI) | corr-only ρ |
|---|---|---|---|---|---|
| AKT undertrained (3 ep, d64) | 0.66 | 0.0 | −0.019 | −0.077 ([−0.10,−0.05]) | 0.003 |
| **AKT tuned (12 ep, d128)** | **0.73** | 0.0 | 0.038 | **+0.041** ([+0.008,+0.071]) | 0.061 |
| AKT tuned (12 ep, d128) | 0.73 | 0.5 | 0.010 | +0.032 | 0.061 |

`competence_blend=0.5` lowers the partial signal (+0.032 vs +0.041) under the
tuned tracer too — consistent with the undertrained run — so wiring tracer
competence into the reward does not help here and the shipped default stays 0.0.

With a **properly trained, above-baseline** tracer, the progression reward's
extra structure adds a small but **robustly positive** residual signal
(partial Spearman +0.041, CI excludes 0 across all seeds) — i.e. beyond raw
predicted correctness and competence, the ZPD terms carry a little genuine
signal about realized same-concept gain. The committed artifact reflects this
tuned run.

**But it still does not clear the bar:** the aggregate reward correlation
(0.038) is below correctness-only (0.061), so `evidenced` is 0/3. The reward
adds signal but does not (yet) beat ranking by predicted correctness alone.

Caveats: realized gain is a noisy observational proxy and |ρ| ≈ 0.04–0.08 is
small. The decisive takeaway is methodological — **evaluate the reward only
against a tracer that beats baseline**, and the honest current verdict is:
"the reward adds a small positive signal but is not yet evidenced to beat
correctness." Closing the gap needs a reward redesign and, ideally, an
online/counterfactual outcome signal rather than this observational proxy.

Related — KT prediction AUC on this data:

| config | item-mean | AKT | SAKT |
|---|---|---|---|
| 3 epochs, d_model 64 | 0.694 | 0.657 | 0.597 |
| 12 epochs, d_model 128, seq 100 | 0.694 | **0.730** | 0.658 |

At a small budget the neural tracers underperform the trivial item-mean
baseline; with adequate capacity/training the faithful **AKT beats it (0.730)**
and lands in the published ASSISTments-2009 range, with AKT > SAKT as expected.
So the faithful implementations are sound — the earlier below-baseline numbers
were undertraining, not a flaw. (Longer training degrades calibration: ECE rises
to ~0.11; apply `calibration.TemperatureScaler` post-hoc to restore it.)

## Interpreting results

- **Evidenced, positive partial Spearman:** the reward's pedagogical shape adds
  signal beyond correctness — the weights are defensible.
- **Positive overall but ~0 partial:** the correlation is driven by `p_correct`
  / `competence`; the stretch/ZPD terms are not pulling weight. Retune.
- **Negative per-term partial for a `+` term (or positive for a `-` penalty):**
  that term is counterproductive on this data.
