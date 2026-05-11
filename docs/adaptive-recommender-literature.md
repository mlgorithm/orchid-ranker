# Adaptive Recommender Literature Notes

This note records the current research direction for Orchid's adaptive-learning
ranker. It focuses on techniques that can be adopted in the library and tested
with the existing ASSISTments/EdNet benchmark path.

## What The Literature Implies

### 1. Standardized KT Evidence Matters More Than Model Count

[pyKT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/75ca2b23d9794f02a92449af65a57556-Abstract-Datasets_and_Benchmarks.html)
is the right benchmark reference point because it emphasizes standardized
preprocessing, comparable protocols, and leakage control. Its empirical warning
is directly relevant to Orchid: many KT improvements are small once evaluation
is made consistent.

Adoption:

- Keep the ASSISTments/EdNet preprocessing and time-ordered replay benchmarks
  as first-class artifacts.
- Treat new KT backbones as benchmarked tracers, not as marketing claims.
- Add SimpleKT next, because it is explicitly designed as a strong simple KT
  baseline.

### 2. AKT And SimpleKT Are The Best Near-Term KT Targets

[AKT](https://arxiv.org/abs/2007.12324) combines attention with cognitive and
psychometric structure, including monotonic attention and Rasch-style item
variation. That matches Orchid's direction: predicted correctness should be
interpretable enough to support policy decisions.

[SimpleKT](https://arxiv.org/abs/2302.06881) is important because it shows that
a simpler Rasch-inspired attention baseline can be hard to beat across public
datasets. Orchid should not add heavier KT models before SimpleKT parity is in
place.

Adoption:

- Keep `AKTTracer` as the current strongest in-repo tracer.
- Add `SimpleKTTracer` before SAINT/DTransformer-style variants.
- Continue exposing pyKT bridge outputs so external model-zoo checkpoints can
  be evaluated inside Orchid's policy/OPE layer.

### 3. Delayed-Gain Policy Is An Offline Bandit Problem

The delayed-gain objective is not just a KT AUC problem. It is a logged-policy
evaluation problem where the target policy can choose items that the historical
policy rarely served.

[Doubly robust policy evaluation](https://arxiv.org/abs/1103.4601) is the right
estimator family when both propensities and reward/value models matter. The
later sequential OPE formulation by Jiang and Li makes the same bias/variance
tradeoff explicit for RL settings, and
[MRDR](https://proceedings.mlr.press/v80/farajtabar18a.html) points to the next
step: train the direct model for OPE usefulness, not only ordinary prediction
loss.

Adoption:

- Keep IPS/SNIPS/DR side by side in benchmark artifacts.
- Do not trust a learned reward model until DR and replay diagnostics agree.
- `fit_delayed_gain_reward_model_from_frame` now supports MRDR-style squared
  importance weighting when target/logging propensities are available, plus
  cross-fitted diagnostics for public logged-data benchmarks.

### 4. Calibration Is Necessary But Not Sufficient

[Calibration work](https://proceedings.mlr.press/v70/guo17a.html) motivates
explicit calibration metrics for value models. Orchid adopted isotonic
calibration for `DelayedGainRewardModel`, but the first ASSISTments check still
failed under DR. That means marginal calibration is not enough; the value model
must be calibrated on the action distribution induced by the target policy.

Adoption:

- `DelayedGainRewardModel` now reports validation RMSE, MAE, ECE, and whether
  post-hoc calibration was fitted.
- The reward-model diagnostic benchmark shows the core failure: validation RMSE
  stays around 0.098, but target-policy matched actions are overpredicted by
  about +0.085 on the seed-11 ASSISTments slice.
- Cross-fitted diagnostics and support-inverse weighting did not solve the
  target-action bias, so calibration remains diagnostic rather than solved.

### 5. Offline RL Pessimism Is The Right Safety Bias

[Conservative Q-Learning](https://papers.nips.cc/paper_files/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
addresses the core failure mode Orchid saw: overestimated values under
distribution shift. Orchid should not implement full CQL yet, but it should
borrow the principle: prefer lower-confidence value estimates and penalize
unsupported actions.

Adoption:

- `SupportConstrainedDelayedGainPolicy` penalizes low-support actions and
  improves target-policy coverage.
- The first DR result is negative, so the pessimistic support idea is useful but
  the direct reward model is not ready.
- Next: behavior-regularized ranking and MRDR/cross-fitted value models.

## Current Adopted Techniques

| Technique | Status | Evidence |
|-----------|--------|----------|
| Time-ordered KT replay | Adopted | ASSISTments KT benchmark |
| AKT-inspired tracer | Adopted | Best in current ASSISTments KT run |
| Progression reward policy | Adopted | Positive progression-reward OPE |
| Delayed-gain prior policy | Adopted | Break-even/slightly positive SNIPS |
| Direct delayed-gain reward model | Experimental | Good validation error, target-action bias |
| Isotonic reward-model calibration | Experimental | Good validation ECE, target-action overprediction remains |
| Support-constrained ranking | Experimental | Coverage improves, value model not ready |
| Cross-fitted reward diagnostics | Adopted | Exposes target-action bias missed by validation RMSE |
| MRDR-style weighted fitting API | Adopted | Available when target/logging propensities are known |

## Next Algorithmic Work

1. Add `SimpleKTTracer` and benchmark it against AKT.
2. Use the reward-model diagnostics benchmark as the gate for any learned value
   policy: target-action bias must shrink before DR policy claims are trusted.
3. Collect or simulate real target/logging propensities so MRDR weighting can
   be used with actual action importance weights, not support proxies.
4. Keep `DelayedGainValuePolicy` as the current best strict delayed-gain policy
   until a learned value model passes DR.
5. Only then add short-horizon planning; otherwise planning will amplify a
   miscalibrated delayed-gain model.
