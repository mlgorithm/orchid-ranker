# Differential Privacy Tutorial

Differential privacy is an opt-in safety layer for adaptive-learning deployments
that update learner models from sensitive interaction data. It belongs beside
knowledge tracing, progression metrics, and rollout guardrails; it is not a
generic recommender benchmark feature.

## 1. Choose a DP preset

```python
from orchid_ranker.dp import get_dp_config

dp_cfg = get_dp_config("eps_1")
dp_cfg["engine"] = "per_sample"  # use "opacus" when opacus is installed
```

The preset expands to the fields consumed by the PyTorch-backed adaptive
components: `enabled`, `noise_multiplier`, `sample_rate`, `delta`, and
`max_grad_norm`.

## 2. Attach DP to an adaptive model

```python
from orchid_ranker.agents import TwoTowerRecommender

model = TwoTowerRecommender(
    num_users=500,
    num_items=2_000,
    user_dim=16,
    item_dim=32,
    dp_cfg=dp_cfg,
)
```

DP is disabled unless `dp_cfg["enabled"]` is true. Private updates clip
per-example gradients, add calibrated Gaussian noise, and return
`epsilon_delta` and `epsilon_cum` telemetry from `update()`.

## 3. Monitor the privacy budget

```python
from orchid_ranker.agents.simple_dp import SimpleDPConfig
from orchid_ranker.dp_accountant import build_accountant

cfg = SimpleDPConfig(
    enabled=True,
    noise_multiplier=1.2,
    sample_rate=0.02,
    max_grad_norm=1.0,
    delta=1e-5,
)
accountant = build_accountant("per_sample", cfg)
epsilon_delta, epsilon_total = accountant.step(10)
```

Store the cumulative epsilon alongside adaptive rollout metrics so operators can
verify that learner updates remain inside the approved privacy budget.

## 4. Combine DP with adaptive guardrails

Use DP for privacy and `SafeSwitchDR` or progression guardrails for outcome
safety. The two mechanisms answer different questions: DP limits what training
can reveal about a learner, while guardrails decide whether the adaptive policy
is improving learning outcomes enough to keep serving.

## 5. Tuning tips

| Parameter | Effect |
| --- | --- |
| `noise_multiplier` | Higher noise gives stronger privacy and lower update signal |
| `sample_rate` | Should match batch size divided by the update population |
| `max_grad_norm` | Clips per-example gradients before noise is added |
| `delta` | Failure probability for the chosen privacy guarantee |
