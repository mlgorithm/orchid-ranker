# Differential Privacy Tutorial

## 1. Quick start (TwoTowerRecommender)

```python
dp_cfg = {
    "enabled": True,
    "engine": "opacus",
    "noise_multiplier": 1.0,
    "sample_rate": 0.02,
    "max_grad_norm": 1.0,
    "delta": 1e-5,
}
model = TwoTowerRecommender(..., dp_cfg=dp_cfg)
```

## 2. Enabling in your training entrypoint

If you expose DP flags through your own CLI or service wrapper, pass the same
core parameters:

- `dp_enabled`
- `noise_multiplier`
- `sample_rate`
- `max_grad_norm`
- `delta`

## 3. Monitoring epsilon

- Each round summary includes `dp.epsilon_cum`.
- Plot epsilon vs rounds to ensure you remain under target budget.

## 4. Using SafeSwitch + DP

Combine SafeSwitch controls with DP configs to track both privacy budget and
non-regression evidence. Production privacy and rollout claims still depend on
the selected parameters, workload, and review process.

## 5. Tuning tips

| Parameter | Effect |
| --- | --- |
| `noise_multiplier` | Higher noise => stronger privacy, lower accuracy |
| `sample_rate` | Should match batch_size / dataset_size |
| `max_grad_norm` | Clip per-sample gradients to reduce sensitivity |

## 6. Accountant utilities

```python
from orchid_ranker.dp import build_accountant
acc = build_accountant(noise_multiplier=1.0, sample_rate=0.02, delta=1e-5)
acc.step(10)
print(acc.epsilon)
```
