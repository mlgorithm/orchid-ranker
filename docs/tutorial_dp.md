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

## 2. Enabling via CLI

```bash
PYTHONPATH=src python benchmarks/run_agentic_ml100k.py \
  --rounds 20 --dp-enabled --dp-noise 1.0 --dp-sample-rate 0.02 \
  --safe-eb --safe-eb-dr --log-dir runs/dp-safe
```

See `benchmarks/run_agentic_ml100k.py --help` for all DP-related switches.

## 3. Monitoring epsilon

- Each round summary includes `dp.epsilon_cum`.
- Plot epsilon vs rounds to ensure you remain under target budget.

## 4. Using SafeSwitch + DP

Combine `--safe-eb` flags with DP configs to guarantee both privacy and non-regression.

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
