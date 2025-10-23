# Differential Privacy Notes

Orchid Ranker ships with a configurable DP-SGD training pathway. When
`dp_cfg` includes `"engine": "per_sample"` (the default), updates to the
`TwoTowerRecommender` perform per-example gradient clipping and add Gaussian
noise before stepping the optimiser. Privacy loss is tracked via an RDP-based
accountant and exposed through the metrics returned from `update()`.

Example configuration:

```python
dp_cfg = {
    "enabled": True,
    "engine": "per_sample",
    "noise_multiplier": 1.0,
    "sample_rate": 0.02,
    "max_grad": 1.0,
    "delta": 1e-5,
}
model = TwoTowerRecommender(..., dp_cfg=dp_cfg)
```

The legacy aggregated-noise pathway is still available via
`"engine": "legacy"`, but delivers weaker guarantees and is kept only for
backwards compatibility.

**Important caveats**

- The per-sample implementation loops over each example in the feedback slate
  and is therefore intended for small candidate sets (as used in the agentic
  simulator). Large-batch production training should adopt a high-performance
  DP optimiser (e.g., Opacus) once integrated.
- Privacy guarantees depend on the chosen hyperparameters. Monitor
  `epsilon_delta` and `epsilon_cum` returned from `update()` and ensure they fit
  organisational policies.
- Remember to sanitise raw logs and upstream preprocessing steps; DP-SGD only
  protects the model training loop.

When publishing results or deploying to users, verify the chosen parameters by
re-running the accountant on your workload and consider migrating to Opacus or
another certified implementation for large-scale deployments.
