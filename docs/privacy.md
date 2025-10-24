# Differential Privacy Notes

Orchid Ranker ships with a configurable DP-SGD training pathway. When
`dp_cfg` includes `"engine": "per_sample"` (the default), updates to the
`TwoTowerRecommender` perform per-example gradient clipping and add Gaussian
noise before stepping the optimiser. Privacy loss is tracked via an RDP-based
accountant and exposed through the metrics returned from `update()`.

From 0.2.0 the library also supports an Opacus-backed engine:

```python
dp_cfg = {
    "enabled": True,
    "engine": "opacus",
    "noise_multiplier": 1.0,
    "sample_rate": 0.02,
    "delta": 1e-5,
}
```

The Opacus pathway relies on `opacus.accountants.analysis.rdp` for privacy
accounting, yielding the same guarantees used in production-grade PyTorch DP
pipelines. Set `engine="opacus"` when your dependency stack includes
`opacus>=1.5` (installed automatically via the `agentic` extra).

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

- The legacy per-sample engine loops over each example in the feedback slate
  and is therefore intended for small candidate sets (as used in the agentic
  simulator). Select the Opacus engine for large-batch production training.
- Privacy guarantees depend on the chosen hyperparameters. Monitor
  `epsilon_delta` and `epsilon_cum` returned from `update()` and ensure they fit
  organisational policies.
- Remember to sanitise raw logs and upstream preprocessing steps; DP-SGD only
  protects the model training loop. The threat model assumes a trusted
  orchestration pipeline, authenticated access to logs, and secure handling of
  epsilon/delta telemetry.
- For compliance programmes (e.g., FERPA/GDPR), export the accountant outputs
  alongside experiment metadata. Audit logs should capture dp_cfg parameters,
  epsilon deltas per update, and operator identity.

When publishing results or deploying to users, verify the chosen parameters by
re-running the accountant on your workload and consider migrating to Opacus or
another certified implementation for large-scale deployments.
