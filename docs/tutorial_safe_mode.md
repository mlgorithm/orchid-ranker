# Tutorial: Running Safe Mode Benchmarks

This walk-through (see also `docs/tutorials/safe_mode.ipynb` for an interactive version) shows how to:

1. Run the standard agentic benchmark on MovieLens 100K.
2. Exercise the SafeSwitch-DR gate via the safe smoke script.
3. Inspect the resulting logs/telemetry to verify non-regression guarantees.

## Prerequisites

- Python 3.10+ with `pip install ".[agentic]"`.
- `PYTHONPATH` pointing to `src/` when running scripts locally.

## 1. Baseline agentic run (optional)

```bash
PYTHONPATH=src python benchmarks/run_agentic_ml100k.py \
  --rounds 20 --top-users 200 --top-items 400 --top-k 6 --dim 16 \
  --log-dir runs/ml100k-baseline --quick
```

This produces `fixed.jsonl` and `adaptive.jsonl` with per-round metrics. The
fixed run is the frozen baseline used when the adaptive policy is unsafe.

## 2. SafeSwitch smoke run

```bash
./scripts/run_ml100k_safe_smoke.sh runs/ml100k-safe-smoke
```

- Uses a 60×120 slice for 5 rounds (`--safe-eb --safe-eb-dr --skip-fixed`).
- Writes `runs/ml100k-safe-smoke/adaptive.jsonl` and `timing.jsonl`.
- Suitable for CI (completes in <2 minutes on CPU/MPS machines).

## 3. Inspecting telemetry

Each `round_summary` entry now includes:

```json
"safe_gate": {
  "serve_policy": "teacher",
  "p_used": 0.05,
  "lcb": -15.1,
  "acc_lcb": -2.3,
  "p": 0.05
}
```

- `serve_policy`: which policy served the slate. The legacy value `teacher`
  means the frozen baseline; `adaptive` means the adaptive policy.
- `p_used` / `p`: mix probability (before & after updates).
- `lcb`: DR uplift lower bound (needs to cross > 0 for the adaptive policy to ramp).
- `acc_lcb`: acceptance-rate lower bound; if below the floor, the gate falls back to the frozen baseline.

Plotting these values over rounds demonstrates the safety guarantee.

## 4. Automating in CI

Add the following step to your pipeline:

```yaml
- name: Safe mode smoke test
  run: ./scripts/ci_safe_smoke.sh runs/ci-safe-smoke
```

The script fails if the `safe_gate` block is missing, ensuring releases keep the non-regression guard wired.

## 5. Next steps

- Increase `--rounds` / `--top-users` to run longer safe experiments once you have a GPU-equipped machine.
- Enable `--native-score` to exercise the optional C++ scoring kernel.
- Use `--timing-log ... --timing-rounds 5` to capture phase-level latency for performance investigations.
- Continue with:
  - Dataset ingestion guide: `docs/tutorial_data_ingestion.md`
  - Pareto/PC-EB controller: `docs/tutorial_pc_eb.md`
  - Differential privacy deep dive: `docs/tutorial_dp.md`
  - Observability and dashboards: `docs/tutorial_observability.md`
  - Performance tuning: `docs/performance_playbook.md`
  - Connectors & deployment: `docs/connectors_deployment.md`
  - Secure deployment walkthrough: `docs/security_walkthrough.md`
