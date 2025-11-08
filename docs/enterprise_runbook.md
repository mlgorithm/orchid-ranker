# Orchid Ranker Enterprise Runbook

This runbook summarizes the operational knobs we currently support when deploying the agentic pipeline in an enterprise setting.

## 1. Release & CI gates

1. `pytest` — run `python -m pytest` (includes native scoring and SafeSwitch unit tests).
2. Safe smoke scenario — `./scripts/ci_safe_smoke.sh runs/ci-safe-smoke`. This runs a 60×120 MovieLens slice with `--safe-eb --safe-eb-dr`, verifies the SafeSwitch telemetry, and completes in under 2 minutes on CPU/MPS hardware.
3. GitHub Actions — `.github/workflows/ci.yaml` runs both steps above automatically on every push/PR so regressions are caught before release.

Embed both steps in CI to block regressions. The smoke run ensures the non-regression gate is wired each release.

## 2. Safe deployment knobs

- `--safe-eb --safe-eb-dr` — enables the SafeSwitch-DR gate, which mixes teacher/adaptive policies using a DR confidence sequence and acceptance floor. Tuning knobs:
  - `--safe-eb-pmin`, `--safe-eb-pstep`, `--safe-eb-accept-floor`.
- The orchestrator logs gate telemetry (`serve_policy`, `p`, `uplift_lcb`, `acc_lcb`) once per round inside each JSONL summary. Stream this into monitoring dashboards to verify non-regression live.

## 3. Monitoring & observability

- Timing logs — pass `--timing-log runs/foo/timing.jsonl --timing-rounds 5` to benchmark scripts to capture phase-level latency (candidate sampling, tower inference, decide, student interaction, train step). Ship these JSON lines to your telemetry backend to track regressions.
- JSONL metrics — each orchestrator round emits `round_summary` events with acceptance, accuracy, novelty, DP epsilon, and `safe_gate` telemetry.
- Prometheus — enable the optional `observability` extra in `pyproject.toml` to expose built-in counters via `prometheus-client` if you integrate the library in a long-running service.

## 4. Packaging & distribution

- Versioned builds — update `pyproject.toml` and tag releases; build with `python -m build` and publish via your internal package index.
- Use `scripts/bump_version.sh [major|minor|patch]` to bump the semantic version string safely before tagging a release.
- Docker — use the benchmark scripts as entrypoints (e.g., `python benchmarks/run_agentic_ml100k.py ...`) in your CI images to validate GPU/MPS compatibility.

## 5. Rollout strategy

1. **Shadow mode**: run SafeSwitch with `p_min` small (e.g., 0.05) so the adaptive student only serves a tiny fraction until uplift evidence appears.
2. **Canary**: set `accept_floor` to the production KPI target and monitor `safe_gate` telemetry. If the gate drops to teacher-only, pause the rollout.
3. **Ramp**: gradually increase `p_min` / `step_up` after the DR uplift lower bound stays positive for N rounds/users.

## 6. Troubleshooting

- Gate stuck on teacher? Inspect `safe_gate` telemetry. If `uplift_lcb` stays negative, pretrain/distill the student (`--funk-distill`) or run additional warmup steps. If `acc_lcb` dips below the floor, revisit recommendation diversity/novelty weights or reduce exploration.
- Slow rounds? Use the timing JSONL to see whether candidate sampling, tower inference, or training dominates. Adjust `min_candidates`, enable `--native-score`, or reduce `train_steps_per_round` in smoke/CI runs.

For deeper integration (custom telemetry, policy APIs, or DP accountants), contact the maintainer listed in `pyproject.toml`.

## 7. Security & compliance checklist

| Item | Status | Notes |
| --- | --- | --- |
| Threat model for DP/telemetry | 🔲 TODO | Document data flows (training logs, telemetry exports). |
| DP parameter guidance | 🔲 TODO | Provide recommended `sigma`, `sample_rate`, `per_round_eps_target` for typical deployments. |
| Secrets management | 🔲 TODO | Define how API keys / DP noise seeds are stored (e.g., Vault, KMS). |
| Audit logging | 🔲 TODO | Integrate `orchid_ranker.security.AuditLogger` with enterprise SIEM. |
| Pen-test / security review | 🔲 TODO | Schedule annual review; capture findings here. |
| Compliance alignment | 🔲 TODO | Map modules to SOC2/GDPR controls; document retention policies. |

## 8. Monitoring dashboards

- **SafeSwitch dashboard:** track `gate.p_used`, `gate.lcb`, and `gate.acc_lcb` per scenario. Alert if `p_used` stays at 0 for >N rounds or if `acc_lcb` < floor.
- **Latency dashboard:** ingest timing JSONL (`candidate_sampling`, `tower_infer`, `decide`, `student_interact`, `train_step`, `warmup_sync`) into Prometheus/Grafana. Set SLOs per phase.
- **DP budget dashboard:** display `eps_cum` and per-round `dp` metrics from the JSONL logs; alert if budget exceeds thresholds.
