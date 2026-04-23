# Secure Deployment Walkthrough

## 1. RBAC & access control

- Use the `orchid_ranker.security` module to enforce per-role permissions (e.g., analysts vs admins).
- Restrict CLI access by wrapping entrypoints in your internal tooling with role checks.

## 2. Audit logging

- Configure the JSONL logger path via config (`cfg.log_path`).
- Forward logs to your SIEM using `fluent-bit` or a custom shipper.
- Include `safe_gate` telemetry, DP stats, and any reject reasons.

## 3. SafeSwitch configuration

- Set `--safe-eb --safe-eb-dr --safe-eb-accept-floor <target>` on orchestration services or validation runs.
- Monitor `safe_gate` telemetry (stored per round) for p-values and acceptance LCBs.

## 4. Differential Privacy

- Follow the DP tutorial (`docs/tutorial_dp.md`) to set noise/clipping parameters.
- Track `epsilon_cum` in telemetry; alert if it exceeds the declared budget.

## 5. Compliance artefacts

- Data retention policy: see `docs/compliance/data_retention.md`.
- FERPA/GDPR alignment: see `docs/compliance/ferpa_gdpr_alignment.md`.
- Incident response: documented in `docs/compliance/incident_response.md`.

## 6. Deployment steps

1. Run `./scripts/ci_safe_smoke.sh` in the target environment.
2. Enable RBAC and DP settings via env vars/secrets.
3. Start metrics exporter (`orchid_ranker.observability.MetricsServer`).
4. Verify dashboards/alerts as described in `docs/tutorial_observability.md`.
5. Log audit trail for go-live decision.
