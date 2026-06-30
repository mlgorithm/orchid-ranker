# Security & Compliance Notes

## SBOM & Vulnerability Scanning
- `cyclonedx-py -e -o sbom.xml --format xml` generates an SBOM for the active Python environment (CycloneDX XML). Uses `cyclonedx-bom<4`/`cyclonedx-python-lib<4` for compatibility.
- `pip-audit --format json --output pip-audit.json` checks Python packages for known CVEs. Exit code may be non-zero when vulnerabilities are found; artefact is still uploaded for review.
- GitHub workflow `.github/workflows/security.yml` now runs on every push/PR to `main` (in addition to manual dispatch) and uploads artefacts under `security-reports/`.

## Role-Based Access Control
- `orchid_ranker.security.AccessControl` is a library primitive: construct it with a policy (the bundled default is `orchid_ranker.security.DEFAULT_POLICY`) and call it from your own service to authorize actions per role. It is not wired into the `orchid-serve` CLI or the ranking path; the integrator is responsible for enforcing it at their API boundary.
- The `orchid-serve` CLI exposes only health/metrics endpoints and has no `--role` flag (its flags are `--host`, `--port`, `--metrics-port`, `--health-port`, `--no-metrics`, `--ready-on-start`).

## Audit Logging
- `AuditLogger` emits JSONL audit records and is a library primitive you wire into your own pipeline. The experimental `TwoTowerRecommender.update()` (not part of the public `__all__`) is the only built-in caller: when an audit logger is attached it writes `update` events capturing the training loss. The flagship `AdaptiveRanker`/`AdaptiveLearningEngine` APIs do not emit audit events themselves; call `AuditLogger.log_event(...)` from your service to record their decisions.
- Configure automatic forwarding with environment variables `ORCHID_AUDIT_ENDPOINT`, `ORCHID_AUDIT_API_KEY`, and `ORCHID_AUDIT_TIMEOUT` (seconds). `AuditLogger.from_env()` builds a logger that posts each event to the configured SIEM endpoint.
- Use `scripts/ship_audit_logs.py` to forward JSONL audit streams to a SIEM/Webhook endpoint in batch or cron workflows.

## Compliance Artefacts
- Data retention policy: `docs/compliance/data_retention.md`
- Incident response playbook: `docs/compliance/incident_response.md`
- FERPA/GDPR alignment notes: `docs/compliance/ferpa_gdpr_alignment.md`
- Third-party pen-test preparation: `docs/security/third_party_pen_test.md`
