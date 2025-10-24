# Security & Compliance Notes

## SBOM & Vulnerability Scanning
- `cyclonedx-py -e -o sbom.xml --format xml` generates an SBOM for the active Python environment (CycloneDX XML). Uses `cyclonedx-bom<4`/`cyclonedx-python-lib<4` for compatibility.
- `pip-audit --format json --output pip-audit.json` checks Python packages for known CVEs. Exit code may be non-zero when vulnerabilities are found; artefact is still uploaded for review.
- GitHub workflow `.github/workflows/security.yml` now runs on every push/PR to `main` (in addition to manual dispatch) and uploads artefacts under `security-reports/`.

## Role-Based Access Control
- CLI tools accept a `--role` flag (defaults to `ml_engineer`). Policy defined in `orchid_ranker.security.ACCESS_POLICY`.

## Audit Logging
- `AuditLogger` emits JSONL audit records. `TwoTowerRecommender.update()` writes `dp_update` events capturing epsilon deltas, noise multiplier, and total DP steps.
- Configure automatic forwarding with environment variables `ORCHID_AUDIT_ENDPOINT`, `ORCHID_AUDIT_API_KEY`, and `ORCHID_AUDIT_TIMEOUT` (seconds). `AuditLogger.from_env()` builds a logger that posts each event to the configured SIEM endpoint.
- Use `scripts/ship_audit_logs.py` to forward JSONL audit streams to a SIEM/Webhook endpoint in batch or cron workflows.

## Compliance Artefacts
- Data retention policy: `docs/compliance/data_retention.md`
- Incident response playbook: `docs/compliance/incident_response.md`
- FERPA/GDPR alignment notes: `docs/compliance/ferpa_gdpr_alignment.md`
- Third-party pen-test preparation: `docs/security/third_party_pen_test.md`
