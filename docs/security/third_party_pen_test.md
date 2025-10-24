# Third-Party Penetration Test Prep (Draft)

## Objectives
- Validate security posture of Orchid Ranker components before enterprise launch.
- Assess RBAC, audit logging, differential privacy safeguards, and deployment configurations.

## Scope
- Python package and CLI surfaces (preprocessing, agentic experiments, evaluator).
- API surface area when embedded into customer platforms (documented in integration guide).
- Infrastructure templates (containers, deployment manifests) once available in Phase 3.

## Deliverables for Vendor
- Architecture overview diagram and data flow description.
- SBOM (`security-reports/sbom.json`) and vulnerability report (`security-reports/pip-audit.json`).
- Security configuration guide (`docs/security.md`) and compliance artefacts (`docs/compliance/`).
- Sample audit log output and forwarding configuration (env variables, `scripts/ship_audit_logs.py`).

## Vendor Selection Criteria
- Experience with privacy-preserving ML or EdTech platforms.
- Ability to perform code review, dependency analysis, and runtime testing.
- Willingness to sign mutually agreeable NDAs and provide detailed remediation report.

## Timeline
- RFP issued: Phase 3 start.
- Testing window: 2 weeks (including fix validation).
- Final report & remediation plan: within 10 business days of test completion.

## Internal Responsibilities
- Product Manager: coordinate vendor communications and scheduling.
- Security Lead: primary point of contact, triage findings, track remediation.
- Engineering: allocate cycles for fixes and retest support.

## Post-Test Actions
- Merge remediation work into roadmap.
- Update documentation (security guide, compliance docs) with lessons learned.
- Share executive summary with design partners/customers under NDA when appropriate.
