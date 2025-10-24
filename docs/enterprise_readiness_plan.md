# Orchid Ranker Enterprise Readiness Plan

## Phase 0 – Baseline Assessment (Weeks 0‑2)
- ✅ Baseline documentation captured in `docs/enterprise_baseline_assessment.md`.
- Audit current architecture, testing coverage, and documentation; catalogue gaps that block SOC2/enterprise reviews.
- Prioritise customer-critical use cases (offline experiments, DP-enabled recommenders) and align internal stakeholders on target SLAs and compliance posture.
- Establish cross-functional tiger team (product, ML eng, infra, privacy, GTM) and cadence for steering updates.

## Phase 1 – Product Hardening (Weeks 3‑8)
- ✅ Stabilised public APIs with semantic versioning, deprecation policy, and support policy doc.
- ✅ Expanded automated coverage (recommender strategy validation, agentic smoke run, legacy compatibility) and added profiling harness.
- 🔄 Still to do: GPU/CPU test matrix and integration with CI coverage gates before release freeze.
- 📌 Maintain focus on being a Python-first library (pip installable), with deployment artefacts remaining optional add-ons.

## Phase 2 – Security & Privacy Foundations (Weeks 6‑12)
- ✅ Integrated Opacus-backed DP accountant and extended privacy documentation/threat model notes.
- ✅ Added role-based access enforcement to preprocessing CLI and JSONL audit logging hooks for DP updates.
- ✅ Drafted compliance artefacts (data retention, incident response, FERPA/GDPR alignment) and SIEM shipping utility.
- ✅ Automated CycloneDX SBOM generation and pip-audit scanning via GitHub workflow.
- ✅ Added environment-driven audit log forwarding for SIEM integration and documented pen-test preparation playbook.
- 🔄 Pending: external third-party pen-test execution and follow-up remediation.

## Phase 3 – Deployment & Observability (Weeks 10‑16)
- ✅ Docker image, Helm chart skeleton, and Terraform reference committed under `deploy/`.
- ✅ Connectors module with Snowflake, BigQuery, S3 streaming, and MLflow helpers (optional dependencies).
- ✅ Prometheus observability helpers (`orchid_ranker.observability`), metrics tests, and deployment guide (`docs/deployment.md`).
- 🔄 Pending: production dashboard templates and health-check endpoints for adaptive services.

## Phase 4 – Customer Success & Support Infrastructure (Weeks 14‑20)
- ✅ Drafted onboarding playbook, solutions-engineering rotation, support SLA, and FAQ under `docs/customer_success/`.
- ✅ Added design-partner pilot plan and seeded `examples/notebooks/` with pilot quickstart notebook.
- 🔄 Pending: staffing solutions engineering rotations and scheduling design-partner pilot execution.
- 🔄 Pending: collect testimonials/case studies and expand notebook gallery with partner-contributed content.

## Phase 5 – Packaging & Go-To-Market Enablement (Weeks 18‑24)
- ✅ Created GTM messaging one-pager, blog template, and license/distribution notes in `docs/gtm/`.
- 🔄 Pending: pricing/packaging work (if applicable), ROI calculator, datasheet, and case studies informed by pilot outcomes.
- 🔄 Pending: enablement sessions (demo scripts, proof-of-value playbook) once pilots complete.

## Continuous Workstreams
- Quarterly roadmap reviews and customer advisory councils.
- Ongoing security audits, dependency updates, and performance tuning.
- Community engagement (webinars, conference talks, open-source contributions) to sustain market visibility.
