# Orchid Ranker Messaging One-Pager (Draft)

## Elevator Pitch
Open-source Python library delivering adaptive learning recommenders, pairing Surprise-style APIs with agentic simulators, differential privacy, and observability hooks.

## Key Messages
- **Developer-first**: install via pip, integrate in notebooks/services like scikit-learn or Surprise.
- **Adaptive & Privacy-aware**: built-in agentic simulations, DP engines (per-sample + Opacus), audit logging.
- **Enterprise-ready options**: optional connectors (Snowflake, BigQuery, S3, MLflow), deployment guides (Docker, Helm).

## Differentiators
- Combines offline-to-online workflows (preprocessing, simulators, dashboards) in one library.
- Privacy tooling & audit trails included out-of-the-box for education customers.
- Transparent, open-source roadmap with active experiment documentation.

## Proof Points
- Benchmarks vs ALS/bandit baselines (see `docs/benchmarking.md`).
- Customer pilot plan and notebooks ready for design partners.
- Security assessments (SBOM, pip-audit, audit logging) baked into CI/docs.

## Calls to Action
- Developers: `pip install orchid-ranker` and run quickstart notebook.
- Partners: join pilot program (see `docs/customer_success/design_partner_pilot.md`).
- Contributors: explore roadmap issues, contribute connectors, share experiments.
