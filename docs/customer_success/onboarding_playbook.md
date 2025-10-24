# Customer Onboarding Playbook (Draft)

## Goals
- Help new teams install Orchid Ranker via pip and reproduce core workflows within the first week.
- Document touchpoints between product, engineering, and customer champions.

## Week 0 – Kickoff
- Confirm stakeholders, success criteria, datasets to evaluate, and privacy posture.
- Share quickstart guide (`README.md`) and deployment overview (`docs/deployment.md`).
- Schedule recurring sync (weekly) for status and blockers.

## Week 1 – Environment Setup
- Customer installs `orchid-ranker` using pip (optionally extras: `agentic`, `viz`, `observability`, `connectors`).
- Run smoke tests:
  - `python -m pytest tests/test_recommender.py -k smoke` (customer copy of repo).
  - `orchid-evaluate --help` to confirm CLI availability.
- Provide sample notebooks from `examples/` demonstrating recommender fitting and evaluation.

## Week 2 – Dataset Ingestion
- Assist customer with preprocessing pipeline (`orchid-preprocess` CLI or direct module use).
- Validate schema using `orchid_ranker.preprocessing` validators.
- Configure RBAC roles aligned to customer team (see `orchid_ranker.security`).

## Week 3 – Experimentation
- Set up agentic simulations and benchmarking scripts.
- Configure DP settings per policy (`docs/privacy.md`).
- Ensure Prometheus metrics and audit logs are captured (if needed) for observability sign-off.

## Week 4 – Review & Next Steps
- Review metrics, compare against baseline targets, gather feedback.
- Capture deployment needs (Docker/Helm/Terraform) if transitioning to staging.
- Document outstanding gaps and plan next iteration.

## Resources
- Product manager owns onboarding timeline and feedback loop.
- ML engineer provides technical support and code walkthroughs.
- Security lead reviews DP/audit configuration where required.
