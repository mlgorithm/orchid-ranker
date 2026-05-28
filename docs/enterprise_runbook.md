# Orchid Ranker Enterprise Runbook

This runbook summarizes operational checks for embedding Orchid's
adaptive-learning stack in an enterprise application.

## Release And CI Gates

1. `python -m pytest` for unit and integration coverage.
2. `python examples/adaptive_learning_quickstart.py` for an install-level
   adaptive fit/rank/observe smoke.
3. `PYTHONPATH=src python benchmarks/adaptive_efficiency_benchmark.py ...` for
   a JSON + Markdown evidence artifact when preparing releases.

GitHub Actions runs linting, type checks, tests with coverage, docs, install
contracts, lower-bound installs, and distribution checks.

## Safe Deployment Knobs

- Use `evaluate_logged_policy`, `compare_logged_policies`,
  `bootstrap_logged_policy`, and `evaluate_rollout_gate` before enabling a new
  learned policy.
- Keep a reviewed prerequisite/difficulty fallback policy for guardrail halts.
- Log candidate sets, chosen item, scores, propensities, policy version, and
  context hash for every served recommendation.

## Monitoring

- Track progression gain, proficiency coverage, difficulty appropriateness,
  rolling correctness/acceptance, and guardrail halt state.
- Export Prometheus metrics with the `observability` extra when embedding
  Orchid in a long-running service.
- Store OPE and benchmark artifacts next to release candidates so reviewers can
  reproduce claims.

## Packaging

- Versioned builds: update `pyproject.toml`, run `python -m build`, then publish
  via the configured package index.
- Use `scripts/bump_version.sh [major|minor|patch]` to bump the semantic version
  string before tagging.

## Troubleshooting

- Low OPE support: increase logged exploration or restrict the target policy to
  candidate regions with enough overlap.
- Guardrail halts: inspect progression gain, sequence adherence, and stretch
  fit before increasing exploration.
- Slow adaptive ranking: profile KT inference, candidate generation, and OPE
  artifact generation separately.

## Security And Compliance Checklist

| Item | Status | Notes |
| --- | --- | --- |
| Threat model | Maintained | See `docs/security/threat_model.md`. |
| Audit logging | Available | Integrate `orchid_ranker.security.AuditLogger` with enterprise SIEM. |
| DP parameter guidance | In progress | Tune epsilon/delta per deployment and document budget ownership. |
| Data retention | Maintained | See `docs/compliance/data_retention.md`. |
| FERPA/GDPR alignment | Maintained | See `docs/compliance/ferpa_gdpr_alignment.md`. |
