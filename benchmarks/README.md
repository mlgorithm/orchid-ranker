# Benchmarks

Benchmark scripts live here; local output should not.

## Local runs

Write exploratory artifacts under the ignored `artifacts/` directory:

```bash
mkdir -p artifacts/benchmarks

PYTHONPATH=src python benchmarks/validate_logged_policy.py \
  --data artifacts/decisions.jsonl \
  --policy-version adaptive-ranker-v1 \
  --output artifacts/benchmarks/policy-validation.json \
  --report-md artifacts/benchmarks/policy-validation.md \
  --require-pass
```

`decisions.jsonl` must contain completed records exported from
`ranker.decision_log_frame(completed_only=True)`. JSON Lines preserves exact
candidate lists and logged action probabilities. The command compares the
versioned logged adaptive policy with a static baseline using a chronological
split and user-cluster bootstrap interval.

This keeps `git status` readable and prevents an exploratory result from
looking like reviewed evidence. See `docs/benchmarks/credibility.md` before
interpreting a pass.

## Committed evidence

Add a result to `benchmarks/` only when all of these are true:

1. The run follows `docs/benchmarks/credibility.md`.
2. JSON and Markdown reports agree.
3. The data contract, policy version, split, bootstrap seed, support, and limitations are documented.
4. Candidate sets and propensities are real logged values, not reconstructions.
5. A reviewer has checked the artifact and its claim.

Historical checked-in results remain for provenance. They are not automatically
current evidence.
