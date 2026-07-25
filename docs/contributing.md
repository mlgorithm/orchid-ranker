# Contributing

Use the repository-level
[CONTRIBUTING.md](https://github.com/mlgorithm/orchid-ranker/blob/main/CONTRIBUTING.md)
for the full contribution workflow, and read
[Coding standards](coding-standards.md) before adding public APIs or examples.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quality Gate

```bash
./scripts/run_full_tests.sh --quick
./scripts/run_full_tests.sh
```

The quick gate runs lint, type checking, documentation-readiness checks,
publish-readiness checks, and the core adaptive-recommender smoke path. The
full gate runs lint, types, all tests, strict docs build, and package build.

## Contribution Scope

Good contributions improve the single `AdaptiveRanker` workflow, its outcome
data contract, evaluation, or documentation.

Do not introduce another package-root model, domain engine, policy selector, or
tuning API. Lower-level algorithms belong in implementation or research
modules.
