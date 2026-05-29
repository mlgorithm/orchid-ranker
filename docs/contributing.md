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
publish-readiness checks, and the core adaptive-learning smoke path. The full
gate runs lint, types, all tests, strict docs build, and package build.

## Contribution Scope

Good contributions improve adaptive learning, knowledge tracing, progression
ranking, prerequisite-aware candidate selection, OPE, safe rollout, privacy,
connectors, or documentation for those workflows.

Do not reintroduce generic recommender APIs, generic feed/movie/music
benchmarks, or package-root tuning and serialization helpers.
