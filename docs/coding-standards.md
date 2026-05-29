# Coding Standards

This standard keeps Orchid Ranker focused, maintainable, and easy to use as an
adaptive-learning and knowledge-tracing recommender library.

## Product Scope

Orchid should be best at:

- learner-state tracing from outcome sequences;
- prerequisite-aware and difficulty-aware progression ranking;
- semantic exercise cold start for sparse catalogs;
- logged-policy learning and offline policy evaluation;
- safe rollout, observability, privacy, and auditability.

Do not reintroduce generic recommender surfaces such as movie/music/feed
benchmarks, broad model-selection APIs, or package-root tuning and
serialization helpers.

## Public API

- Start new user-facing workflows from `AdaptiveRanker` or
  `AdaptiveLearningEngine`.
- Use lower-level APIs only for explicit building blocks: KT tracers,
  `DependencyGraph`, `ProgressionRecommender`, progression policies, OPE,
  guardrails, connectors, and observability.
- Keep heavy dependencies optional. Torch-backed workflows belong behind extras
  such as `[adaptive]`; torch-free utilities must continue to import from a
  base install.
- Keep deprecated names isolated to compatibility shims and deprecation tests.
  New examples and docs should use the current names.

## Python Style

- Use Python 3.11+ syntax that is accepted by the configured `mypy` and `ruff`
  settings.
- Type public functions, dataclasses, and return values. Avoid expanding
  `ignore_errors` coverage unless there is a concrete migration plan.
- Use `logging.getLogger(__name__)` in library code. Do not add `print()` calls
  outside examples, CLI entry points, or scripts.
- Prefer explicit validation errors over silent coercion for user-provided
  schema, tensor shape, and policy configuration.
- Keep randomness reproducible in tests and examples through `random_state`,
  local RNG instances, or seeded torch/numpy calls.
- Prefer small, direct functions over speculative abstractions.

## Data And ML Code

- Use structured pandas, numpy, sklearn, and torch APIs rather than ad hoc
  string parsing or manual dtype guessing.
- Preserve chronological splits for KT and policy evaluation. Do not leak future
  learner outcomes into training examples or candidate features.
- Report policy quality with support diagnostics, clipped weights, and
  confidence intervals when making rollout claims.
- Treat benchmark results as evidence artifacts: include command, dataset
  shape, seeds, metrics, and limitations.

## Tests

Every public change should have focused tests. Expand to the full suite when a
change touches shared policy behavior, package exports, docs navigation,
install contracts, or CI.

Local gates:

```bash
python -m ruff check .
python -m mypy src/orchid_ranker
python -m pytest tests -q
python -m mkdocs build --strict
python -m build
```

Use `./scripts/run_full_tests.sh --quick` during development and
`./scripts/run_full_tests.sh` before review.

## Documentation

- Put the fastest successful path first: install, data shape, fit, rank,
  observe, evaluate.
- Link runnable examples from guides.
- Update `README.md`, `docs/README.md`, `docs/index.md`, and `mkdocs.yml` when
  adding or moving public guides.
- Keep benchmark and comparison claims precise. If evidence is not published,
  say what the benchmark is designed to measure instead of claiming a result.

## Review Bar

A change is ready when it is scoped to Orchid's adaptive-learning purpose,
uses current terminology, has tests for public behavior, keeps docs
discoverable, and passes the required quality gates.
