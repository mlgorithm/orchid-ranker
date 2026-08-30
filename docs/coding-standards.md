# Coding Standards

This standard keeps Orchid Ranker focused, maintainable, and easy to use as an
outcome-driven adaptive recommender.

## Product Scope

Orchid does one public job:

- fit chronological user outcomes;
- rank an application-provided candidate set;
- observe the result and adapt; and
- log decisions safely when evaluation requires it.

Do not introduce additional package-root models, policy selectors, domain
engines, or tuning APIs. Lower-level algorithms may support the implementation
and research, but they are not separate product surfaces.

## Public API

- Start every new user-facing workflow from `AdaptiveRanker`.
- Keep `orchid_ranker.__all__` limited to `AdaptiveRanker`.
- Treat lower-level modules as implementation details, not alternate products.
- Keep the normal install complete: `pip install orchid-ranker` must run the
  supported workflow without an extra.
- Do not add compatibility aliases. For a stable documented API, follow the
  [API support policy](api-support-policy.md): announce a deprecation before
  removal, then make breaking changes only in a major release unless a
  data-integrity or safety defect requires an immediate correction.

## Python Style

- Use Python 3.11+ syntax that is accepted by the configured `mypy` and `ruff`
  settings.
- Type public functions, dataclasses, and return values. Avoid expanding
  `ignore_errors` coverage unless there is a concrete migration plan.
- Use `logging.getLogger(__name__)` in library code. Do not add `print()` calls
  outside examples or scripts.
- Prefer explicit validation errors over silent coercion for user-provided
  schema, tensor shape, and policy configuration.
- Keep randomness reproducible in tests and examples through `random_state`,
  local RNG instances, or seeded torch/numpy calls.
- Prefer small, direct functions over speculative abstractions.

## Data And ML Code

- Use structured pandas, numpy, sklearn, and torch APIs rather than ad hoc
  string parsing or manual dtype guessing.
- Preserve chronological splits for fitting and policy evaluation. Do not leak
  future user outcomes into training examples or candidate features.
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

- Put the fastest successful path first: install, data shape, fit, recommend,
  observe, evaluate.
- Link runnable examples from guides.
- Update `README.md`, `docs/README.md`, `docs/index.md`, and `mkdocs.yml` when
  adding or moving public guides.
- Keep benchmark and comparison claims precise. If evidence is not published,
  say what the benchmark is designed to measure instead of claiming a result.

## Review Bar

A change is ready when it strengthens the single adaptive recommendation loop,
uses neutral public terminology, has tests for public behavior, keeps docs
discoverable, and passes the required quality gates.
