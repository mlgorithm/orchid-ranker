# Contributing to Orchid Ranker

Orchid Ranker is an outcome-driven adaptive recommender. Contributions should
strengthen its single fit, recommend, and observe workflow; production
decision logging; evaluation; or documentation.

Do not add another package-root model, domain engine, policy selector, or
tuning abstraction. Lower-level algorithms can support research and the
implementation without becoming additional user-facing products.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the local quality gate before a pull request:

```bash
./scripts/run_full_tests.sh --quick
./scripts/run_full_tests.sh
```

The quick mode checks lint, types, documentation readiness, publish readiness,
and the core adaptive-recommender smoke path. The full mode mirrors CI.

## Required Quality Gates

Run the relevant focused tests while developing, then run the full gate before
review:

```bash
python -m ruff check .
python -m mypy src/orchid_ranker
python -m pytest tests -q
python -m pytest tests -q --cov=orchid_ranker --cov-fail-under=75
python -m mkdocs build --strict
python -m build
```

CI also smoke-tests the built wheel, checks lower-bound dependency resolution,
runs the core suite on macOS and Windows, and audits resolved dependencies.

## Coding Standards

The detailed standard lives in
[docs/coding-standards.md](docs/coding-standards.md). The short version:

- Keep the package-root product API to `AdaptiveRanker`.
- Use `user_id`, `item_id`, `outcome`, and `timestamp` in new public examples.
- Keep lower-level algorithms in research or implementation modules.
- Keep the standard install complete. `pip install orchid-ranker` must run the
  supported workflow without an extra.
- Use explicit, typed public APIs. Preserve `py.typed`, avoid hidden global
  state, and prefer deterministic tests with `random_state` or seeded fixtures.
- Library code should use `logging`, not `print`. Examples and CLIs may print
  concise user-facing output.
- Use structured pandas/numpy/torch APIs instead of ad hoc string parsing or
  shape guessing.
- Add tests for public behavior, import contracts, documentation links, and
  regression-prone edge cases.

## Vocabulary

Use neutral adaptive-recommendation language consistently:

| Prefer | Avoid in new public docs/API | Notes |
|:--|:--|:--|
| user | learner in a domain-neutral workflow | Use `user_id` |
| outcome | correct in a domain-neutral workflow | Outcomes are binary `0` or `1` |
| timestamp | `ts` | Use `timestamp` |
| category | concept or skill outside education | Categories are optional |
| adaptive recommender | model zoo or generic ranking toolkit | The feedback loop is the product |

Do not add deprecated names, compatibility shims, or parallel public APIs.

## Documentation Standards

- Every public feature needs a runnable example, API reference entry, or guide.
- User-facing docs should start from data shape, fit, recommend, observe,
  then operate safely.
- Update `README.md`, `docs/README.md`, `docs/index.md`, and `mkdocs.yml` when
  adding a new user-facing guide.
- Do not link deleted generic docs, examples, benchmarks, or modules.
- Keep claims evidence-backed. Benchmark claims belong in
  `docs/benchmarks/credibility.md` and related benchmark cards.

## Pull Request Checklist

- The change strengthens the single adaptive-recommendation workflow.
- Public names and docs follow the vocabulary table.
- New behavior has focused tests.
- Docs and examples were updated when public behavior changed.
- `./scripts/run_full_tests.sh --quick` passes locally.
- The PR description lists the commands run and any skipped checks.
