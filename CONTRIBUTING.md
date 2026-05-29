# Contributing to Orchid Ranker

Orchid Ranker is an adaptive-learning and knowledge-tracing recommender
library. Contributions should strengthen learner-state estimation, progression
ranking, prerequisite-aware candidate selection, offline policy evaluation,
safety, privacy, deployment, or documentation for those workflows.

Do not add generic recommender APIs, movie/music/feed benchmarks, broad model
selection helpers, or package-root tuning/serialization abstractions. Those
surfaces were intentionally removed so the library can be excellent at adaptive
learning instead of broad but shallow recommendation.

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
and the core adaptive-learning smoke path. The full mode mirrors the important
CI gates.

## Required Quality Gates

Run the relevant focused tests while developing, then run the full gate before
review:

```bash
python -m ruff check .
python -m mypy src/orchid_ranker
python -m pytest tests -q
python -m mkdocs build --strict
python -m build
```

CI also runs package install contracts, lower-bound dependency smoke tests, and
wheel-content checks.

## Coding Standards

The detailed standard lives in
[docs/coding-standards.md](docs/coding-standards.md). The short version:

- Keep public workflows adaptive-first: `AdaptiveRanker`,
  `AdaptiveLearningEngine`, KT tracers, progression policies, OPE, guardrails,
  semantic exercise cold start, connectors, and deployment utilities.
- Keep optional heavy dependencies behind extras and compatibility checks. A
  base install must still expose torch-free progression utilities.
- Use explicit, typed public APIs. Preserve `py.typed`, avoid hidden global
  state, and prefer deterministic tests with `random_state` or seeded fixtures.
- Library code should use `logging`, not `print`. Examples and CLIs may print
  concise user-facing output.
- Use structured pandas/numpy/torch APIs instead of ad hoc string parsing or
  shape guessing.
- Add tests for public behavior, import contracts, documentation links, and
  regression-prone edge cases.

## Vocabulary

Use adaptive-learning language consistently:

| Prefer | Avoid in new public docs/API | Notes |
|:--|:--|:--|
| learner | generic customer/user when progress is the point | `user_id` remains the identifier column |
| concept/category | skill when the domain is not education-specific | Existing compatibility aliases may still mention skill |
| competence/proficiency | mastery as a new API name | Deprecated aliases are isolated in compatibility shims |
| progression reward | click/rating objective | Orchid optimizes learning progress |
| stretch fit | difficulty appropriateness | Old metric aliases are deprecated |
| dependency graph | prerequisite graph class name in new APIs | Prerequisite prose is fine when describing content |
| adaptive-learning recommender | generic recommender | This is the core positioning |

Deprecated names such as `StudentAgent`, `MasteryTracker`, and
`CurriculumRecommender` should appear only in compatibility shims, migration
notes, changelog history, or tests that assert deprecation behavior.

## Documentation Standards

- Every public feature needs a runnable example, API reference entry, or guide.
- User-facing docs should start from data shape, fit, rank, observe, evaluate,
  then operate safely.
- Update `README.md`, `docs/README.md`, `docs/index.md`, and `mkdocs.yml` when
  adding a new user-facing guide.
- Do not link deleted generic docs, examples, benchmarks, or modules.
- Keep claims evidence-backed. Benchmark claims belong in
  `docs/benchmarks/credibility.md` and related benchmark cards.

## Pull Request Checklist

- The change fits the adaptive-learning / KT recommender scope.
- Public names and docs follow the vocabulary table.
- New behavior has focused tests.
- Docs and examples were updated when public behavior changed.
- `./scripts/run_full_tests.sh --quick` passes locally.
- The PR description lists the commands run and any skipped checks.
