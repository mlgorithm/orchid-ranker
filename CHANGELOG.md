# Changelog

## 0.3.0 - 2026-04-11

### Breaking Changes (with backward-compat aliases until v1.0)

All renamed symbols emit `DeprecationWarning` when accessed via their old name. Old names will be removed in v1.0.

- **Classes renamed** (domain-neutral terminology):
  - `MasteryTracker` → `ProficiencyTracker`
  - `PrerequisiteGraph` → `DependencyGraph`
  - `CurriculumRecommender` → `ProgressionRecommender`
  - `StudentAgent` → `AdaptiveAgent`
  - `StudentAgentFactory` → `AdaptiveAgentFactory`
  - `EducationalReport` → `ProgressionReport`

- **Functions renamed**:
  - `learning_gain()` → `progression_gain()`
  - `knowledge_coverage()` → `proficiency_coverage()`
  - `curriculum_adherence()` → `sequence_adherence()`

- **Parameter renames** (old keyword-only aliases still accepted):
  - `DependencyGraph.prerequisites_met(mastered=)` → `completed=`
  - `DependencyGraph.available(mastered=)` → `completed=`
  - `DependencyGraph.path_to(mastered=)` → `completed=`
  - `ProgressionRecommender.recommend(student_mastery=)` → `completed=`
  - `proficiency_coverage(mastered_skills=, total_skills=)` → `achieved=, total=`

### Enterprise Hardening

- **Logging**: Replaced all `print()` calls with `logging.getLogger(__name__)` across 30+ modules.
- **Thread safety**: Added `threading.Lock` with double-checked locking for `fast_score.py` native extension loading and `observability.py` readiness state.
- **Security**: `torch.load(weights_only=True)` enforced in serialization; legacy pickle fallback emits `DeprecationWarning`.
- **Input validation**: Probability parameters validated in [0, 1], dimension parameters validated as positive, remediation guidance in error messages.
- **Module exports**: Added `__all__` to all public modules for IDE discoverability and `from module import *` safety.
- **Deprecation strategy**: PEP 562 `__getattr__` for lazy loading of deprecated names with `DeprecationWarning` at point of use (not import time).

### Bug Fixes

- Fixed `NeuralMatrixFactorizationBaseline` using unset `emb_dim` instead of `self.emb_dim` for embedding layers.
- Fixed `UserKNNBaseline` crash when `k >= num_users`; now clamps to `num_users - 1`.
- Fixed `LinUCBBaseline` crash on out-of-bounds item IDs in `fit()`.
- Fixed internal imports of `StudentAgent` in `orchestrator.py` and `legacy_orchestrator.py` that triggered spurious `DeprecationWarning` at import time.

### Documentation

- README rewritten with domain-neutral framing and updated code examples using new API names.
- `pyproject.toml` description and keywords updated for multi-domain positioning.
- Full docstrings added to `TwoTowerRecommender.think()` and `decide()` methods.

### Testing

- All 1061 tests passing including backward-compatibility tests for all renamed APIs.

## 0.2.1 - 2025-11-04

### New Modules

- **Knowledge Tracing** (`orchid_ranker.knowledge_tracing`):
  - `BayesianKnowledgeTracing` — Hidden Markov Model for skill mastery estimation with Bayesian updates.
  - `MasteryTracker` — Multi-skill portfolio tracking with per-skill BKT instances and prerequisite awareness.
  - `ForgettingCurve` — Ebbinghaus exponential decay model for spaced repetition scheduling.

- **Curriculum Sequencing** (`orchid_ranker.curriculum`):
  - `PrerequisiteGraph` — DAG-based skill dependency graph with cycle detection (DFS), Kahn's topological sort, learning path planning, and serialization.
  - `CurriculumRecommender` — ZPD-aware recommendations respecting prerequisite ordering and difficulty targeting.

- **Model Selection** (`orchid_ranker.model_selection`):
  - `train_test_split` — Per-user stratified or global train/test splitting.
  - `cross_validate` — K-fold cross-validation with configurable metrics.
  - `compare_models` — Side-by-side strategy comparison returning a DataFrame.
  - `evaluate_on_holdout` — Held-out evaluation for fitted models.

- **Hyperparameter Tuning** (`orchid_ranker.tuning`):
  - `GridSearchCV` — Exhaustive search over parameter grid with cross-validation scoring.
  - `RandomSearchCV` — Randomized search with reproducible results.

- **Model Serialization** (`orchid_ranker.serialization`):
  - `save_model` / `load_model` — Versioned checkpoint save/restore for OrchidRecommender and TwoTowerRecommender with corruption detection and unfitted-model guards.

- **Educational Metrics** (`orchid_ranker.evaluation`):
  - `learning_gain` — Normalized learning gain (pre/post).
  - `knowledge_coverage` — Fraction of skills mastered.
  - `curriculum_adherence` — Prerequisite satisfaction rate.
  - `difficulty_appropriateness` — ZPD fit fraction.
  - `engagement_score` — Interaction ratio.
  - `EducationalReport` — Structured evaluation report dataclass.

### Recommender Improvements

- Added `explicit_mf` (FunkSVD-style explicit matrix factorization) to `OrchidRecommender`.
- Extended `NeuralMatrixFactorizationBaseline` with `loss="bpr"` and `loss="softmax"` for stronger implicit ranking.
- Added `STRATEGY_GUIDE` dict with human-readable descriptions for all 9 strategies.
- Added `available_strategies()` classmethod and "did you mean?" typo suggestions via `difflib.get_close_matches`.
- Added `.save(path)` / `.load(path)` convenience methods on `OrchidRecommender`.

### Agentic Improvements

- Warmup pre-loop with pseudo labels (`warmup_preloop`) and scheduling knobs (`warmup_rounds`, `warmup_steps`, `warmup_top_k_boost`, `warmup_diversity_scale`).
- Training augmentations: `train_on_all_shown`, `train_steps_per_round`.
- Optional Funk integration: distillation (`funk_distill`, `funk_lambda`) and Funk-based candidate generation (`use_funk_candidates`, `funk_pool_size`).
- DualRecommender warm-start and replay buffer options.

### Production Hardening

- Replaced global `np.random.seed()` with `np.random.RandomState` instances throughout model_selection and tuning.
- Replaced `print()` with `logging.getLogger(__name__)` in production modules.
- Added input validation for empty DataFrames, invalid fold counts, and empty parameter grids.
- Narrowed exception handling from broad `Exception` to specific `(ValueError, RuntimeError, KeyError)`.
- Added checkpoint validation in serialization (unfitted check, required keys, corrupted file detection).
- Connector fix: `_require_lib()` calls before retry logic so ImportError is raised immediately.

### Testing

- 440+ tests across 25 test files including 84 stress tests.
- Stress test coverage: concurrency (20 threads), numerical stability (81 BKT param combos), scale (200k interactions), mutation safety, serialization roundtrips.
- New test files: `test_knowledge_tracing.py`, `test_curriculum.py`, `test_educational_metrics.py`, `test_model_selection.py`, `test_tuning.py`, `test_serialization.py`, `test_stress.py`.

### Benchmarks

- `benchmarks/eval_implicit.py` (multi-seed implicit apples-to-apples).
- `benchmarks/run_agentic_adaptive.py` (fixed vs adaptive with warmup/scheduling, optional Funk).
- `benchmarks/run_agentic_sklearn_digits.py` (CPU-safe agentic bench on sklearn digits).

### Documentation

- Complete README rewrite with 6 quick-start examples, strategy reference, and full module documentation.
- `docs/api_reference.md` — Complete API reference for all public classes and functions.
- Updated docs for new strategy list, benchmarking instructions, and agentic knobs.

## 0.2.0 - 2025-02-15

- Expanded OrchidRecommender with `user_knn` strategy and formal SUPPORTED_STRATEGIES constant.
- Added DP accountant factory with Opacus support; introduced AuditLogger and RBAC enforcement.
- Delivered deployment assets: Dockerfile, Helm chart skeleton, Terraform reference.
- Introduced connectors (Snowflake, BigQuery, S3 streaming, MLflow) and Prometheus observability helpers.
- Created customer success docs (onboarding, SLAs, pilot plan) and GTM messaging pack.
- Automated SBOM + vulnerability scanning workflow and refreshed documentation across security/privacy/compliance.
