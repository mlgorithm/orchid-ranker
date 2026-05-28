# Changelog

## Unreleased

### Adaptive-First Cleanup

- Removed the old generic catalog-recommendation API surface from public
  exports, docs, examples, CI, and benchmarks.
- Retired batch recommendation, strategy-search, generic tuning,
  checkpoint-helper, catalog cold-start, curated-feed, taste-progression, and
  movie/music/e-commerce benchmark artifacts.
- Re-centered the package contract on adaptive learning: knowledge tracing,
  learner state, prerequisite-aware progression, offline policy evaluation,
  semantic new-exercise ranking, differential privacy, and safe rollout
  guardrails.

## 0.5.0 - 2026-05-11

### Adaptive Learning Focus

- Repositioned Orchid Ranker around progression-aware adaptive recommendation:
  learner state, prerequisite-aware candidate selection, learning-value policy
  evaluation, and safe rollout checks.
- Added business-facing docs for adaptive-learning fit, competitive positioning,
  algorithm roadmap, offline policy evaluation, progression policy, and pyKT
  integration.

### Knowledge Tracing And Policy Evaluation

- Added experimental `SAKTTracer` and `AKTTracer` for time-ordered correctness
  prediction on adaptive-learning logs.
- Added ASSISTments and EdNet preprocessing/benchmark paths for KT replay.
- Added `ProgressionValuePolicy`, `DelayedGainValuePolicy`, and
  `SupportConstrainedDelayedGainPolicy` for next-item ranking from KT state.
- Added IPS, SNIPS, direct-method, and doubly robust policy evaluation utilities
  for logged adaptive recommendations.
- Added delayed same-skill gain rewards, training-only delayed-gain priors, and
  reward-model diagnostics with calibration, cross-fit, and target-action bias
  reporting.
- Added pyKT sequence export/import helpers and prediction adapters so external
  KT model-zoo outputs can be evaluated inside Orchid's policy layer.

### Benchmarks

- Added KT quality, KT policy OPE, adaptive-efficiency, and delayed-gain
  reward-model benchmark CLIs.
- Current ASSISTments evidence: AKT is the strongest one-epoch in-repo tracer;
  progression reward shows positive offline value; strict delayed-gain policy is
  near break-even; learned support-constrained delayed-gain policy remains
  experimental because the direct value model overpredicts target-policy
  actions.

### Release Readiness

- Kept torch-dependent recommender APIs lazy so base installs remain
  torch-optional.
- Added public API exports for OPE, progression policy, delayed-gain modeling,
  and pyKT bridge helpers.
- Cleaned package metadata and extras for PyPI distribution checks.

## 0.3.2 - 2026-04-12

### Enterprise Hardening

- Fixed offline evaluation correctness so cross-validation uses in-user holdouts, explicit ratings flow into training, and missing users count as zero in aggregate ranking metrics.
- Tightened security and serialization by requiring HTTPS for JWT issuer/JWKS configuration, preserving encrypted audit-log verification, and removing unsafe pickle-based model loading.
- Corrected agentic and DP runtime behavior across `TwoTowerRecommender`, `AdaptiveAgent`, and safety-gating paths, including feedback ID normalization, DP budget enforcement, and boundary-probability handling.
- Serialized shared mutable state in adaptive-serving, `BigQueryConnector`, and MLflow tracking flows to eliminate mixed-state races under concurrent use.
- Aligned the published support contract and docs with the tested runtime matrix (Python 3.11–3.13), repaired the docs landing page quick start, and restored changelog coverage for the current release.

## 0.3.1 - 2026-04-11

- Changed license from MIT to Apache 2.0 (adds patent grant for enterprise adoption).
- Fixed repository URLs to point to `mlgorithm/orchid-ranker`.

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
  - `knowledge_coverage()` → `category_coverage()`
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

- **Educational Metrics** (`orchid_ranker.evaluation`):
  - `learning_gain` — Normalized learning gain (pre/post).
  - `knowledge_coverage` — Fraction of skills mastered.
  - `curriculum_adherence` — Prerequisite satisfaction rate.
  - `difficulty_appropriateness` — ZPD fit fraction.
  - `engagement_score` — Interaction ratio.
  - `EducationalReport` — Structured evaluation report dataclass.

### Adaptive Recommender Improvements

- Added explicit matrix-factorization and neural-ranking baselines for internal
  adaptive experiments and teacher/student warm starts.
- Extended neural baselines with `loss="bpr"` and `loss="softmax"` for stronger
  implicit ranking inside adaptive simulations.

### Agentic Improvements

- Warmup pre-loop with pseudo labels (`warmup_preloop`) and scheduling knobs (`warmup_rounds`, `warmup_steps`, `warmup_top_k_boost`, `warmup_diversity_scale`).
- Training augmentations: `train_on_all_shown`, `train_steps_per_round`.
- Optional Funk integration: distillation (`funk_distill`, `funk_lambda`) and Funk-based candidate generation (`use_funk_candidates`, `funk_pool_size`).
- DualRecommender warm-start and replay buffer options.

### Production Hardening

- Replaced global `np.random.seed()` with `np.random.RandomState` instances throughout offline evaluation helpers.
- Replaced `print()` with `logging.getLogger(__name__)` in production modules.
- Added input validation for empty DataFrames, invalid fold counts, and empty parameter grids.
- Narrowed exception handling from broad `Exception` to specific `(ValueError, RuntimeError, KeyError)`.
- Added checkpoint validation for internal PyTorch state persistence.
- Connector fix: `_require_lib()` calls before retry logic so ImportError is raised immediately.

### Testing

- 440+ tests across 25 test files including 84 stress tests.
- Stress test coverage: concurrency (20 threads), numerical stability (81 BKT param combos), scale (200k interactions), and mutation safety.
- New test files covered knowledge tracing, curriculum, educational metrics, and stress behavior.

### Benchmarks

- Added adaptive fixed-vs-online, warmup, scheduling, and CPU-safe agentic benchmarks.

### Documentation

- Complete README rewrite with quick-start examples and full module documentation.
- `docs/api_reference.md` — Complete API reference for all public classes and functions.
- Updated docs for new strategy list, benchmarking instructions, and agentic knobs.

## 0.2.0 - 2025-02-15

- Expanded the early experimental ranking stack with nearest-neighbor strategy support.
- Added DP accountant factory with Opacus support; introduced AuditLogger and RBAC enforcement.
- Delivered deployment assets: Dockerfile, Helm chart skeleton, Terraform reference.
- Introduced connectors (Snowflake, BigQuery, S3 streaming, MLflow) and Prometheus observability helpers.
- Created customer success docs (onboarding, SLAs, pilot plan) and GTM messaging pack.
- Automated SBOM + vulnerability scanning workflow and refreshed documentation across security/privacy/compliance.
