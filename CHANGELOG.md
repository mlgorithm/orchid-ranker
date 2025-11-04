# Changelog

## 0.2.1 - 2025-11-04
- Added `explicit_mf` (FunkSVD-style explicit matrix factorization) to `OrchidRecommender`.
- Extended `NeuralMatrixFactorizationBaseline` with `loss="bpr"` and `loss="softmax"` for stronger implicit ranking.
- Agentic improvements:
  - Warmup pre-loop with pseudo labels (`warmup_preloop`) and scheduling knobs
    (`warmup_rounds`, `warmup_steps`, `warmup_top_k_boost`, `warmup_diversity_scale`).
  - Training augmentations: `train_on_all_shown`, `train_steps_per_round`.
  - Optional Funk integration: distillation (`funk_distill`, `funk_lambda`) and
    Funk-based candidate generation (`use_funk_candidates`, `funk_pool_size`).
  - DualRecommender warm-start and replay buffer options.
- Benchmarks:
  - `benchmarks/eval_implicit.py` (multi-seed implicit apples-to-apples).
  - `benchmarks/run_agentic_adaptive.py` (fixed vs adaptive with warmup/scheduling, optional Funk).
  - `benchmarks/run_agentic_sklearn_digits.py` (CPU-safe agentic bench on sklearn digits).
- Docs updated: README strategy list and benchmarking instructions; overview of new agentic knobs.

## 0.2.0 - 2025-02-15
- Expanded OrchidRecommender with `user_knn` strategy and formal SUPPORTED_STRATEGIES constant.
- Added DP accountant factory with Opacus support; introduced AuditLogger and RBAC enforcement.
- Delivered deployment assets: Dockerfile, Helm chart skeleton, Terraform reference.
- Introduced connectors (Snowflake, BigQuery, S3 streaming, MLflow) and Prometheus observability helpers.
- Created customer success docs (onboarding, SLAs, pilot plan) and GTM messaging pack.
- Automated SBOM + vulnerability scanning workflow and refreshed documentation across security/privacy/compliance.
