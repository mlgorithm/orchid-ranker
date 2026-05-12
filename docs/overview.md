# Orchid Ranker Overview

This document summarises the core building blocks exposed by Orchid's
adaptive-learning engine. It is intentionally lightweight so scientists and
engineers can locate entry points quickly.

## Adaptive-Learning Stack

- `orchid_ranker.AdaptiveLearningEngine`: alias for the primary adaptive
  learning API when you want the product concept in the import name.
- `orchid_ranker.AdaptiveLearningRecommender`: existing implementation class for fit -> rank ->
  observe adaptive-learning loops. It composes KT prediction, progression
  reward, prerequisite gating, and live policy state.
- `orchid_ranker.scenarios.recommend_scenarios`: choose an Orchid workflow
  from product/data signals before picking a model.
- `orchid_ranker.kt.AKTTracer`: difficulty-aware tracer for candidate learning
  items from a learner's recent interaction sequence.
- `orchid_ranker.kt.SAKTTracer`: compact SAKT-style tracer when difficulty
  metadata is unavailable.
- `orchid_ranker.learning_policy.ProgressionValuePolicy`: stable default policy
  for adaptive-learning serving. It scores target-correctness fit, stretch-zone
  fit, mastery-gain potential, difficulty, and repetition/easy-item penalties.
- `orchid_ranker.learning_policy.DelayedGainValuePolicy`: experimental
  delayed-gain-aware policy using training-only same-concept gain priors.
- `orchid_ranker.learning_policy.SupportConstrainedDelayedGainPolicy`: learned
  delayed-gain reward policy with support penalties for low-coverage actions.
- `orchid_ranker.pykt_bridge`: export Orchid interactions to pyKT sequence
  format and reuse pyKT prediction tables behind Orchid policies and OPE.

## Ranking Machinery and Fallbacks

- `orchid_ranker.OrchidRecommender`: Surprise-style API with strategies such as
  `explicit_mf` (FunkSVD-style explicit MF), `als`, `neural_mf` (with `loss="bce"|"bpr"|"softmax"`),
  `implicit_als`, `implicit_bpr`, `linucb`, `user_knn`, `popularity`, and `random`.
  This is supporting machinery for non-learning metadata, baseline comparison,
  and fallback paths rather than the main product surface.
- `orchid_ranker.recommender.Recommendation`: lightweight dataclass returned by
  `recommend()`.
- `recommend(..., candidate_item_ids=[...])`: rank a caller-provided candidate
  pool using the original item IDs from the fitted interaction data.
- `OrchidRecommender.from_interactions(...)`: one-call fit path. The default
  `strategy="auto"` chooses `als` for binary feedback and `explicit_mf` for
  explicit rating ranges.

## Streaming and Safety
- `OrchidRecommender.as_streaming()`: promote a fitted `neural_mf` recommender
  into a live adaptive ranker. The bridge accepts the same external user and
  item IDs used in training data.
- `orchid_ranker.streaming.StreamingAdaptiveRanker`: lower-level streaming
  runtime for custom towers.
- `orchid_ranker.live_metrics.RollingProgressionMonitor`: rolling progression
  metrics for production monitoring.
- `orchid_ranker.live_metrics.ProgressionGuardrail` and
  `orchid_ranker.safety.SafeSwitchDR`: fallback controls for adaptive rollouts.
- `orchid_ranker.ope.evaluate_logged_policy`: IPS, SNIPS, direct-method, and
  doubly robust value estimates from logged propensities.
- `orchid_ranker.ope.compare_logged_policies`: paired target-vs-baseline policy
  uplift with confidence intervals and weight diagnostics.

## Knowledge Tracing Benchmarks

- `orchid_ranker.kt.build_sakt_examples`: leakage-safe sequence builder for
  next-response training examples.
- `orchid_ranker.kt_benchmark`: time-ordered replay evaluation for
  EdNet/ASSISTments-style CSVs.
- `orchid_ranker.learning_policy.KTValuePolicy`: transparent next-item policy
  using predicted correctness, stretch fit, uncertainty, and expected-gain
  proxy.

## Agentic Simulation
- `orchid_ranker.agents.MultiUserOrchestrator`: primary orchestrator coordinating
  simulated users and recommenders.
- `orchid_ranker.agents.MultiConfig`: configuration dataclass controlling
  adaptive policy bounds and privacy toggles. New knobs include warmup controls
  (`warmup_preloop`, `warmup_rounds`, `warmup_steps`, `warmup_top_k_boost`,
  `warmup_diversity_scale`), training augmentation (`train_on_all_shown`,
  `train_steps_per_round`), and optional Funk integration (`funk_distill`,
  `funk_lambda`, `use_funk_candidates`, `funk_pool_size`).
- `orchid_ranker.agents.AdaptiveAgent`: behavioural user simulator used for
  agentic evaluation loops.

## Privacy and Differential Privacy Helpers
- `orchid_ranker.dp.get_dp_config`: fetch ready-made DP presets.
- `orchid_ranker.agents.simple_dp`: minimalist DP-SGD utilities for experimental
  use; see `docs/privacy.md` for limitations.
- `orchid_ranker.dp_accountant.build_accountant`: factory for per-sample and
  Opacus-backed privacy accountants.
- `orchid_ranker.security`: role-based access control (`AccessControl`) and
  JSONL audit logging (`AuditLogger`).

## Connectors & Observability
- `orchid_ranker.connectors`: optional integrations with Snowflake, BigQuery, S3, and MLflow.
- `orchid_ranker.observability`: Prometheus registry helpers (`start_metrics_server`, `record_training`, `export_metrics`).
- Deployment artefacts: Dockerfile, Helm chart (`deploy/helm`), and Terraform reference (`deploy/terraform`).

## Automation helpers
- `tests/`: run `pytest` for quick regressions (see `requirements-dev.txt`).
- `benchmarks/compare_surprise.py`: fit Orchid ALS and Surprise SVD side by side
  on your dataset to compare RMSE.
- `benchmarks/compare_implicit.py`: benchmark Orchid ALS against the
  `implicit` library on identical train/test splits.
- `benchmarks/compare_reclab.py`: evaluate Orchid versus ReCLaB's TopPop
  baseline using dense ratings generated from ReCLaB environments.
- `benchmarks/run_agentic_smoke.py`: execute a synthetic multi-round agentic
  simulation to ensure orchestrator wiring works end-to-end.
- `orchid_ranker.logging.configure_logging`: centralise logs for deployment.
- `OrchidRecommender(validate_inputs=True)`: enforce schema validation on
  incoming interaction data.

Refer to the README quickstart for runnable examples and the `examples/`
folder for scripts that can be copied into notebooks. For product-specific
recipes, start with [Usage scenarios](scenarios.md).
