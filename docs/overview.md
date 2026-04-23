# Orchid Ranker Overview

This document summarises the core building blocks exposed by the Orchid Ranker
library. It is intentionally lightweight so scientists and engineers can locate
entry points quickly.

## Recommenders
- `orchid_ranker.OrchidRecommender`: Surprise-style API with strategies such as
  `explicit_mf` (FunkSVD-style explicit MF), `als`, `neural_mf` (with `loss="bce"|"bpr"|"softmax"`),
  `implicit_als`, `implicit_bpr`, `linucb`, `user_knn`, `popularity`, and `random`.
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

## Development helpers
- `tests/`: run `pytest` for quick regressions (see `requirements-dev.txt`).
- `examples/`: runnable scripts for common integration flows.
- `orchid_ranker.logging.configure_logging`: centralise logs for deployment.
- `OrchidRecommender(validate_inputs=True)`: enforce schema validation on
  incoming interaction data.

Refer to the README quickstart for runnable examples and the `examples/`
folder for scripts that can be copied into notebooks. For product-specific
recipes, start with [Usage scenarios](scenarios.md).
