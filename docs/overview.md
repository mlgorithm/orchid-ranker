# Orchid Ranker Overview

This document summarises the core building blocks exposed by the Orchid Ranker
library. It is intentionally lightweight so scientists and engineers can locate
entry points quickly.

## Recommenders
- `orchid_ranker.OrchidRecommender`: Surprise-style API with strategies such as
  `als`, `neural_mf`, `implicit_als`, `implicit_bpr`, `linucb`, `popularity`, and `random`.
- `orchid_ranker.recommender.Recommendation`: lightweight dataclass returned by
  `recommend()`.

## Agentic Simulation
- `orchid_ranker.agents.MultiUserOrchestrator`: primary orchestrator coordinating
  simulated learners and recommenders.
- `orchid_ranker.agents.MultiConfig`: configuration dataclass controlling
  adaptive policy bounds and privacy toggles.
- `orchid_ranker.agents.StudentAgent`: behavioural simulator used for agentic
  evaluation loops.

## Privacy and Differential Privacy Helpers
- `orchid_ranker.dp.get_dp_config`: fetch ready-made DP presets.
- `orchid_ranker.agents.simple_dp`: minimalist DP-SGD utilities for experimental
  use; see `docs/privacy.md` for limitations.

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
folder for scripts that can be copied into notebooks.
