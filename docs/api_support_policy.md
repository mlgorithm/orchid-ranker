# Orchid Ranker Support & Versioning Policy

## Semantic Versioning
- The `orchid-ranker` package follows semantic versioning (MAJOR.MINOR.PATCH).
- Breaking API changes are introduced only in MAJOR releases, with deprecation notices provided at least one MINOR release beforehand.
- Public interfaces center on `AdaptiveRanker`, `AdaptiveLearningEngine`,
  `AdaptiveLearningRecommender`, adaptive algorithm helpers, scenario-selection
  helpers, adaptive-learning benchmark scripts, and documented extras. Generic
  recommender model-zoo APIs are not part of the supported public surface.

## Supported Runtime Matrix
- Python: 3.11 – 3.13 (verified in CI across CPython builds).
- PyTorch: 2.x for torch-backed extras (`adaptive`, `torch`, `offline_rl`, `bandits`, `streaming`, `enterprise`, `agentic`, `benchmarks`).
- Operating systems: Ubuntu 22.04 LTS, macOS 14+, Windows Server 2022 (CPU paths).

## Typing Support
- The package ships inline type hints with a partial `py.typed` marker.
- CI runs `mypy src/orchid_ranker` with repository-owned overrides that keep the
  hardened public core checked: adaptive learning, learner-state tracing,
  progression evaluation, and safe-rollout utilities.
- Experimental, benchmark, connector, and agentic modules remain import- and
  runtime-tested, but are not yet covered by the typed-core guarantee.

## Deprecation Workflow
- Deprecated parameters or modules raise `DeprecationWarning` starting with the announcing release.
- Removal schedule is communicated in release notes and the project roadmap, typically two MINOR releases after deprecation.
- A migration guide accompanies every major change.

## Release & Testing Cadence
- Main branch kept releasable; CI covers unit tests, CLI smoke tests, and benchmark regressions for each supported runtime.
- Nightly performance jobs gather latency/throughput metrics for all supported strategies; regressions >10% trigger release gates.

## Support Channels
- GitHub issues for bug reports and feature requests.
- Enterprise subscribers gain access to dedicated support SLAs, onboarding sessions, and private roadmap briefings.

Updated: 2026-05-28
