# Orchid Ranker Support & Versioning Policy

## Semantic Versioning
- The `orchid-ranker` package follows semantic versioning (MAJOR.MINOR.PATCH).
- Breaking API changes are introduced only in MAJOR releases, with deprecation notices provided at least one MINOR release beforehand.
- Public interfaces include the `OrchidRecommender` class, experiment runner, preprocessing CLI entry points, and documented extras.

## Supported Runtime Matrix
- Python: 3.9 – 3.13 (verified in CI across CPython builds).
- PyTorch: 1.13 – 2.9 (primary development targets 2.x; earlier versions may miss optional GPU optimisations).
- Operating systems: Ubuntu 22.04 LTS, macOS 14+, Windows Server 2022 (CPU paths).

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

Updated: 2025-02-15
