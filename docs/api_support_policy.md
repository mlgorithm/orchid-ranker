# Orchid Ranker Support & Versioning Policy

## Semantic Versioning
- The `orchid-ranker` package follows semantic versioning (MAJOR.MINOR.PATCH).
- Breaking API changes are introduced only in MAJOR releases, with deprecation notices provided at least one MINOR release beforehand.
- Public interfaces include the `OrchidRecommender` class, streaming and safety APIs, and documented extras.

## Supported Runtime Matrix
- Python: 3.11 – 3.13 (verified in CI across CPython builds).
- PyTorch: 1.13 – 2.9 (primary development targets 2.x; earlier versions may miss optional GPU optimisations).
- Operating systems: Ubuntu 22.04 LTS, macOS 14+, Windows Server 2022 (CPU paths).

## Deprecation Workflow
- Deprecated parameters or modules raise `DeprecationWarning` starting with the announcing release.
- Removal schedule is communicated in release notes and the project roadmap, typically two MINOR releases after deprecation.
- A migration guide accompanies every major change.

## Release & Testing Cadence
- Main branch kept releasable; CI covers unit tests and smoke checks for supported runtimes.
- Performance and compatibility checks may be run before releases when the change touches serving or safety behavior.

## Support Channels
- GitHub issues for bug reports and feature requests.
- Enterprise subscribers gain access to dedicated support SLAs and onboarding sessions.

Updated: 2026-04-12
