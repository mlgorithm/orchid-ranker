# API support policy

Orchid Ranker follows [semantic versioning](https://semver.org/) starting with
the 1.0 release. A stable public API is one documented in the current
[API reference](api_reference.md) and not labelled experimental.

## What 1.x keeps compatible

Within the 1.x line, Orchid will preserve the documented import paths, method
signatures, accepted data contracts, and return-object fields for:

- `from orchid_ranker import AdaptiveRanker` and its `fit`, `recommend`,
  `observe`, `recommend_and_log`, `observe_decision`, and recovery/export
  methods;
- the documented durable-decision interfaces and `SQLiteDecisionStore`;
- catalog validation in `orchid_ranker.learning_catalog`; and
- the documented `orchid_ranker.pilot` reference adapter, including its
  immutable assignment, lifecycle, and analysis-export contracts.

The normal public workflow remains `fit → recommend → observe`. Supporting
modules are stable only where the API reference documents them. Undocumented
module members, private names (beginning with `_`), serialized internal model
state, and exact ranking scores are not compatibility contracts.

The public interface is an in-process Python API. Orchid does not publish a
REST, GraphQL, or gRPC service contract. Authentication, authorization, rate
limits, and access to learner records belong to the embedding application.
Stable operations use `ValueError` for invalid or conflicting data, `KeyError`
for missing identities, `RuntimeError` for unmet lifecycle prerequisites, and
`TypeError` for wrong types or unknown configuration fields. Exception message
text is diagnostic rather than a compatibility contract.

## Experimental features

The knowledge-tracing research classes in `orchid_ranker.kt`, offline CQL and
delayed-gain policy promotion, off-policy evaluation, semantic retrieval, and
the one-course estimator in `orchid_ranker.pilot_analysis` are experimental.
The estimator's report schema may change as study methods mature. These
features' import paths or behavior may change in a minor release. They are not
required for the supported adaptive-practice workflow.

Experimental does not mean untested. It means users should pin an exact Orchid
version and validate the feature in their own chronological holdout before
depending on it in a product.

## Deprecations and breaking changes

For a stable API, Orchid will announce a deprecation in documentation and use
`DeprecationWarning` where practical. It will keep the deprecated behavior for
at least one subsequent minor release before removal. A breaking stable-API
change requires a new major version, except when correcting a data-integrity
or safety defect that cannot safely retain the old behavior.

Bug fixes may improve recommendations, diagnostics, or validation messages
without being considered a breaking change. Orchid does not promise identical
ranking order, scores, or learned model parameters across patch versions.

## Python and dependency support

Orchid 1.0 supports Python 3.11, 3.12, and 3.13. The continuous-integration
matrix tests each version; supported dependency ranges are declared in the
repository's `pyproject.toml`. The base installation supports empirical
adaptive practice without PyTorch; the optional `[kt]` extra supplies PyTorch
for neural knowledge tracing. New Python support or the retirement of an
upstream end-of-life Python version is announced in the changelog.
The `main` branch uses a development version between releases; the versioned
published documentation describes the matching released distribution.

## Scope of support

Report reproducible bugs using the repository issue templates and ask usage
questions through the channels in the repository's `SUPPORT.md`. Orchid
provides an adaptive decision component, not an LMS, curriculum authoring
system, or a guarantee of learning efficacy. Production integrations remain
responsible for candidate eligibility, access controls, privacy obligations,
and outcome measurement.
