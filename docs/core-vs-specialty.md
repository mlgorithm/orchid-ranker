# Core vs. specialty modules

Orchid Ranker ships two kinds of modules. Understanding the distinction will save you time and set correct expectations.

---

## Core

The core library delivers **progression-aware recommendation** for domains that satisfy three conditions:

1. Items have an ordering (prerequisites, difficulty, curator-authored progression).
2. Each interaction produces a signal richer than "they clicked" (completed, answered correctly, adhered, adopted, kept).
3. The stakeholder measures long-term outcomes, not short-term engagement (retention, pass rates, time-to-competence, clinical outcome).

Core modules have:

- Published benchmarks on public datasets with honest, reproducible results.
- Stable APIs under semver. Breaking changes require a major version bump.
- Regression tests that run on every PR.

Core modules: `OrchidRecommender`, `StreamingAdaptiveRanker`, `BayesianKnowledgeTracing`, `DependencyGraph`, `ProgressionRecommender`, `SafeSwitchDR`, evaluation metrics, serialization.

---

## Specialty

Specialty modules extend Orchid into adjacent territories where the three conditions hold with domain-specific reinterpretation. Each is narrowly scoped — the module name says exactly what it does and what it doesn't.

Specialty modules have:

- Unit tests with good coverage.
- A gate paragraph in the module docstring that tells a senior engineer in two sentences whether the module is for them.
- An "experimental" label until they pass the graduation bar below.

Specialty modules:

| Module | Domain | Status | Benchmark |
|--------|--------|--------|-----------|
| `orchid_ranker.scaling` | 100M registered users, ~10M concurrent active | Experimental | Synthetic 1M users: 99.8% memory savings, 840K ops/s |
| `orchid_ranker.curated_feed` | Editorially curated publications with reader learning arcs | Experimental | Synthetic only: +30.7% combined score |
| `orchid_ranker.cold_start` | New-user bootstrapping with transparent blend to Orchid | Experimental | **MovieLens-1M: +67% Surv@5 vs popularity** |
| `orchid_ranker.taste_progression` | Expertise-driven product domains (wine, photography, coffee) | Experimental | **Amazon Cell Phones: +0.9% kept-rate, 40.6% stretch accuracy; 92.9% warm-phase kept-rate in E2E** |

---

## Graduation bar

A specialty module graduates to core when all three conditions are met:

1. **Benchmark.** A published benchmark on a public dataset with an honest win over the relevant baseline. The benchmark must be reproducible (script in `benchmarks/`, data download automated, results in `docs/benchmarks/`). "Honest win" means the module moves a metric that matters for the domain — not just a unit-test assertion.

2. **External validation.** At least one external adopter running the module in production (or on their own data) with reported results. A design-partner case study counts. A single internal experiment does not.

3. **API stability.** The module survives one full release cycle (at least one minor version) with no breaking API changes. If the API needed breaking changes, the clock resets.

When all three conditions are met, the module moves from the specialty table to the core list in this document, the "experimental" label is removed from its docstring, and it is added to `__all__` in `__init__.py`.

Until then, specialty modules are shipped, supported, and tested — but adopters should benchmark on their own data before deploying.

### Current graduation progress

| Module | Condition 1 (Benchmark) | Condition 2 (External) | Condition 3 (API stable) |
|--------|:-----------------------:|:----------------------:|:------------------------:|
| `cold_start` | **Met** — MovieLens-1M, +67% Surv@5 | Not yet | In progress |
| `taste_progression` | **Partial** — Amazon Cell Phones +0.9% standalone, 92.9% warm-phase in E2E pipeline. Domain-dependent. | Not yet | In progress |
| `scaling` | **Partial** — synthetic benchmark. Memory and throughput are hardware properties, but no real-workload validation. | Not yet | In progress |
| `curated_feed` | **Not met** — synthetic data only. Needs a real news/content dataset. | Not yet | In progress |

`cold_start` is closest to graduation — it has a real-data benchmark with a meaningful win. It needs one external adopter and one stable release cycle.

---

## Why this matters

Open-source recommender libraries lose credibility one of two ways: they claim too little (and nobody tries them) or they claim too much (and early adopters get burned). The specialty/core distinction is how we claim precisely as much as we can defend.

If you're evaluating Orchid for a core use case (education, corporate training,
clinical rehab, fitness, gaming, onboarding), the library is built around that
shape of problem: ordered items, richer outcomes than clicks, and stakeholders
who care about long-term progress.

If you're evaluating a specialty module, two of four now have public-dataset benchmarks (`cold_start` on MovieLens-1M, `taste_progression` on Amazon Digital Music). The remaining burden is domain-specific: benchmark on your data, verify the fit, and tell us what you find. Your results are what graduate these modules.
