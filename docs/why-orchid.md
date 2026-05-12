# Why Orchid Ranker

## The thesis

Most recommenders optimize for the next click. Orchid's primary workflow
optimizes for **adaptive learning**: what the learner should work on next so
they make measurable progress.

That means modeling learner state, prerequisite structure, item difficulty,
live outcomes, progression reward, and safe rollout instead of treating every
interaction as an independent click.

## Five pillars

| Pillar | What it means | How Orchid delivers |
|--------|--------------|-------------------|
| **Learner state** | Estimate what the learner knows now | AKT/SAKT-style tracing plus BKT utilities |
| **Progression reward** | Optimize learning progress, not engagement | Target-correctness, stretch-zone, mastery-gain, and repetition terms |
| **Live adaptation** | Update after each learner outcome | `AdaptiveLearningEngine.observe()` |
| **Safety-native** | Gate rollouts on user outcomes | Doubly-robust confidence sequences + progression guardrails |
| **Privacy-native** | Compliance built in | DP-SGD presets, RBAC, HMAC audit chains |

## Honest comparison

| Feature | Orchid Ranker | RecBole | NVIDIA Merlin | implicit |
|---------|:---:|:---:|:---:|:---:|
| Learner-state tracing | Yes | No | No | No |
| Prerequisite-aware ranking | Yes | No | No | No |
| Progression reward | Yes | No | No | No |
| Safety guardrails | Yes | No | No | No |
| Differential privacy | Yes | No | No | No |
| Model breadth (50+ models) | No | Yes | Yes | No |
| GPU retrieval throughput | No | No | Yes | No |
| Lightweight / minimal deps | Yes | No | No | Yes |

**Where Orchid is not the right choice:**
- You need 50+ model architectures for academic benchmarking -- use RecBole
- You need GPU-accelerated retrieval at ad-platform scale -- use NVIDIA Merlin
- You want the simplest possible implicit-feedback library -- use implicit

**Where Orchid shines:**
- Your users are learning, training, onboarding, or progressing through ordered tasks
- You need learner-state updates after each outcome
- You operate in a regulated domain (GDPR, FERPA, EU AI Act)

For a broader stack-by-stack comparison, see [Competitor comparison](competitors.md).

## Benchmark evidence

Orchid's adaptive-learning path and specialty modules have been benchmarked on
public or public-derived datasets:

| Module | Dataset | Headline result |
|--------|---------|----------------|
| AKT tracing | ASSISTments | AUC 0.7355 vs item-mean 0.6934 |
| Progression policy | ASSISTments OPE | +0.3212 mean uplift on progression reward |
| Cold-start bridge | MovieLens-1M (1M ratings) | **+67% Surv@5** vs popularity for brand-new users |
| Taste progression | Amazon Cell Phones (1.1M reviews) | **+0.9% kept-rate** standalone; **92.9% warm-phase kept-rate** in full pipeline |
| End-to-end pipeline | MovieLens-1M | Bridge -> Orchid -> Taste progression: **Surv@5 = 0.275** vs 0.150 popularity (**+80%**) |
| Scaling | Synthetic 1M users | **99.8% memory savings** at typical active ratios; **840K ops/s** concurrent synthetic load |
| Curated feed | Synthetic 500 users | +30.7% engagement x diversity; real-data benchmark pending |

The strongest adaptive-learning claim is KT + progression reward. Delayed-gain
policies are still experimental: current diagnostics show mixed or weak OPE
evidence until reward-model calibration and logged support improve. The
cold-start and taste-progression results are specialty evidence for adjacent
progression domains, not the main reason to adopt Orchid.

Full methodology and reproducibility instructions start with the
[ASSISTments KT benchmark](benchmarks/assistments-kt.md),
[KT policy OPE benchmark](benchmarks/kt-policy-ope.md), and linked specialty
benchmark pages.
