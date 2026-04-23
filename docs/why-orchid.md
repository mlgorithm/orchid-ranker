# Why Orchid Ranker

## The thesis

Most recommenders optimize for the next click. Orchid optimizes for **long-term user value**.

Every recommendation surface has users who are *progressing* --- discovering taste, building competence, refining preference, advancing through a product. A recommender that models this trajectory produces better outcomes than one that treats each interaction as independent.

## Five pillars

| Pillar | What it means | How Orchid delivers |
|--------|--------------|-------------------|
| **Long-term value** | Optimize for user growth, not engagement | Progression-aware scoring with stretch-zone targeting |
| **Adaptive** | Update per user, per interaction | Online residual adapters + Bayesian outcome tracing |
| **Streaming** | Real-time event ingestion | Kafka-native event bus with sub-10ms rank latency |
| **Safety-native** | Gate rollouts on user outcomes | Doubly-robust confidence sequences + progression guardrails |
| **Privacy-native** | Compliance built in | DP-SGD presets, RBAC, HMAC audit chains |

## Honest comparison

| Feature | Orchid Ranker | RecBole | NVIDIA Merlin | implicit |
|---------|:---:|:---:|:---:|:---:|
| Long-term value optimization | Yes | No | No | No |
| Online per-user adaptation | Yes | No | No | No |
| Streaming (Kafka) | Yes | No | Yes | No |
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
- Your users are *progressing* through something (content, competencies, products, preferences)
- You need real-time adaptation with safety guarantees
- You operate in a regulated domain (GDPR, FERPA, EU AI Act)

For a broader stack-by-stack comparison, see [Competitor comparison](competitors.md).

## Benchmark evidence

Orchid's specialty modules have been benchmarked on public datasets:

| Module | Dataset | Headline result |
|--------|---------|----------------|
| Cold-start bridge | MovieLens-1M (1M ratings) | **+67% Surv@5** vs popularity for brand-new users |
| Taste progression | Amazon Cell Phones (1.1M reviews) | **+0.9% kept-rate** standalone; **92.9% warm-phase kept-rate** in full pipeline |
| End-to-end pipeline | MovieLens-1M | Bridge -> Orchid -> Taste progression: **Surv@5 = 0.275** vs 0.150 popularity (**+80%**) |
| Scaling | Synthetic 1M users | **99.8% memory savings** at typical active ratios; **840K ops/s** concurrent synthetic load |
| Curated feed | Synthetic 500 users | +30.7% engagement x diversity; real-data benchmark pending |

The cold-start bridge is the strongest standalone real-data result (+67%
Surv@5). The +80% number belongs to the integrated end-to-end MovieLens
pipeline, where cold-start handling is composed with the rest of Orchid. Taste
progression's value is clearest as a re-ranker: modest standalone uplift
(+0.9%), but a large warm-phase improvement (92.9% vs 60.0%) in the full
pipeline. That warm-phase sample is small, so treat it as promising evidence,
not a final production guarantee.

Full methodology and reproducibility instructions start with the
[end-to-end benchmark](benchmarks/end-to-end.md) and linked specialty benchmark
pages.
