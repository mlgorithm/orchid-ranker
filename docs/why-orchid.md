# Why Orchid Ranker

## The thesis

Most recommenders optimize for the next click. Orchid is built for products
where the user is progressing through something and the operator cares about
that trajectory.

That makes Orchid strongest in training, certification, onboarding,
rehabilitation, and other outcome-sensitive products. It is not a general claim
that every recommendation surface should be treated this way.

## Five pillars

| Pillar | What it means | How Orchid delivers |
|--------|--------------|-------------------|
| **Long-term value** | Optimize for user growth, not engagement | Progression-aware scoring with stretch-zone targeting |
| **Adaptive** | Update per user, per interaction | Online residual adapters + Bayesian outcome tracing |
| **Streaming** | Real-time event ingestion | Kafka-native event bus with sub-10ms rank latency |
| **Safety-native** | Gate rollouts on user outcomes | Doubly-robust confidence sequences + progression guardrails |
| **Privacy-native** | Compliance evidence support | DP-SGD presets, RBAC, HMAC audit chains, operator runbooks |

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
- You need real-time adaptation with measurable safety-gate evidence
- You operate in a regulated domain (GDPR, FERPA, EU AI Act)

## Where Orchid fits best

Orchid is strongest in products where:

- users are progressing through content, skills, or product capability over time
- outcomes are richer than a click
- operators care about safety, auditability, and long-term results

For a broader stack-by-stack comparison, see [Competitor comparison](competitors.md).
