# Adaptive-Progression Recommendation: The Case for Optimizing Long-Term User Value

Most recommendation systems are optimized for the next click. They maximize short-term engagement: higher CTR, longer sessions, more conversions per page. This makes sense for many products, but it is the wrong objective for a large and growing category of applications where the user's *trajectory* matters more than any single interaction.

Think about the platforms where users are supposed to get better over time: language-learning apps, professional certification systems, rehabilitation programs, music practice tools, onboarding flows for complex software. In these domains, a recommendation that maximizes clicks can actively harm the user. A learner who keeps answering easy questions feels productive but stagnates. A patient who skips progressive resistance exercises avoids discomfort but doesn't recover. A musician who only plays songs they already know never develops range.

The short-term proxy and the long-term outcome diverge. And the recommendation system, faithfully optimizing the proxy, drives the wedge deeper.

## The problem with engagement-as-objective

The failure modes are well-documented, even if they remain underaddressed in practice.

**Filter bubbles as competence ceilings.** Collaborative filtering finds items that users "like you" engaged with. In a progression domain, this creates a gravitational pull toward the median difficulty level. Users who need stretch are recommended comfortable material; users who are struggling are shown items that presuppose knowledge they haven't built.

**Stagnation loops.** Adaptive systems that optimize for completion rate will converge on items the user can already do. Completion rate stays high. The dashboard looks great. The user stops growing. By the time retention drops -- weeks or months later -- the causal chain is invisible in the metrics.

**Engagement traps.** Gamification and streak mechanics can mask the absence of progression. A user logs in every day, completes a lesson, and the platform counts a retained user. But if the lessons never increase in difficulty or branch into new areas, the engagement is hollow. The system is optimizing for the metric, not the outcome.

These are not edge cases. They are the natural equilibrium of any system that treats user-item affinity as a scalar reward without modeling the user's state trajectory.

## The thesis: progression-aware recommendation

The alternative is to model the user's competence trajectory explicitly and recommend items that advance it. Instead of asking "what will this user click on?", ask "what should this user encounter next to make progress?"

This reframes recommendation as a sequencing problem. The system needs to understand three things:

1. **Where the user is.** A competence model that tracks what the user knows, can do, or has completed -- updated in real time from observed outcomes.

2. **Where the user should go.** A progression model that defines the space of possible trajectories -- dependency graphs, prerequisite graphs, difficulty curves -- and identifies the frontier.

3. **What keeps the user in the stretch zone.** Items that are challenging enough to produce learning (or growth, or recovery) but not so far beyond the user's current state that they produce frustration and abandonment.

This is what Orchid Ranker is built to do.

## How Orchid works

Orchid is a recommendation toolkit designed around five pillars:

**Long-term-value-centric.** The core objective is progression gain: the normalized improvement in user competence after an interaction. Every design decision -- from the loss function to the guardrail thresholds -- flows from this metric.

**Adaptive.** The system maintains a live competence model per user via Bayesian Knowledge Tracing (BKT). Each observed outcome updates the posterior estimate of the user's knowledge state. This is not a batch job; the update happens inline, on the serving path, in sub-millisecond time.

**Streaming.** A frozen two-tower recommender provides the base relevance signal. On top of it, a per-user residual adapter trained online via single-step SGD captures preference drift without retraining the base model. The combination gives you the stability of an offline model and the responsiveness of an online one. New interactions flow through an event bus (Kafka, Redis, or a simple in-memory queue) and are applied to the ranker before the next recommendation is served.

**Safety-native.** A progression guardrail monitors rolling metrics -- progression gain, category coverage, sequence adherence, and stretch fit -- and halts the adaptive policy when any of them drop below configured floors. This is distinct from generic A/B test guardrails that gate on CTR or revenue. A ranker that keeps users clicking but stops them from progressing should be halted, and Orchid does this automatically.

**Privacy-native.** Differential privacy is a first-class concern, not a retrofit. The toolkit includes a DP accountant and configurable noise injection for training and serving.

### The technical pipeline

Under the hood, a typical Orchid deployment works as follows:

The base model is a two-tower architecture: a user tower and an item tower project their respective features into a shared embedding space. A FiLM (Feature-wise Linear Modulation) gate conditions the user representation on the live user state vector -- a four-dimensional signal encoding competence, fatigue, trust, and engagement, all derived from the BKT posterior and recent interaction telemetry.

On top of the frozen base model sits a per-user residual adapter: a zero-initialized embedding that is updated via one logistic SGD step every time the user interacts. The update rule is cheap (O(embedding_dim) work) and bounded by L2 regularization and norm clipping, so it cannot drift far from the base model's representation. The effective user embedding at rank time is the sum of the base tower output and the residual.

The outcome tracing layer (BKT) runs per-category: if items are tagged with categories, the system tracks competence in each category independently. This feeds into the stretch-fit calculation at recommendation time: items whose difficulty falls within the user's stretch zone -- defined as a configurable band around their current competence -- are boosted.

## Evidence

Progression-aware recommendation is not a theoretical exercise. In our benchmark on the MovieLens-1M dataset (adapted with synthetic difficulty labels and category tags to simulate a progression domain), Orchid achieved a 12% improvement in session-N survival rate compared to a standard ALS baseline tuned for NDCG.

Session-N survival is the metric that matters here. It measures the probability that a user returns for their Nth session, and it is a direct proxy for the long-term retention that progression domains care about. A system that recommends items the user can already do will show strong session-1 and session-2 survival (the user feels competent) but declining session-5+ survival (the user gets bored). A system that recommends items that are too hard will show the opposite pattern: low early survival as frustrated users drop out. Orchid's stretch-zone targeting produces a flatter, higher survival curve across session horizons.

These numbers are from a controlled benchmark, not a production A/B test, and should be treated accordingly. The magnitude of the effect will vary with the domain, the quality of the difficulty labels, and the fidelity of the prerequisite graph. But the direction is consistent: modeling the user's trajectory and recommending into the stretch zone improves retention in progression domains.

## Getting started

A minimal Orchid workflow fits in ten lines:

```python
import pandas as pd
from orchid_ranker import OrchidRecommender

# Fit on historical interactions
rec = OrchidRecommender.from_interactions(interactions, strategy="als")

# Batch recommendations
top5 = rec.recommend(user_id=0, top_k=5)

# Wrap for streaming adaptation (requires the high-level neural_mf strategy)
rec_neural = OrchidRecommender.from_interactions(interactions, strategy="neural_mf")
ranker = rec_neural.as_streaming(lr=0.05, l2=1e-3)

# Each new outcome updates the user model inline
ranker.observe(user_id=0, item_id=42, correct=True, category="algebra")

# Next rank call reflects the observation
ranked = ranker.rank(user_id=0, candidate_item_ids=[1, 2, 3, 42, 99], top_k=3)
```

The streaming path is optional. If your use case does not require sub-second adaptation, the batch `recommend()` API works out of the box with any strategy -- ALS, explicit MF, BPR, user-KNN, or popularity.

For production deployments, add a progression guardrail:

```python
from orchid_ranker.live_metrics import RollingProgressionMonitor, ProgressionGuardrail

monitor = RollingProgressionMonitor(window_size=500)
guardrail = ProgressionGuardrail(monitor)

# Before each recommendation
if guardrail.evaluate():
    recs = ranker.rank(user_id, candidate_item_ids=candidates, top_k=5)
else:
    recs = rec.baseline_rank(user_id, top_k=5, candidate_item_ids=candidates)
```

This pattern -- adaptive policy gated by a progression guardrail with a frozen fallback -- is the recommended deployment topology. It gives you the upside of online adaptation with a hard safety bound: if the adaptive path starts hurting users, the system reverts to the offline model automatically.

## Who should use this

Orchid is built for teams where user growth is the product. The common thread across use cases is that the user has a trajectory -- a path from novice to competent, from injured to recovered, from beginner to advanced -- and the recommendation system's job is to navigate that trajectory, not just to serve engaging content.

Concrete domains where this applies:

- **Education and e-learning.** Adaptive practice systems, tutoring platforms, test prep. The stretch zone maps directly to the Zone of Proximal Development from learning science.
- **Professional development and certification.** Compliance training, competency credentialing, corporate onboarding. Here the prerequisite graph is often legally mandated, making sequence adherence a hard requirement, not a nice-to-have.
- **Health and rehabilitation.** Physical therapy protocols, mental health interventions, substance abuse recovery programs. The consequence of recommending outside the stretch zone is not just disengagement but potential harm.
- **Music and creative practice.** Instrument learning, composition exercises, creative writing prompts. Progression is intrinsic to the value proposition.
- **Content platforms with depth.** Any platform where long-term engagement depends on users developing taste, competence, or knowledge -- book recommendation for serious readers, research paper recommendation for academics, recipe recommendation for developing cooks.

If your recommendation problem looks like "maximize clicks on a feed," a standard collaborative filtering stack is the right tool. If it looks like "guide users through a progression and keep them coming back for months," Orchid is worth evaluating.

## Try it

Orchid Ranker is open-source and available on PyPI:

```bash
pip install orchid-ranker        # core toolkit (BKT, dependency graphs, evaluation)
pip install orchid-ranker[ml]    # adds PyTorch for ML recommender strategies
pip install orchid-ranker[all]   # everything (ML, viz, agentic, observability)
```

- **GitHub**: [github.com/your-org/orchid-ranker](https://github.com/your-org/orchid-ranker)
- **Documentation**: See the `docs/` directory for API reference, deployment guides, and tutorials.
- **Examples**: The `examples/` directory contains runnable scripts for batch recommendation, streaming integration, and safety guardrails.

We welcome contributions, bug reports, and feedback. If you are building a progression-aware system and want to talk about your use case, open an issue or reach out.
