# Application Scenarios

Where Orchid Ranker fits — and where it doesn't.

---

## The core differentiator

Orchid Ranker isn't a general-purpose "Netflix-style" recommender. It has a specific thesis: **the user is getting better at something over time, and a recommender that models that trajectory outperforms one that only maximizes the next click.**

Orchid has capabilities most recommender libraries lack:

| Capability | What it does |
|------------|-------------|
| **Outcome tracing (BKT)** | Tracks per-user competence as a hidden Bayesian state |
| **Dependency graphs** | Enforces prerequisite ordering (can't recommend calculus before algebra) |
| **Stretch-zone scoring** | Recommends items at the right difficulty — not too easy, not too hard |
| **Streaming adaptation** | Sub-10ms per-interaction updates without retraining |
| **Safety guardrails** | Circuit breakers that halt the adaptive policy if metrics degrade |
| **Differential privacy** | Built-in DP-SGD for regulated environments |

Standard recommenders (implicit, LightFM, RecBole) optimize `argmax P(click)`. Orchid optimizes **progression** — how much the user grows over time.

For implementation recipes, see [Usage scenarios](scenarios.md). This page is
for fit and positioning; `scenarios.md` is for code paths.

---

## Scenario 1: Education platforms

**Fit:** Native. This is the clearest domain for Orchid because the catalog has
structured categories, ordered prerequisites, and explicit success/failure
outcomes.

**Examples:** Duolingo-style language learning, Khan Academy-style tutoring, corporate LMS, test prep (CFA, USMLE, bar exam)

**Why Orchid fits:**

| Need | Orchid feature |
|------|---------------|
| Track each user's completed categories | BKT outcome tracing per category |
| Don't assign problems that are too easy or too hard | Stretch-zone scoring (`stretch_fit`) |
| Enforce prerequisite order (algebra -> calculus) | `DependencyGraph` with topological sorting |
| Adapt in real-time as the user responds | `StreamingAdaptiveRanker.observe()` -> instant re-rank |
| Measure learning, not just engagement | `progression_gain`, `category_coverage`, `sequence_adherence` |
| FERPA/COPPA compliance | DP-SGD presets, audit logging |

**What it looks like in code:**

```python
rec = OrchidRecommender.from_interactions(user_activity, strategy="neural_mf")
streamer = rec.as_streaming(lr=0.05)

# User completes an interaction
streamer.observe(user_id=42, item_id=137, correct=True, category="fractions")

# Get next question from the eligible pool
next_items = streamer.rank(
    user_id=42,
    candidate_item_ids=question_pool,
    top_k=5,
)
```

---

## Scenario 2: Corporate training & upskilling

**Commercial priority:** Lead tier — clearest buyer intent, measurable outcomes (pass rates, time-to-competence), regulated procurement that rewards compliance posture.

**Examples:** Compliance training platforms, professional certification paths, employee onboarding, L&D platforms (Degreed, Coursera for Business)

**Why Orchid fits:**

- **Structured curricula.** Certification paths have strict prerequisite trees. `DependencyGraph` enforces them.
- **Competence verification.** Need to prove employees actually learned, not just clicked through. BKT tracks competence probability per category.
- **Personalized pace.** Senior hires skip basics, new grads need foundations. Stretch-zone scoring adapts automatically.
- **Audit requirements.** SOC 2 and regulated industries need evidence of training completion. HMAC-chained audit logs and DP-SGD provide the paper trail.
- **Time-constrained learners.** Corporate learners have 15 min/day — must recommend the highest-value next item. Progression-gain optimization does this.

---

## Scenario 3: Clinical rehabilitation & physical therapy

**Commercial priority:** Lead tier — stakes are high (injury risk), metrics are clinical (pain, range of motion, adherence), and the privacy stack is a genuine wedge. Longer sales cycles, larger deals.

**Examples:** PT platforms (Hinge Health, Kaia Health, Sword Health), post-surgical recovery, stroke rehabilitation, addiction recovery programs

**Why Orchid fits:**

- **Progressive difficulty.** Exercises must ramp up gradually — knee bend, full squat, weighted squat, jump landing. `DependencyGraph` + stretch-zone handles this.
- **Regression detection.** If a patient's competence drops (pain increase, ROM decrease), `ProgressionGuardrail` halts progression and falls back to safer exercises.
- **Per-patient adaptation.** Each patient recovers at different rates. Streaming per-user adaptation personalizes the pace.
- **HIPAA/medical privacy.** DP-SGD with configurable epsilon ensures no raw data exposure.
- **Outcome tracking.** "Did the patient successfully complete this exercise?" maps directly to BKT's binary observation model.

---

## Scenario 4: Music & audio with learning intent

**Commercial priority:** Showcase tier — great for demos and the generalization claim; smaller commercial fit for the pure-engagement sub-domains. Target the progression-native sub-domains (language-through-music, theory apps, curated-listening education), not Spotify-style homepage discovery.

**Examples:** Language-learning-through-music (Lyricly, LyricsTraining), music theory apps (Tenuto, TonedEar), Masterclass-style guided listening, curated classical-music pedagogy apps, podcast series with learning arc (Huberman Lab, Hardcore History), audiobook language courses.

**Why Orchid fits differently here:**

- **Taste evolution.** A user's music taste *develops* — from pop to indie rock to post-punk to shoegaze. This is a progression trajectory, not a static preference.
- **Stretch-zone = discovery zone.** Items slightly outside comfort zone lead to discovery. Too far outside leads to rejection. This is exactly stretch-zone scoring.
- **Novelty vs. familiarity balance.** Standard recommenders over-exploit (play the same 50 songs). Orchid's `category_coverage` metric naturally encourages genre breadth.
- **Session retention.** The session-N survival metric directly measures "does this playlist keep the user listening?"

**Limitation:** Music platforms typically don't expose prerequisite structure in their catalog, so `DependencyGraph` is less useful unless a curator authors one. The streaming adaptation and stretch-zone features are what matter here. Pure Spotify-DW-style discovery is an engagement problem and belongs on a CTR-optimized ranker, not Orchid.

---

## Scenario 5: Gaming progression systems

**Commercial priority:** Showcase tier — visceral demos ("watch Orchid pick chess puzzles at your level in real time"), smaller TAM. Strong for credibility and marketing; secondary for pipeline.

**Examples:** Skill-based matchmaking, chess/go/bridge trainers (Chessable, Lichess puzzle streaks, chess.com), esports coaching (ProGuides, Mobalytics), aim trainers with progressive difficulty (Aim Lab, KovaaK's), cognitive training apps (Peak, BrainHQ), challenge sequencing, tutorial systems, roguelike difficulty scaling

**Why Orchid fits:**

- **Ability-based matchmaking.** BKT estimates player competence; match against opponents/challenges at the stretch-zone boundary (not too easy = boring, not too hard = ragequit).
- **Unlock trees.** Game progression trees (unlock pistol, shotgun, sniper) map directly to `DependencyGraph`.
- **Adaptive difficulty.** Real-time streaming observation adjusts difficulty mid-session.
- **Churn prevention.** The guardrail detects when a player is failing too much and falls back to easier recommendations.

---

## Scenario 6: Structured knowledge content

**Commercial priority:** Showcase tier for the course-marketplace sub-domain; limited fit for general article-ranking (Medium/Substack homepages are engagement problems).

**Examples:** Online course marketplaces with progression paths (Udemy, Pluralsight, O'Reilly Learning), technical documentation with learning paths (AWS, Stripe, Auth0 quickstarts), research-paper recommendation with topic-dependency structure (Semantic Scholar-style tracks), curated reading lists with prerequisite order (ML roadmaps, CS self-study paths).

**Why Orchid fits:**

- **Reading level progression.** Articles have difficulty; readers develop comprehension over time. Stretch-zone scoring surfaces the right next article.
- **Topic dependency.** Understanding "transformers" requires understanding "attention mechanisms" first. `DependencyGraph` encodes this.
- **Engagement vs. learning.** A content platform that recommends articles the user *learns from* (not just clicks) retains better long-term.
- **Category coverage.** Encourages breadth — don't just recommend ML papers to someone who also wants to learn systems design.

---

## Scenario 7: Fitness & wellness apps

**Commercial priority:** High-TAM tier — big consumer market, clear outcome metrics (adherence, weight, PR), and the guardrail story resonates (progress safely, avoid injury). Subscription model means retention uplift translates directly to revenue.

**Examples:** Peloton-style workout recommendation, running training plans (Strava, Runna), strength training apps (Caliber, Future, Tonal), yoga progression (Glo, Alo Moves), meditation sequences (Calm, Headspace Unwinding programs), habit-formation apps (Noom, Finch, Fabulous, Streaks)

**Why Orchid fits:**

- **Progressive overload.** Fitness is fundamentally a progression problem — gradually increase intensity. This is Orchid's thesis.
- **Injury prevention.** Guardrails halt progression if the user reports pain or performance drops.
- **Personalized periodization.** Some users recover faster; streaming adaptation personalizes rest/intensity cycles.
- **Category coverage.** A balanced workout plan covers cardio, strength, flexibility, mobility — not just bench press every day.

---

## Scenario 8: Product onboarding & user activation

**Commercial priority:** High-TAM tier — every SaaS company has this problem, retention metric ties directly to revenue, fastest sales cycles of any scenario. Strong candidate for a case study that validates the "general recommender" pitch in a commercially compelling slice.

**Examples:** SaaS in-product guidance platforms (Intercom Product Tours, Appcues, Pendo Guides, Userpilot, Chameleon, WalkMe, Whatfix), analytics-driven recommendation (Amplitude Recommend, Mixpanel Experiment), developer-tool onboarding (Stripe, Auth0, AWS learning paths), in-app feature discovery / activation flows

**Why Orchid fits:**

- **Feature unlock sequences.** Show users features in dependency order (can't use "advanced filters" before "basic search").
- **Competence tracking.** Has the user actually used the feature successfully, or just seen it?
- **Adaptive pace.** Power users skip the basics; new users get hand-held through each step.
- **Churn risk.** The guardrail detects if onboarding is failing and falls back to simpler tasks.

---

## Specialty modules (benchmarked extensions)

These modules extend Orchid into adjacent territories. Each is tightly scoped — the name says exactly what it does and what it doesn't.

| Module | What it solves | What it does NOT solve | Benchmark status |
|--------|---------------|----------------------|-----------------|
| `orchid_ranker.scaling` | Memory + lock contention at 100M registered / ~10M concurrent active users | Throughput past Python GIL limits (10k+ QPS requires profiling); see caveats below | Synthetic 1M users — [results](benchmarks/scaling.md) |
| `orchid_ranker.curated_feed` | Editorially curated publications where readers develop topic expertise (Stratechery, Economist, specialist newsletters) | Engagement-driven social feeds (Meta, TikTok, Reddit) — those are CTR problems | Synthetic only — [results](benchmarks/curated-feed.md) |
| `orchid_ranker.cold_start` | New-user bootstrapping with transparent blend to full Orchid | Replacing a mature collaborative filter at scale — this is a bridge, not a competitor | **MovieLens-1M** — [results](benchmarks/cold-start.md) |
| `orchid_ranker.taste_progression` | Expertise-driven product domains where taste evolves (wine, photography, coffee, fashion) | Commodity e-commerce (paper towels, batteries, USB cables) — no trajectory to model | **Amazon Digital Music** — [results](benchmarks/taste-progression.md) |

### What the benchmarks show

Two specialty modules have real-data benchmark evidence (`cold_start` and
`taste_progression`). Two others are useful but should be treated as
engineering/prototype evidence until validated on real workloads (`scaling` and
`curated_feed`). Here is what we can defend and what remains open:

**Cold-start bridge (strongest result).** On MovieLens-1M (1M ratings, 6K users), the ColdStartBridge achieves Surv@5 = 0.230 — a **+67% uplift** over a popularity baseline (0.138). Orchid direct (no cold-start handling) scores 0.000 because it has zero information about new users. The bridge solves a real problem that every collaborative-filtering system has, and the improvement is large enough to matter in production.

Key design insight validated: uncovered items must receive **neutral scores** (mean of covered items), not zero. Giving zeros implicitly penalises niche items during blending — exactly the items a cold-start bridge should surface.

**Taste progression (domain-dependent, strongest as a re-ranker).** Tested on two Amazon domains: Cell Phones (1.1M reviews, 157K users — clear price tiers) and Digital Music (169K reviews — weak price signal). Cell Phones shows **+0.9% kept-rate uplift** and **40.6% stretch accuracy**; Digital Music shows -4.0% uplift and only 5.1% stretch accuracy. The algorithm works when the sophistication signal is real; **the quality of the domain signal is the bottleneck**, not the model.

The standalone uplift is modest, but the real value appears in the **end-to-end pipeline**: when used as a re-ranker on top of Orchid's collaborative filtering, the full pipeline achieves **92.9% warm-phase kept-rate** vs 60.0% without taste re-ranking.

**Scaling (solid engineering, not a competitive moat).** At 1% active users (the common production ratio), sparse tables save **99.8% of memory** — from 122 MB to 0.2 MB for 1M registered users. Sharded BKT enables **840K+ ops/sec** under concurrent load. This is table-stakes infrastructure done right, not a differentiator — but it means adopters don't have to build it.

**Curated feed (unproven on real data).** +30.7% engagement on synthetic data with gradual diversity penalty. The MIND dataset (Microsoft News) requires authenticated access we couldn't obtain. Synthetic benchmarks prove the code works and the diversity penalty is correctly calibrated, but don't prove it beats alternatives on real content. This module still needs a real-data benchmark before it can claim a concrete advantage.

### End-to-end pipeline (the integrated story)

The [end-to-end benchmark](benchmarks/end-to-end.md) on MovieLens-1M measures the full lifecycle: cold-start → warm transition → taste progression re-ranking.

| System | Surv@5 | Warm-phase kept-rate |
|--------|--------|---------------------|
| Popularity | 0.150 | 0.0% |
| Orchid direct | 0.000 | 0.0% |
| Bridge only | 0.270 | 60.0% |
| **Full pipeline** | **0.275** | **92.9%** |

The bridge is the biggest win for cold users (+80% survival). Taste progression's value appears in the warm phase — 92.9% vs 60.0% kept-rate for users with established trajectories. The full pipeline is the only system to achieve any Surv@20.

---

## When NOT to use Orchid Ranker

These scenarios are genuinely outside Orchid's thesis — the underlying optimization objective is fundamentally different:

| Scenario | Why not | Use instead |
|----------|---------|------------|
| Ad serving / CTR optimization | Pure click optimization; Orchid is anti-click-maximization by design | Wide & Deep, DeepFM |
| Social / engagement feed ranking | Relationship + engagement driven; no progression signal | Custom ranking model, two-tower CTR ranker |
| Search result ranking | Relevance to a query, not progression through a trajectory | BM25, semantic retrieval (ColBERT, bi-encoders) |
| Dating / bidirectional matching | Mutual preference; fundamentally different optimization problem | Custom mutual-ranking systems |
| Commodity e-commerce (basket fill) | No taste progression — buying paper towels does not develop expertise | LightFM, RecBole, implicit |
| Engagement-driven news feeds | Meta/TikTok/Reddit-style feeds are CTR problems where Orchid has no edge | Two-tower engagement ranker with diversity terms |

---

## The Orchid sweet spot

**Orchid is the right choice when all three conditions hold:**

1. **There is an ordering over items** — from prerequisites, difficulty grading, or a curator-authored progression. The ordering can be hard (calculus requires algebra) or soft (this track is harder than that one).
2. **Each interaction produces a richer signal than "they clicked"** — completed with comprehension, answered correctly, adhered to the plan, adopted the feature, graduated to the next level, didn't return the purchase, replayed voluntarily. Without this, Orchid has nothing to distinguish itself from an engagement ranker.
3. **The stakeholder measures long-term outcomes, not short-term engagement** — 90-day retention, pass rates, time-to-competence, adherence, feature-adoption depth, clinical outcome. If the OKR is clicks or watch-minutes, Orchid is the wrong tool regardless of the technical fit.

That covers: education, corporate training, certification, clinical rehab, fitness, progression-gaming, learning-intent music/audio, structured knowledge content, and product onboarding.

Specialty modules extend into adjacent territories where the three conditions hold with domain-specific reinterpretation: taste-progression commerce (`orchid_ranker.taste_progression`) and editorially curated content feeds (`orchid_ranker.curated_feed`). These are narrowly scoped — see [specialty modules](#specialty-modules-benchmarked-extensions) for what they do and don't cover, including benchmark evidence.

It does NOT cover: ads, social feeds, engagement-driven news feeds, search ranking, dating, commodity e-commerce, or any surface where the objective is short-term engagement.

---

## Priority guide for adopters and outreach

Not all scenarios are equally important commercially or technically. The table below covers **core** scenarios only — specialty modules (curated feed, taste progression, cold-start, scaling) are separate and documented in [Core vs. specialty](core-vs-specialty.md).

| Tier | Core scenarios | Why |
|:-----|:---------------|:----|
| **Native** | Education (#1) | Library was conceived for this; organic adoption will happen here for free. |
| **Lead** | Corporate training (#2), Clinical rehab (#3) | Clearest buyer intent, measurable ROI, compliance stack is a real wedge. Highest-priority outreach targets. |
| **High-TAM** | Product onboarding (#8), Fitness (#7) | Largest commercial opportunity if we prove the fit. Shorter sales cycles. Ideal for second-wave case studies. |
| **Showcase** | Progression gaming (#5), Music/audio with learning intent (#4), Structured knowledge content (#6) | Great for demos and the generalization claim. Smaller TAM, secondary for pipeline. |

---

## Scaling to 100M registered users

The default `StreamingAdaptiveRanker` stores per-user state in a dense embedding table. That works well up to ~1M users. Beyond that, use the `orchid_ranker.scaling` module:

```python
from orchid_ranker.scaling import ScalingConfig

config = ScalingConfig(
    max_active_users=10_000_000,  # LRU eviction for inactive users
    num_state_shards=32,          # parallel BKT state access
)

streamer = rec.as_streaming(lr=0.05, scaling_config=config)
```

**What changes at scale:**

| Component | Default (< 1M users) | With ScalingConfig |
|-----------|---------------------|-------------------------------|
| Residual embeddings | Dense `nn.Embedding` (all users in memory) | `SparseEmbeddingTable` with LRU eviction |
| BKT state | Single dict + single lock | `ShardedBKTStateProvider` (N shards, N locks) |
| Memory per user | Fixed allocation for all users | Only active users in memory |
| Lock contention | Single RLock (serialized) | Per-shard locks (N-way parallel) |
| Inactive users | Stay in memory forever | LRU-evicted; return to cold-start on re-access |

**Memory footprint at scale (emb_dim=32):**

| Registered users | Active users | Default memory | With scaling |
|-----------------|-------------|---------------|-------------|
| 100K | 100K | 0.44 GB | 0.42 GB |
| 1M | 1M | 1.6 GB | 0.55 GB |
| 10M | 10M | 13+ GB (breaks) | 1.8 GB |
| 100M | ~10M active | impossible | 3.5 GB |

**Benchmark evidence (1M registered users, emb_dim=32):**

| Active users | Dense (MB) | Sparse (MB) | Savings |
|-------------|-----------|------------|---------|
| 1,000       | 122.1     | 0.2        | 99.8%   |
| 10,000      | 122.1     | 2.4        | 98.1%   |
| 100,000     | 122.1     | 23.7       | 80.6%   |

Concurrent throughput with sharded BKT:

| Shards | Threads | Throughput |
|--------|---------|-----------|
| 4      | 8       | ~838K ops/s |
| 16     | 8       | ~843K ops/s |

See [full benchmark](benchmarks/scaling.md) for methodology and LRU eviction validation.

!!! warning "Caveats"
    **LRU eviction = cold-start on return.** Evicted users lose their learned residual embeddings and BKT state. When they return after inactivity, they re-enter via cold-start (zero residual). This is the right trade-off for memory, but set expectations: a user who disappears for 3 months and returns will get baseline-quality recommendations until their trajectory is re-established. Combine with `orchid_ranker.cold_start` to smooth this re-entry.

    **The "100M+ users" claim requires qualification.** It means 100M *registered* users with up to ~10M *concurrent active* users in memory. If your active set is 50M simultaneously, you need 50M × emb_dim × 4 bytes of memory. The module solves memory scaling, not throughput scaling.

    **Throughput (QPS) is a separate axis.** Sharded BKT reduces lock contention N-way, but all shards still run through Python locks. For 10k+ QPS sustained throughput, you'll need to benchmark on your hardware. For horizontal scaling beyond a single process, combine with Kafka consumer-group partitioning: each partition maps to one ingestor shard.

---

## Curated feed ranking

For editorially curated publications where readers develop topic expertise over time. Think: Stratechery, Matt Levine's *Money Stuff*, Economist Espresso, specialist industry press, technical newsletter ecosystems, course marketplaces.

This module does **not** target engagement-driven social feeds (Meta, TikTok, Reddit, Twitter/X). Those are CTR-optimization problems where every modern feed ranker already has diversity and freshness terms. Orchid's edge in this space comes from the stretch-fit and topic-competence signals — and those only matter when readers are building understanding through a body of content.

```python
from orchid_ranker.curated_feed import FeedRanker, FeedItem, FreshnessScorer, TopicTracker

ranker = FeedRanker(
    freshness=FreshnessScorer(halflife_hours=12),
    topic_tracker=TopicTracker(),
    w_relevance=0.3,
    w_freshness=0.25,
    w_stretch=0.2,
    w_diversity=0.15,
    w_competence=0.1,
)

candidates = [
    FeedItem(item_id=1, topic="ai-policy", difficulty=0.7, timestamp=time.time() - 3600),
    FeedItem(item_id=2, topic="ai-basics", difficulty=0.3, timestamp=time.time() - 7200),
    FeedItem(item_id=3, topic="climate", difficulty=0.5, timestamp=time.time() - 1800),
]

ranked = ranker.rank(user_id=42, candidates=candidates, top_k=10)
```

**Five scoring components:**

| Component | What it does | Weight (default) |
|-----------|-------------|-----------------|
| **Relevance** | Base recommender score (collaborative filtering) | 0.30 |
| **Freshness** | Exponential time-decay (`exp(-age / halflife)`) | 0.25 |
| **Stretch fit** | How well item difficulty matches user's reading level | 0.20 |
| **Diversity** | MMR-style penalty for topics already in the list | 0.15 |
| **Competence** | User's readiness for the item's topic (via BKT) | 0.10 |

**Where the progression signal matters:** A reader who has consumed 20 articles on AI policy can handle a nuanced analysis piece. A new reader needs the explainer first. The `TopicTracker` models this as topic-level competence, and the `ReadingLevelEstimator` tracks overall reading complexity preference. These signals are unique to Orchid — standard feed rankers don't have them.

**Benchmark evidence (synthetic, 500 users / 1000 items):**

| System | Engagement | Diversity | Combined |
|--------|-----------|-----------|----------|
| Popularity           | 0.401 | 0.630 | 0.504 |
| Reverse-chron        | 0.263 | 0.770 | 0.466 |
| **FeedRanker**       | **0.411** | **0.850** | **0.586** |

The gradual diversity penalty (`1 − 0.3 × same_count`) allows a second item from the same topic while discouraging three or more, producing +30.7% combined engagement × diversity over popularity. See [full benchmark](benchmarks/curated-feed.md).

!!! warning "Synthetic data only"
    These results are on synthetic data — they prove the code works and the diversity penalty is correctly calibrated, but not that the module beats alternatives on real content. The MIND dataset (Microsoft News) requires authenticated access. Before deploying, verify that stretch-fit and topic-competence signals move a metric that existing feed rankers can't (e.g. session depth on a curated publication, not CTR on a general-purpose feed).

---

## Cold-start handling

Orchid needs interaction history to model user trajectories. The `orchid_ranker.cold_start` module bridges the gap:

```python
from orchid_ranker.cold_start import ColdStartBridge, ColdStartConfig, PopularityPrior

# Fit a popularity prior from historical data
pop = PopularityPrior(smoothing=1.0)
pop.fit(all_item_ids_from_interactions)

# Wrap your Orchid recommender
bridge = ColdStartBridge(
    recommender=rec,
    item_features=item_feature_matrix,
    popularity_prior=pop,
    config=ColdStartConfig(min_interactions=5, blend_until=30),
)

# Works for brand-new users AND experienced users
recs = bridge.recommend(user_id=new_user_id, top_k=10)

# As the user interacts, Orchid is automatically blended in
bridge.observe(user_id=new_user_id, item_id=42, outcome=1.0)
print(bridge.warmth(new_user_id))  # 0.0 → 1.0 as interactions accumulate
```

**How blending works:**

| Interactions | Behaviour | Orchid weight |
|-------------|-----------|---------------|
| 0 – `min_interactions` | Pure cold-start (popularity + content similarity) | 0.0 |
| `min_interactions` – `blend_until` | Linear blend (cold-start fades, Orchid ramps up) | 0.0 → 1.0 |
| ≥ `blend_until` | Pure Orchid | 1.0 |

**Cold-start scoring components:**

| Component | What it does |
|-----------|-------------|
| **Content similarity** | Scores candidates by cosine similarity to the user's seed items (stated preferences or early interactions) |
| **Popularity prior** | Global or segment-level item popularity (Laplace-smoothed) |

The transition is transparent — the user never sees a "cold start" vs "warm" switch. The bridge automatically ramps up Orchid's influence as the user's trajectory becomes modelable.

**Benchmark evidence (MovieLens-1M):**

| System | Surv@5 | Mean session length |
|--------|--------|---------------------|
| Popularity         | 0.138 | 2.83 |
| Content-only       | 0.070 | 2.28 |
| **ColdStartBridge** | **0.230** | **3.12** |
| Orchid direct      | 0.000 | 0.48 |

The bridge outperforms popularity by **+67%** on session survival for brand-new users. See [full benchmark](benchmarks/cold-start.md) for methodology and discussion.

---

## Taste progression (expertise-driven commerce)

This module applies to product domains where users develop expertise and taste over time. It does **not** apply to commodity e-commerce.

**Gate question:** *Does your user develop taste or expertise through repeated purchases in this category?* If yes, this module fits. If no (paper towels, batteries, USB cables), use a standard collaborative filter.

| Domain | Progression | Why Orchid fits |
|--------|------------|----------------|
| Wine | Table wine → regional varietals → reserve → rare vintages | Palate develops; novices who buy a $200 Barolo return it |
| Photography | Kit lens → prime → L-series → specialised macro | Understanding develops; premature upgrade → unused gear |
| Coffee | Drip → pour-over → espresso → single-origin → competition | Brewing competence gates equipment purchases |
| Fashion | Fast fashion → contemporary → designer → bespoke | Fit preference and brand literacy develop over years |
| Cooking | Basic knife set → Japanese → Damascus → custom forged | Technique gates whether expensive tools get used |
| Audio / hi-fi | Bluetooth speaker → bookshelf → floorstanding → tubes | Listening ability develops; premature upgrade → no perceived difference |

```python
from orchid_ranker.taste_progression import TasteProgressionRanker, TasteConfig, SophisticationMapper

# Map items to a sophistication tier (0 = entry-level, 1 = premium)
soph = SophisticationMapper.from_prices(item_prices)

ranker = TasteProgressionRanker(
    recommender=rec,  # optional — works standalone too
    sophistication_scores=soph,
    config=TasteConfig(stretch_width=0.15),
)

# Set category metadata
ranker.set_item_categories({1: "wine", 2: "wine", 3: "coffee", 4: "coffee"})

# Observe purchase outcomes — "kept" = positive, "returned" = negative
ranker.observe(user_id=42, item_id=1, purchased=True, returned=False,
               category="wine", rating=4.5)

# Recommend items matching the user's taste trajectory
recs = ranker.recommend(user_id=42, top_k=10, candidate_item_ids=catalog)
```

**The reframing:**

| Orchid concept | Taste-progression mapping |
|---------------|--------------------------|
| "Correct" observation | Purchased AND kept (not returned), or rated ≥ threshold |
| "Incorrect" observation | Returned, or rated poorly |
| Category / attribute | Product category (wine, photography, running shoes) |
| Difficulty | Sophistication (price tier, expert rating, feature complexity) |
| Stretch zone | Next tier up (recommend a step up, not two tiers ahead) |
| Competence | Taste readiness (has the user successfully adopted products at this tier?) |

**Four scoring components:**

| Component | What it does | Weight (default) |
|-----------|-------------|-----------------|
| **Relevance** | Collaborative filtering score from Orchid | 0.40 |
| **Stretch fit** | Gaussian match between item sophistication and user taste level | 0.35 |
| **Momentum** | Bonus for items slightly above user's current tier (upward discovery) | 0.15 |
| **Exploration** | Bonus for product categories the user hasn't tried yet | 0.10 |

**Benchmark evidence (Amazon Cell Phones, 1.1M reviews):**

| System | Kept rate | Stretch accuracy |
|--------|----------|-----------------|
| Popularity         | 80.3% | — |
| Recent popularity  | 78.1% | — |
| **TasteProgression** | **81.1%** | **40.6%** |

**Kept-rate uplift vs popularity: +0.9% (standalone), +55% warm-phase kept-rate in end-to-end pipeline.**

The progression curve correctly models expertise development: users move from taste level 0.508 (mid-range) to 0.651 (developing expertise) over their interaction history. Stretch accuracy of 40.6% confirms price tiers are a meaningful sophistication proxy for electronics. See [full benchmark](benchmarks/taste-progression.md) for multi-domain comparison.

!!! note "Domain signal quality matters"
    On Digital Music (weak price tiers), kept-rate uplift is -4.0% and stretch accuracy is only 5.1%. On Cell Phones (clear price tiers), uplift is positive and stretch accuracy is 8x higher. The algorithm works when the sophistication signal is real. For domains without price-based tiers, use domain-specific sophistication signals (expert ratings, feature complexity, prerequisite structure).
