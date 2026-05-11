# Usage scenarios

Orchid is a first-class adaptive-learning recommender library. The primary
path is next-item learning: estimate learner state, rank eligible items, observe
the outcome, and update the next decision. The other scenarios are extensions
or fallbacks for products that do not yet have the full learning signal stack.

## Choosing in code

Use the scenario selector when you know your product shape but have not chosen
an algorithm yet.

```python
from orchid_ranker import recommend_scenarios

matches = recommend_scenarios(
    has_outcomes=True,
    has_concepts=True,
    has_difficulty=True,
    has_prerequisites=True,
    needs_live_adaptation=True,
    use_case="adaptive math practice",
)

for fit in matches:
    print(f"{fit.scenario.id}: {fit.score}")
    print("  ", fit.scenario.entrypoints)
```

`available_scenarios()` returns the full stable catalog as `ScenarioRecipe`
objects.

## Scenario 1: Adaptive-learning next-item recommendation

Use this when each outcome should affect the next recommendation immediately:
practice questions, tutoring exercises, certification tasks, product onboarding
steps, rehab exercises, or progression games.

**Signals:** historical learner outcomes, item concepts, item difficulty,
optional prerequisites, plus live outcome events.

**Algorithm path:** AKT/SAKT knowledge tracing, progression-value policy,
prerequisite gating, and optional delayed-gain policy after diagnostics and
offline policy evaluation.

**Orchid path:** `AdaptiveLearningRecommender.fit()` -> `rank()` -> `observe()`.

```python
from orchid_ranker import AdaptiveLearningRecommender

rec = AdaptiveLearningRecommender(
    tracer_model="akt",
    policy="auto",
    epochs=2,
).fit(
    outcomes,
    user_col="user_id",
    item_col="item_id",
    correct_col="correct",
    timestamp_col="timestamp",
    concept_col="skill_id",
    item_difficulty_col="difficulty",
    prerequisite_by_concept={"fractions": ["number-sense"]},
)

next_items = rec.rank(
    user_id=42,
    candidate_item_ids=[101, 102, 137, 144],
    top_k=5,
)
rec.observe(user_id=42, item_id=next_items[0].item_id, correct=True)
```

`policy="auto"` uses the progression-value policy by default. Delayed-gain
policies are explicit opt-ins after reward-model diagnostics and OPE checks.

## Scenario 2: Safe adaptive rollout

Use this when adaptive recommendations are user-facing and you need a clear
fallback if progression metrics degrade.

**Signals:** live outcomes, logged policy scores or probabilities, category
labels, optional item difficulty, and a frozen baseline policy.

**Algorithm path:** offline policy evaluation, rolling progression metrics,
guardrail thresholds, and baseline fallback.

```python
from orchid_ranker import compare_logged_policies
from orchid_ranker.live_metrics import (
    GuardrailConfig,
    ProgressionGuardrail,
    RollingProgressionMonitor,
)

report = compare_logged_policies(
    logged_events,
    reward_col="progression_reward",
    propensity_col="logged_probability",
    target_probability_col="candidate_probability",
    baseline_probability_col="baseline_probability",
)

monitor = RollingProgressionMonitor(window_size=500, success_threshold=0.7)
guardrail = ProgressionGuardrail(
    monitor,
    GuardrailConfig(min_progression_gain=0.0, warmup_samples=50),
)

ranked = adaptive_rec.rank(user_id=42, candidate_item_ids=candidates, top_k=10)
if not guardrail.evaluate():
    ranked = baseline_rec.baseline_rank(
        user_id=42,
        candidate_item_ids=candidates,
        top_k=10,
    )
```

Use the same `candidate_item_ids` on both paths so the adaptive and fallback
policies rank the same eligible set.

## Scenario 3: Regulated training or clinical workflows

Use this when you need progression-aware ranking plus audit and privacy
controls: compliance training, certification, clinical rehab, or other
regulated workflows.

**Signals:** completion outcomes, competency labels, operator identity,
deployment metadata, and audit requirements.

**Algorithm path:** adaptive-learning recommender, guardrail policy, audit log,
and privacy hooks.

```python
from orchid_ranker import AdaptiveLearningRecommender, AuditLogger

rec = AdaptiveLearningRecommender(policy="auto").fit(
    outcomes,
    correct_col="completed",
    concept_col="competency",
    item_difficulty_col="difficulty",
)
logger = AuditLogger.from_env(path="orchid_audit.jsonl")

ranked = rec.rank(user_id=42, candidate_item_ids=eligible_items, top_k=5)
logger.log_event(
    "recommendation_served",
    actor="training-service",
    payload={
        "user_id": 42,
        "items": [item.item_id for item in ranked],
        "policy": rec.policy_name_,
    },
)
```

Keep a frozen baseline and use safe rollout checks when recommendations change
during a regulated session.

## Scenario 4: New-user cold start

Use this when brand-new users have no interaction history but you still need
reasonable recommendations from day one.

**Signals:** item feature matrix, optional popularity prior, early user
interactions, or stated preference seed items.

**Algorithm path:** content-similarity bridge, popularity prior, and
warmth-aware blend.

```python
from orchid_ranker.cold_start import ColdStartBridge, ColdStartConfig

bridge = ColdStartBridge(
    recommender=rec,
    item_features=item_feature_matrix,
    config=ColdStartConfig(min_interactions=3, blend_until=20),
)

first_recs = bridge.recommend(
    user_id=999,
    top_k=10,
    candidate_item_ids=catalog,
    seed_item_ids=[12, 44],
)
bridge.observe(user_id=999, item_id=first_recs[0][0], outcome=1.0)
```

This is a bridge, not a replacement for a mature recommender. Once the user
has enough interactions, the fitted recommender becomes the dominant score.

## Scenario 5: Batch catalog recommendations

Use this when you have historical interactions and want a simple recommender
inside a notebook, batch job, or service endpoint.

**Signals:** `user_id`, `item_id`, optional `rating` or binary label.

**Algorithm path:** ALS, explicit matrix factorization, neural MF, and baseline
ranking.

```python
from orchid_ranker import OrchidRecommender

rec = OrchidRecommender.from_interactions(
    interactions,
    strategy="als",
    user_col="user_id",
    item_col="item_id",
    rating_col="rating",
    epochs=3,
)

recs = rec.recommend(
    user_id=42,
    top_k=5,
    candidate_item_ids=[10, 12, 14, 20],
)
```

Use `strategy="auto"` when you want Orchid to choose `als` for binary feedback
or `explicit_mf` for wider rating ranges.

## Scenario 6: Generic streaming recommender

Use this when recommendations must adapt from live feedback, but the catalog
does not have learning concepts, difficulty, or prerequisite metadata yet.

**Signals:** historical interactions, live interaction events, and a candidate
item set.

**Algorithm path:** streaming recommender, rolling metrics when category labels
exist, and batch baseline fallback.

```python
streamer = rec.as_streaming()
ranked = streamer.rank(user_id=42, candidate_item_ids=candidates, top_k=10)
streamer.observe(user_id=42, item_id=ranked[0].item_id, outcome=1.0)
```

Prefer Scenario 1 when you can model actual learning state. This path is for
generic online recommendation.

## Scenario 7: Expertise-driven commerce

Use this when users develop taste or expertise over time: wine, photography,
coffee, fashion, audio gear, specialist tools, or similar categories.

**Signals:** purchases, returns, ratings, product category, sophistication score
such as price tier or expert rating.

**Algorithm path:** taste progression, stretch-fit scoring, momentum, and
exploration.

```python
from orchid_ranker.taste_progression import (
    SophisticationMapper,
    TasteConfig,
    TasteProgressionRanker,
)

sophistication = SophisticationMapper.from_prices(item_prices)
ranker = TasteProgressionRanker(
    recommender=rec,
    sophistication_scores=sophistication,
    config=TasteConfig(stretch_width=0.15),
)
ranker.set_item_categories(item_categories)
ranker.observe(
    user_id=42,
    item_id=501,
    purchased=True,
    returned=False,
    category="coffee",
    rating=4.5,
)
```

Do not use this for commodity basket-fill categories where there is no real
user trajectory.

## Scenario 8: Curated publication feed

Use this for editorial content where readers build topic expertise over time:
specialist newsletters, analyst briefings, technical publications, guided
reading lists, or course marketplaces.

**Signals:** topic, difficulty, publication time, meaningful engagement.

**Algorithm path:** freshness scoring, topic competence, stretch-fit ranking,
and topic diversity.

```python
import time
from orchid_ranker.curated_feed import FeedItem, FeedRanker, FreshnessScorer

ranker = FeedRanker(freshness=FreshnessScorer(halflife_hours=12))
candidates = [
    FeedItem(1, topic="ai-basics", difficulty=0.30, timestamp=time.time() - 1800),
    FeedItem(2, topic="ai-policy", difficulty=0.70, timestamp=time.time() - 3600),
    FeedItem(3, topic="markets", difficulty=0.50, timestamp=time.time() - 7200),
]

ranked = ranker.rank(user_id=42, candidates=candidates, top_k=3)
ranker.observe(user_id=42, item=ranked[0].item, engaged=True)
```

This is not a social-feed ranker. Use it when topic understanding and curated
progression matter more than raw CTR.

## Choosing quickly

| Need | Start with |
|------|------------|
| Adapt after every learner outcome | Scenario 1 |
| Prevent unsafe adaptive rollout | Scenario 2 |
| Add audit or compliance controls | Scenario 3 |
| Support users with no history | Scenario 4 |
| Fit and recommend from historical data | Scenario 5 |
| Adapt online without learning metadata | Scenario 6 |
| Recommend products along an expertise curve | Scenario 7 |
| Rank curated content with topic growth | Scenario 8 |
