# Usage scenarios

This page turns Orchid's product thesis into concrete implementation recipes.
Start with the scenario closest to your product, then move to the linked guide
for the full workflow.

## Scenario 1: Batch catalog recommendations

Use this when you have historical interactions and want a simple recommender
inside a notebook, batch job, or service endpoint.

**Signals:** `user_id`, `item_id`, optional `rating` or binary label.

**Orchid path:** `OrchidRecommender.from_interactions()` -> `recommend()`.

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

eligible_items = [10, 12, 14, 20]
recs = rec.recommend(
    user_id=42,
    top_k=5,
    candidate_item_ids=eligible_items,
)
```

Use `strategy="auto"` when you want Orchid to choose `als` for binary feedback
or `explicit_mf` for wider rating ranges.

## Scenario 2: Live adaptive practice or onboarding

Use this when each outcome should affect the next recommendation immediately:
practice questions, product onboarding tasks, guided setup steps, rehab
exercises, or progression games.

**Signals:** historical interactions plus live outcome events.

**Orchid path:** fit `neural_mf` -> `as_streaming()` -> `observe()` -> `rank()`.

```python
from orchid_ranker import OrchidRecommender

rec = OrchidRecommender.from_interactions(
    interactions,
    strategy="neural_mf",
    rating_col="label",
    epochs=3,
    emb_dim=16,
)
streamer = rec.as_streaming(lr=0.05, l2=1e-3)

streamer.observe(
    user_id=42,
    item_id=137,
    correct=True,
    category="activation",
)

next_items = streamer.rank(
    user_id=42,
    candidate_item_ids=[101, 102, 137, 144],
    top_k=5,
)
```

The streaming bridge accepts the original `user_id` and `item_id` values from
your interactions DataFrame.

## Scenario 3: Safe adaptive rollout

Use this when adaptive recommendations are user-facing and you need a clear
fallback if progression metrics degrade.

**Signals:** live outcomes, category labels, optional item difficulty.

**Orchid path:** `RollingProgressionMonitor` + `ProgressionGuardrail` + frozen
baseline fallback.

This assumes `rec` is a fitted `neural_mf` recommender from Scenario 2.

```python
from orchid_ranker.live_metrics import (
    GuardrailConfig,
    ProgressionGuardrail,
    RollingProgressionMonitor,
)

monitor = RollingProgressionMonitor(
    window_size=500,
    total_categories={"setup", "activation", "advanced"},
    success_threshold=0.7,
    stretch_width=0.25,
)
guardrail = ProgressionGuardrail(
    monitor,
    GuardrailConfig(
        min_progression_gain=0.0,
        min_accept_rate=0.3,
        warmup_samples=50,
        consecutive_violations=3,
    ),
)
streamer = rec.as_streaming(monitor=monitor)

if guardrail.evaluate():
    ranked = streamer.rank(user_id=42, candidate_item_ids=candidates, top_k=10)
else:
    ranked = [
        (r.item_id, r.score)
        for r in rec.baseline_rank(
            user_id=42,
            top_k=10,
            candidate_item_ids=candidates,
        )
    ]
```

Use the same `candidate_item_ids` on both paths so the adaptive and fallback
policies rank the same eligible set.

## Scenario 4: New-user cold start

Use this when brand-new users have no interaction history but you still need
reasonable recommendations from day one.

**Signals:** item feature matrix, optional popularity prior, early user
interactions or stated preference seed items.

**Orchid path:** `ColdStartBridge` blends popularity/content similarity into
Orchid as the user warms up.

This assumes `rec` is a fitted batch recommender from Scenario 1 and
`item_feature_matrix` rows align to item IDs.

```python
from orchid_ranker.cold_start import ColdStartBridge, ColdStartConfig

bridge = ColdStartBridge(
    recommender=rec,
    item_features=item_feature_matrix,
    config=ColdStartConfig(min_interactions=3, blend_until=20),
)

# Brand-new user: popularity + content similarity.
first_recs = bridge.recommend(
    user_id=999,
    top_k=10,
    candidate_item_ids=catalog,
    seed_item_ids=[12, 44],
)

# Each outcome increases the user's warmth.
bridge.observe(user_id=999, item_id=first_recs[0][0], outcome=1.0)
print(bridge.warmth(999))
```

This is a bridge, not a replacement for a mature recommender: once a user has
enough interactions, Orchid becomes the dominant score.

## Scenario 5: Expertise-driven commerce

Use this when users develop taste or expertise over time: wine, photography,
coffee, fashion, audio gear, specialist tools, or similar categories.

**Signals:** purchases, returns, ratings, product category, sophistication score
such as price tier or expert rating.

**Orchid path:** `TasteProgressionRanker` combines relevance with taste level,
stretch fit, momentum, and exploration.

This assumes `rec` is a fitted batch recommender and `catalog` contains item
IDs from the same catalog.

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

recs = ranker.recommend(
    user_id=42,
    top_k=10,
    candidate_item_ids=catalog,
    category="coffee",
)
```

Do not use this for commodity basket-fill categories where there is no real
user trajectory.

## Scenario 6: Curated publication feed

Use this for editorial content where readers build topic expertise over time:
specialist newsletters, analyst briefings, technical publications, guided
reading lists, or course marketplaces.

**Signals:** topic, difficulty, publication time, meaningful engagement.

**Orchid path:** `FeedRanker` combines base relevance, freshness, stretch fit,
topic diversity, and topic competence.

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

## Scenario 7: Regulated training or clinical workflows

Use this when you need progression-aware ranking plus audit and privacy
controls: compliance training, certification, clinical rehab, or other
regulated workflows.

**Signals:** completion outcomes, category labels, operator identity,
deployment or audit metadata.

**Orchid path:** recommender + guardrail + DP/audit hooks.

```python
from orchid_ranker import AuditLogger, OrchidRecommender

rec = OrchidRecommender.from_interactions(
    interactions,
    strategy="als",
    rating_col="completed",
)
logger = AuditLogger.from_env(path="orchid_audit.jsonl")

recs = rec.recommend(user_id=42, top_k=5, candidate_item_ids=eligible_items)
logger.log_event(
    "recommendation_served",
    actor="training-service",
    payload={
        "user_id": 42,
        "items": [r.item_id for r in recs],
        "policy": "baseline",
    },
)
```

Add streaming and guardrails when recommendations change during the session;
keep the frozen batch baseline when auditability matters more than adaptation.

## Choosing quickly

| Need | Start with |
|------|------------|
| Fit and recommend from historical data | Scenario 1 |
| Adapt after every outcome | Scenario 2 |
| Prevent unsafe adaptive rollouts | Scenario 3 |
| Support users with no history | Scenario 4 |
| Recommend products along an expertise curve | Scenario 5 |
| Rank curated content with topic growth | Scenario 6 |
| Add audit/compliance controls | Scenario 7 |
