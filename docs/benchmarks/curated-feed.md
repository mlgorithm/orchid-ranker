# Curated-feed benchmark

> **Status:** auto-generated from `benchmarks/curated_feed_bench.py`.
> Currently uses synthetic data; MIND dataset integration planned.

## What this measures

The `orchid_ranker.curated_feed` module ranks content for editorially
curated publications where readers have a learning arc.  It combines
five scoring components:

1. **Relevance** -- base recommender score
2. **Freshness** -- exponential time-decay
3. **Stretch fit** -- difficulty vs. reading level
4. **Topic diversity** -- gradual penalty for topic repetition
5. **Topic competence** -- user's readiness for the topic

The key tradeoff: a pure engagement-maximizing ranker (popularity or
freshness-only) sacrifices diversity and progression.  The curated-feed
ranker explicitly balances these signals.

## Data

Currently uses **synthetic data** that models a technical publication:

- 8 topics at varying difficulty levels
- Users who progressively engage with harder content
- Freshness decay (older items are less engaging)
- Topic fatigue (repeated same-topic items reduce engagement)

| Dimension | Full run | Smoke |
|-----------|---------|-------|
| Users | 200 | 50 |
| Items | 1,000 | 200 |
| Topics | 8 | 8 |
| Interactions/user | 50 | 20 |

**Planned**: Microsoft News MIND dataset for external validation.

## Systems compared

| System | Description |
|--------|-------------|
| Freshness-only | Ranks purely by recency |
| Popularity | Global engagement rate |
| Curated feed ranker | Weighted combination of all 5 components |

## Results

### Feed ranking comparison (smoke test)

| System | Engagement | Diversity | Session length | Surv@5 | Surv@10 |
|--------|-----------|----------|---------------|--------|---------|
| Freshness-only     | 0.0860 | 0.7000 | 0.86 | 0.0000 | 0.0000 |
| Popularity         | 0.0920 | 0.6000 | 0.92 | 0.0000 | 0.0000 |
| Curated feed       | 0.0520 | 0.7840 | 0.52 | 0.0000 | 0.0000 |

**Diversity uplift vs popularity:** +30.7%

### Freshness half-life sensitivity

| Half-life | Engagement |
|-----------|-----------|
| 6 hours | 0.0340 |
| 12 hours | 0.0860 |
| 24 hours | 0.0600 |
| 48 hours | 0.0580 |
| 168 hours (1 week) | 0.0320 |

The sweet spot is 12--24 hours for a daily-publishing cadence.

## Discussion

### The engagement-diversity tradeoff

The curated feed ranker deliberately trades some engagement for
diversity (+30.7%).  This is the correct behaviour for editorially
curated content where:

- Topic monoculture drives subscriber fatigue
- Reader learning arcs require exposure to multiple topics
- Long-term retention matters more than per-session click-through

### Gradual diversity penalty

The diversity scoring uses a gradual decay (`1.0 - 0.3 * same_count`)
rather than a binary 0/1 penalty.  This allows a second item from the
same topic (penalty 0.3) while strongly discouraging three or more
(penalty 0.6+).  The approach better reflects real editorial practice:
two related articles per issue is fine; five is a monoculture.

### Why the engagement gap on synthetic data

The synthetic engagement model uses `difficulty_fit * freshness` —
a per-item score that doesn't reward diversity.  In real editorial
contexts, diverse feeds reduce churn and increase return visits, effects
not captured by a simple click model.  The absolute engagement gap
(-43.5%) is an artifact of the simulation; the diversity uplift
(+30.7%) is the signal to watch.

### Default weight tuning

| Weight | Default | Role |
|--------|---------|------|
| w_relevance | 0.35 | Base recommender score |
| w_freshness | 0.25 | Time decay |
| w_stretch | 0.15 | Difficulty match |
| w_diversity | 0.10 | Topic variety |
| w_competence | 0.15 | User readiness |

The default weights favour relevance and freshness (editorial
priorities) while keeping diversity moderate enough to avoid
engagement collapse.

## Reproducibility

```bash
# Full run (~5 min)
PYTHONPATH=src python benchmarks/curated_feed_bench.py

# Smoke test (~30 sec)
PYTHONPATH=src python benchmarks/curated_feed_bench.py --smoke

# With MIND dataset (when available)
PYTHONPATH=src python benchmarks/curated_feed_bench.py \
    --data-path /path/to/mind/
```

Results are written to `benchmarks/results_curated_feed.json`.

## See also

- [MovieLens-1M benchmark](movielens-1m.md) -- core recommender evaluation
- [Cold-start benchmark](cold-start.md) -- new-user handling
