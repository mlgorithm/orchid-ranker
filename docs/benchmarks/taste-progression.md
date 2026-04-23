# Taste-progression benchmark

> **Status:** auto-generated from `benchmarks/taste_progression_bench.py`.
> Multi-domain numbers on Amazon Reviews (Cell Phones 1.1M reviews,
> Digital Music 169K reviews).

## What this measures

The `orchid_ranker.taste_progression` module models users who develop
expertise and taste over time in product domains like wine, cameras,
photography, fragrances, and specialty coffee.  The module reinterprets
Orchid's core concepts:

- "correct" = **kept** (the user kept and was satisfied with the item)
- "difficulty" = **sophistication** (product tier / complexity)
- "stretch zone" = next sophistication tier the user is ready for

The headline metric is **kept-rate uplift**: how much more often do
users keep items recommended by the taste-progression ranker compared
to popularity baselines?

## Data

### Amazon Cell Phones & Accessories (primary)

[Amazon Reviews 2018](https://jmcauley.ucsd.edu/data/amazon_v2/) --
Cell Phones & Accessories category (5-core subset).  Chosen because
phones have clear price-based tier structure ($100 budget → $300
mid-range → $600 mid-premium → $1000+ flagship).

| Dimension | Value |
|-----------|-------|
| Users | 157,190 |
| Items | 48,120 |
| Reviews | 1,128,437 |
| Train | 848,921 |
| Test | 279,321 |

Sophistication is derived from price quantile rank within category.

### Amazon Digital Music (comparison)

| Dimension | Value |
|-----------|-------|
| Users | 16,510 |
| Items | 11,669 |
| Reviews | 169,387 |
| Train | 129,455 |
| Test | 39,932 |

Included to demonstrate the effect of domain signal quality.
Digital music has weak price-based sophistication gradients.

## Systems compared

| System | Description |
|--------|-------------|
| Popularity | Global kept-rate ranking |
| Recent popularity | Windowed popularity (last 100 interactions) |
| TasteProgressionRanker | 4-component scoring: relevance (0.40), stretch-fit (0.35), momentum (0.15), exploration (0.10) |

## Results

### Cross-domain comparison

| Domain | Users | Reviews | Stretch accuracy | Kept-rate uplift |
|--------|-------|---------|-----------------|-----------------|
| **Cell Phones** | 157K | 1.1M | **40.6%** | **+0.9%** |
| Digital Music | 16.5K | 169K | 5.1% | -4.0% |

**Key finding:** Domain signal quality determines whether taste
progression helps or hurts.  Cell Phones (clear price tiers) shows
8x better stretch accuracy and positive uplift.  Digital Music (weak
price gradients) shows negative uplift — the ranker makes bad calls
based on a bad signal.

### Cell Phones: kept-rate comparison

| System | Kept rate | Hit@10 | NDCG@10 |
|--------|----------|--------|---------|
| Popularity         | 0.8033 | 0.5948 | 0.6354 |
| Recent popularity  | 0.7813 | 0.8665 | 0.9396 |
| **TasteProgression** | **0.8108** | 0.3314 | 0.5006 |

**Kept-rate uplift vs popularity: +0.9%**

### Digital Music: kept-rate comparison

| System | Kept rate | Hit@10 | NDCG@10 |
|--------|----------|--------|---------|
| Popularity         | 0.9444 | 0.5537 | 0.5922 |
| Recent popularity  | 0.9398 | 0.9684 | 0.8853 |
| TasteProgression   | 0.9064 | 0.0554 | 0.5224 |

**Kept-rate uplift vs popularity: -4.0%**

### Progression curve (Cell Phones)

The ranker correctly models user expertise development via an
EMA-blended taste level:

- **Start**: mean taste level = 0.508 (mid-range)
- **End**: mean taste level = 0.651 (developing expertise)

### Stretch zone accuracy

| Domain | Stretch accuracy | Interpretation |
|--------|-----------------|----------------|
| Cell Phones | **0.4061** | 40.6% of test items fall within predicted stretch zone |
| Digital Music | 0.0514 | 5.1% — price is a poor sophistication proxy for music |

## Discussion

### Standalone vs. re-ranker performance

The +0.9% standalone uplift on Cell Phones is modest.  The taste
progression ranker's real value appears when used as a **re-ranker on
top of collaborative filtering**.  In the [end-to-end benchmark](end-to-end.md),
the full pipeline (ColdStartBridge + Orchid + TasteProgression) achieves
**92.9% kept-rate in the warm phase** vs 60.0% for the bridge alone.

The standalone benchmark measures taste progression without any
collaborative filtering backing — it ranks purely by stretch-zone fit.
This optimises for item quality (will the user keep it?) not item
discovery (will the user find it?), which explains the low Hit@10.

### Why domain signal quality matters

The 8x difference in stretch accuracy between Cell Phones (40.6%)
and Digital Music (5.1%) shows that the algorithm works — the
bottleneck is the **sophistication signal**, not the model.

Price quantile rank is a reasonable sophistication proxy for:
- Electronics (budget → flagship tiers)
- Wine (table → reserve → rare vintages)
- Photography equipment (kit lens → professional)

It is a poor proxy for:
- Digital music (a $9.99 album isn't meaningfully "harder" than $7.99)
- Books (price doesn't correlate with reading difficulty)
- Streaming content (no price signal)

For domains without clear price tiers, sophistication should come from
domain-specific signals: expert ratings, feature complexity scores,
prerequisite structure, or community-derived difficulty rankings.

### EMA sophistication tracking

The `TasteProfile` blends two signals for taste level:

- **EMA of consumed sophistication** (60% weight) -- tracks *where*
  the user is on the sophistication scale based on actual purchases
- **BKT p_known** (40% weight) -- tracks *confidence* that the user has
  succeeded at their current tier

This hybrid approach fixes the scale mismatch between BKT's non-linear
p_known and the linear sophistication scores used for stretch-zone
computation.

### Why Hit@10 is low

The taste-progression ranker has no collaborative filtering component
in this benchmark (no Orchid recommender backing it).  It ranks purely
by stretch-zone fit, which optimises for kept-rate (item quality) rather
than hit-rate (item discovery).  In production, the ranker wraps an
Orchid recommender and combines both signals.

### Evaluation methodology

Each test user is evaluated against a candidate set of their test
items plus 50 random negatives.  All systems rank the same candidate
set, ensuring a fair comparison.  This prevents the stretch-zone
ranker from being penalised for recommending items outside a narrow
holdout set.

## Reproducibility

```bash
# Full run on Amazon Cell Phones (~4 min)
PYTHONPATH=src python benchmarks/taste_progression_bench.py --data-path auto --amazon-category CellPhones

# Full run on Amazon Digital Music (~10 sec)
PYTHONPATH=src python benchmarks/taste_progression_bench.py --data-path auto --amazon-category DigitalMusic

# Smoke test with synthetic data (~5 sec)
PYTHONPATH=src python benchmarks/taste_progression_bench.py --smoke

# Smoke test on Amazon Reviews (~10 sec)
PYTHONPATH=src python benchmarks/taste_progression_bench.py --smoke --data-path auto
```

Results are written to `benchmarks/results_taste_progression.json`.

## References

- Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations
  using distantly-labeled reviews and fine-grained aspects. *EMNLP*.
