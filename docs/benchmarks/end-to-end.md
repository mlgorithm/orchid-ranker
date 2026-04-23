# End-to-end pipeline benchmark (MovieLens-1M)

> **Status:** auto-generated from `benchmarks/end_to_end_bench.py`.
> Full-run numbers (200 users, 30 steps) on MovieLens-1M.

## What this measures

The end-to-end benchmark measures the **full user lifecycle**: a user
arrives cold, the ColdStartBridge handles them, interactions accumulate,
Orchid's collaborative filtering takes over, and the TasteProgressionRanker
re-ranks for expertise-appropriate recommendations.

This is the integrated pipeline test — it proves the modules work
**together**, not just individually.

## Dataset

[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) --
1,000,209 ratings from 6,040 users on 3,883 movies.

| Dimension | Value |
|-----------|-------|
| Users | 6,040 |
| Items | 3,883 |
| Interactions | 1,000,209 |
| Rating scale | 1--5 (binarised: >= 4 is positive) |

Movie sophistication uses inverted log-popularity as a proxy
(niche movies = more sophisticated).

## Protocol

1. **Train/test split**: leave-one-out per user (last-in-time rating
   is held out as test).
2. **Click simulator**: 2-layer MLP trained on training data to
   predict P(click | user, item).
3. **Cold-start simulation**: all test users start with zero interactions.
   The replay loop recommends items one at a time; on each successful
   click, the system observes the interaction.
4. **Phase tracking**: interactions are bucketed into cold (0--5),
   transition (5--15), and warm (15+) phases.

## Systems compared

| System | Description |
|--------|-------------|
| Popularity | Global popularity baseline — same ranking for every user |
| Orchid direct | Raw OrchidRecommender with no cold-start handling |
| Bridge only | ColdStartBridge transitioning to Orchid, no taste re-ranking |
| Full pipeline | ColdStartBridge → Orchid → TasteProgressionRanker |

## Results

### Session survival

| System | Surv@5 | Surv@10 | Surv@20 | Mean session |
|--------|--------|---------|---------|-------------|
| Popularity         | 0.150 | 0.050 | 0.000 | 2.83 |
| Orchid direct      | 0.000 | 0.000 | 0.000 | 0.41 |
| Bridge only        | 0.270 | 0.060 | 0.000 | 3.29 |
| **Full pipeline**  | **0.275** | **0.045** | **0.005** | **3.23** |

### Kept-rate by phase

| System | Cold (0--5) | Transition (5--15) | Warm (15+) |
|--------|-----------|-------------------|-----------|
| Popularity         | 73.9% | 73.7% | 0.0% |
| Orchid direct      | 41.0% | 0.0% | 0.0% |
| Bridge only        | 77.9% | 73.2% | 60.0% |
| **Full pipeline**  | **77.8%** | **69.7%** | **92.9%** |

## Discussion

### The bridge is the biggest win

Surv@5 jumps from 0.150 (popularity) to 0.270 (bridge) — a **+80%
improvement**.  The ColdStartBridge solves a problem every
collaborative-filtering system has: what to do with users who have no
history.  Orchid direct scores 0.000 — it completely fails for cold
users.

### Taste progression's value appears in the warm phase

The headline survival numbers are similar between bridge-only and full
pipeline (0.270 vs 0.275).  But look at **warm-phase kept-rate**: the
full pipeline achieves **92.9% vs 60.0%** for bridge-only.  Once users
have established trajectories, taste-progression re-ranking
substantially improves what they actually keep.

The full pipeline is also the only system to achieve any Surv@20
(0.5%) — meaning some users stay engaged for 20+ interactions.

### Phase-by-phase story

1. **Cold phase (0--5)**: Bridge and full pipeline perform identically
   (77.8--77.9%).  Taste progression has no profile data yet, so it
   contributes nothing.  The bridge's popularity + content similarity
   signals carry the load.

2. **Transition phase (5--15)**: Bridge-only slightly outperforms
   full pipeline (73.2% vs 69.7%).  The taste ranker is still building
   profiles and occasionally makes suboptimal re-ranking decisions.

3. **Warm phase (15+)**: Full pipeline dramatically outperforms
   (92.9% vs 60.0%).  With sufficient interaction history, taste
   progression correctly targets the user's expertise level.

### Statistical caveat

The warm-phase sample is small (5--14 users reaching 15+ interactions),
so the 92.9% figure has high variance.  The direction is consistent
with the standalone taste-progression benchmark, but more users or
longer sessions would strengthen the statistical significance.

### Warmth transition

The bridge's blending weight ramps linearly:

| Interactions | Alpha |
|-------------|-------|
| 0--2 | 0.00 (pure cold-start) |
| 3 | 0.00 |
| 5 | 0.12 |
| 10 | 0.41 |
| 15 | 0.71 |
| 20 | 1.00 (pure Orchid) |

## Reproducibility

```bash
# Full run (~5 min, requires MovieLens-1M download)
PYTHONPATH=src python benchmarks/end_to_end_bench.py

# Quick smoke test (~1 min)
PYTHONPATH=src python benchmarks/end_to_end_bench.py --smoke
```

Results are written to `benchmarks/results_end_to_end.json`.

## References

- Harper, F. M. & Konstan, J. A. (2015). The MovieLens Datasets:
  History and Context. *ACM TIIS*, 5(4), 19:1--19:19.
