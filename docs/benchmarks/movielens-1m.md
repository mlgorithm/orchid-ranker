# MovieLens-1M Benchmark

> **Status:** auto-generated scaffold. Real numbers are filled in by
> `python benchmarks/movielens_1m/run.py` and copied here after review.

## Dataset

[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) (Harper &
Konstan 2015). 1,000,209 ratings from 6,040 users on 3,706 movies.
Ratings on a 1–5 star scale.

## Preprocessing

| Step | Detail |
|------|--------|
| Binarisation | Rating >= 4 -> positive (label=1); else negative (label=0). All ratings kept. |
| Split | Global leave-one-out per user: last-in-time -> test. 10% stratified validation from remainder. |
| Item features | Genre multi-hot (18 dim) + year-decade one-hot (5 dim) + mean training rating (1 dim) = **24-dim vector**. |

## Systems compared

| System | Description |
|--------|-------------|
| **Popularity** | Items ranked by training-set positive-interaction count. No personalisation. |
| **Orchid Frozen (ALS)** | `OrchidRecommender(strategy="als")`, fit once on training data. Grid-searched over `n_factors` and `regularization`. |
| **Orchid Adaptive** | `OrchidRecommender(strategy="neural_mf")` + `.as_streaming()` for per-user online adaptation. Grid-searched over `lr` and `l2`. |
| **Implicit ALS** | `OrchidRecommender(strategy="implicit_als")`. Grid-searched over `factors`, `regularization`, `iterations`. |
| **Implicit BPR** | `OrchidRecommender(strategy="implicit_bpr")`. Grid-searched over `factors`, `learning_rate`. |

All hyperparameters grid-searched on validation NDCG@10. Orchid uses library
defaults plus a modest search grid — intentionally *not* over-tuned.

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **NDCG@10** | Classic ranking quality on the held-out test set. |
| **Surv@N** (N = 5, 10, 20) | Fraction of users whose simulated replay session survives at least N steps. *This is the headline metric.* |
| **Coverage** | Fraction of the full item catalog recommended to at least one user. |
| **Unique ratio** | Fraction of distinct items across all recommendation lists. |
| **Novelty** | Mean information-theoretic item unfamiliarity: −log₂(popularity / total_users). |

### Session-N survival protocol

A 2-layer MLP is trained on the training split to predict `P(click | user, item)`.
For each user, a recommender proposes items sequentially. The simulator samples
a Bernoulli click at each step. If the user clicks, the session continues
(the item is added to a seen set and the next recommendation is produced).
If the user does *not* click, the session ends. **Surv@10** measures the
fraction of users whose session lasted at least 10 steps.

This proxy captures a recommender's ability to sustain user engagement over
time — a system that is accurate *and* diverse will naturally achieve higher
survival rates.

## Results

<!-- Replace the placeholder table below with the actual results from
     `benchmarks/movielens_1m/results.json` after running the benchmark. -->

| System | NDCG@10 | Surv@5 | Surv@10 | Surv@20 | Mean Sess. | Coverage | Novelty |
|--------|---------|--------|---------|---------|------------|----------|---------|
| Popularity | — | — | — | — | — | — | — |
| Orchid Frozen (ALS) | — | — | — | — | — | — | — |
| Orchid Adaptive | — | — | — | — | — | — | — |
| Implicit ALS | — | — | — | — | — | — | — |
| Implicit BPR | — | — | — | — | — | — | — |

## Discussion

### What to look for

1. **Surv@10 ≥ 10% above best non-Orchid baseline** is the acceptance
   criterion from the implementation plan. If Orchid Adaptive achieves this,
   the generalisation thesis holds.
2. **NDCG@10 parity** — Orchid should be *close* to the best baseline on
   pure ranking quality. Winning here is a bonus, not the goal.
3. **Coverage and novelty** — Orchid's progression-aware scoring should
   naturally diversify recommendations, leading to higher coverage and
   novelty than pure accuracy-optimised systems.

### Failure modes

- If Orchid Adaptive is worse than Orchid Frozen on Surv@N, the online
  adapter may be over-fitting on too few interactions per user.
- If all systems cluster tightly on all metrics, MovieLens-1M may not have
  enough sequential signal to differentiate progression-aware ranking.
  See §5.4 of the implementation plan for the fallback strategy.

## Reproducibility

```bash
# Full run (deterministic under fixed seed)
PYTHONPATH=src python benchmarks/movielens_1m/run.py --seed 42

# Smoke test (< 5 min, for CI)
PYTHONPATH=src python benchmarks/movielens_1m/run.py --smoke
```

All seeds are fixed. Results are written to
`benchmarks/movielens_1m/results.json` with the full configuration.

## References

- Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History
  and Context.* ACM Transactions on Interactive Intelligent Systems, 5(4), 19.
