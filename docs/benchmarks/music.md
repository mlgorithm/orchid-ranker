# Last.fm 1K Music Benchmark

> **Status:** auto-generated scaffold. Real numbers are filled in by
> `python benchmarks/music/run.py` and copied here after review.

## Dataset

[Last.fm Dataset 1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)
(Celma 2010). Approximately 19 million timestamped listening events from
~1,000 users. Columns: userid, timestamp, musicbrainz-artist-id,
artist-name, musicbrainz-track-id, track-name. Purely implicit feedback
(listen events serve as positive signals).

## Preprocessing

| Step | Detail |
|------|--------|
| Deduplication | Each unique (user, track) pair collapsed into one interaction. Play count preserved as a feature. Timestamp = last listen time. |
| Track key | MusicBrainz track ID when available; otherwise `artist_name::track_name` composite key. |
| Filtering | Users with < 20 interactions removed. Tracks with < 10 interactions removed. Iterative until convergence. |
| Implicit label | All interactions are positive (label=1). Negatives sampled during simulator training. |
| Split | Global leave-one-out per user: last-in-time interaction -> test. 10% stratified validation from remainder. |
| Item features | Top-200 artist one-hot (201 dims, with "other" bucket) + log-scaled play count (1 dim) = **202-dim vector**. |

## Systems compared

| System | Description |
|--------|-------------|
| **Popularity** | Tracks ranked by training-set interaction count. No personalisation. |
| **Orchid Frozen (ALS)** | `OrchidRecommender(strategy="als")`, fit once on training data. Grid-searched over `n_factors` and `regularization`. |
| **Orchid Adaptive** | `OrchidRecommender(strategy="neural_mf")` + `.as_streaming()` for per-user online adaptation. Grid-searched over `lr` and `l2`. |
| **Implicit ALS** | `OrchidRecommender(strategy="implicit_als")`. Grid-searched over `factors`, `regularization`, `iterations`. |
| **Implicit BPR** | `OrchidRecommender(strategy="implicit_bpr")`. Grid-searched over `factors`, `learning_rate`. |

All hyperparameters grid-searched on validation NDCG@10. Orchid uses library
defaults plus a modest search grid -- intentionally *not* over-tuned.

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **NDCG@10** | Classic ranking quality on the held-out test set. |
| **Surv@N** (N = 5, 10, 20) | Fraction of users whose simulated replay session survives at least N steps. *This is the headline metric.* |
| **Coverage** | Fraction of the full item catalog recommended to at least one user. |
| **Unique ratio** | Fraction of distinct items across all recommendation lists. |
| **Novelty** | Mean information-theoretic item unfamiliarity: -log2(popularity / total_users). |

### Session-N survival protocol

A 2-layer MLP is trained on the training split (with negative sampling) to
predict `P(engage | user, item)`. For each user, a recommender proposes tracks
sequentially. The simulator samples a Bernoulli engagement at each step. If
the user engages, the session continues (the track is added to a seen set and
the next recommendation is produced). If the user does *not* engage, the
session ends. **Surv@10** measures the fraction of users whose session lasted
at least 10 steps.

This proxy captures a recommender's ability to sustain user engagement over
time -- a system that is accurate *and* diverse will naturally achieve higher
survival rates. For music this is particularly relevant: a good playlist
keeps the user listening, while a bad one causes them to skip away.

## Results

<!-- Replace the placeholder table below with the actual results from
     `benchmarks/music/results.json` after running the benchmark. -->

| System | NDCG@10 | Surv@5 | Surv@10 | Surv@20 | Mean Sess. | Coverage | Novelty |
|--------|---------|--------|---------|---------|------------|----------|---------|
| Popularity | -- | -- | -- | -- | -- | -- | -- |
| Orchid Frozen (ALS) | -- | -- | -- | -- | -- | -- | -- |
| Orchid Adaptive | -- | -- | -- | -- | -- | -- | -- |
| Implicit ALS | -- | -- | -- | -- | -- | -- | -- |
| Implicit BPR | -- | -- | -- | -- | -- | -- | -- |

## Discussion

### What to look for

1. **Surv@10 >= 10% above best non-Orchid baseline** is the acceptance
   criterion from the implementation plan. If Orchid Adaptive achieves this,
   the generalisation thesis holds across domains (movies -> music).
2. **NDCG@10 parity** -- Orchid should be *close* to the best baseline on
   pure ranking quality. Winning here is a bonus, not the goal.
3. **Coverage and novelty** -- Orchid's progression-aware scoring should
   naturally diversify recommendations, leading to higher coverage and
   novelty than pure accuracy-optimised systems. This is especially important
   for music, where discovery of new artists drives long-term engagement.
4. **Cross-domain consistency** -- comparing this table with the MovieLens-1M
   results shows whether Orchid's advantage generalises from explicit-rating
   movie data to implicit-feedback music data.

### Music-specific considerations

- **Long-tail distribution**: music listening follows an extreme power law.
  Most tracks have very few listeners, while a handful dominate. Coverage
  and novelty metrics are especially informative here.
- **Temporal patterns**: users' music taste shifts over time (seasonal,
  mood-based). The leave-one-out-by-last-timestamp split tests whether
  recommenders can anticipate evolving preferences.
- **Implicit feedback only**: unlike MovieLens (1-5 star ratings), there
  are no explicit dislikes. The negative sampling strategy for the simulator
  introduces an assumption that non-listened tracks are less preferred.

### Failure modes

- If Orchid Adaptive is worse than Orchid Frozen on Surv@N, the online
  adapter may be over-fitting on too few interactions per user.
- If popularity dominates on NDCG, the dataset may be too sparse after
  filtering -- consider relaxing the minimum interaction thresholds.
- If all systems cluster tightly on all metrics, the implicit feedback
  signal may be too noisy. See the implementation plan for the fallback
  strategy (use Last.fm 360K for more data).

## Reproducibility

```bash
# Full run (deterministic under fixed seed)
PYTHONPATH=src python benchmarks/music/run.py --seed 42

# Smoke test (< 5 min, for CI)
PYTHONPATH=src python benchmarks/music/run.py --smoke
```

All seeds are fixed at 42. Results are written to
`benchmarks/music/results.json` with the full configuration.

## References

- Celma, O. (2010). *Music Recommendation and Discovery -- The Long Tail,
  Long Fail, and Long Play in the Digital Music Space.* Springer.
- http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
