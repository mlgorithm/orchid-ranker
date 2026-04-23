# Case study: Learning platforms — OULAD

> **Domain:** Education (online university).
> Orchid's progression-aware features originated in this domain.
> OULAD is one of two flagship case studies; see also [EdNet](ednet.md).

## Dataset overview

The [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset)
contains interaction logs from the Open University (UK), one of the largest
distance-learning institutions in the world.

| Dimension | Value |
|-----------|-------|
| Users | ~32,000 |
| Courses | 7 (across multiple presentations) |
| Assessment types | 22 |
| Interaction type | VLE click-stream (page views, forum posts, resource downloads) |
| Side information | Demographics, prior attempts, credit load, final result |

Each record ties a user to a VLE activity within a dated presentation of a
course module. The combination of timestamped click-stream data, structured
assessments, and rich user metadata makes OULAD an ideal proving ground for
progression-aware ranking.

## How Orchid applies

Orchid treats every VLE activity as an *item* and every user's click-stream
as an interaction history. Three features are particularly relevant here:

1. **Competence tracking via outcome tracing.** Orchid maintains a per-user,
   per-category competence estimate that updates after every interaction.
   In the OULAD context, categories correspond to course modules and
   activity types. The outcome-tracing model captures how a user's
   competence evolves across a presentation, using decayed success signals
   (`decayed_success`, `recent_success_4w`) from the config.

2. **Stretch-zone targeting.** Rather than serving the easiest or most
   popular activities, Orchid's scoring function biases toward items whose
   difficulty falls inside the user's current stretch zone — challenging
   enough to promote growth, but not so hard as to cause disengagement.
   The `difficulty` field in the item features (historical difficulty
   computed from logs) feeds directly into this mechanism.

3. **Progression-aware sequencing.** The structured catalog of 7 courses
   and 22 assessment types provides a natural progression graph. Orchid's
   adaptive policy respects prerequisite ordering and paces items according
   to estimated competence, rather than simply maximizing predicted click
   probability.

## Orchid configuration

The reference configuration lives at `configs/oulad.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `datasets.oulad.users` | 7 categorical features (gender, region, education, IMD band, age band, disability, final result) + 12 numeric features (prior attempts, credits, click velocity, competence signals) |
| `datasets.oulad.items` | 3 categorical features (activity type, module, presentation) + 9 numeric features (week offset, duration, click stats, difficulty, recency signals) |
| `datasets.oulad.sensitive` | Privacy controls — strong identifiers (region, age band, disability, gender, IMD band) are flagged for sanitization |
| `datasets.oulad.difficulty` | Difficulty computed from logs with a 45-day recency halflife |

Run parameters default to 50 rounds of adaptive interaction.

## Expected metrics

| Metric | What it captures |
|--------|-----------------|
| **Engagement retention (Surv@N)** | Fraction of users whose simulated session survives at least N steps. The headline metric — measures whether Orchid sustains engagement over time. |
| **Assessment completion rate** | Proportion of recommended assessments that the user actually completes. Higher is better, but only if difficulty is appropriate. |
| **Stretch-zone adherence** | Fraction of served items whose difficulty falls within the user's estimated stretch zone. Tracks whether Orchid is targeting the right challenge level. |
| **NDCG@10** | Standard ranking quality on held-out interactions. Orchid aims for parity with baselines here; the win is on the metrics above. |
| **Coverage** | Fraction of the activity catalog recommended to at least one user. Progression-aware scoring should naturally diversify. |

## Why OULAD is the home turf

Education is the domain where progression-aware ranking is most obviously
valuable. Users have a latent competence that changes over time; items have
an intrinsic difficulty; and the optimal recommendation depends on the gap
between the two. Every feature Orchid provides — outcome tracing, stretch-zone
targeting, structured catalog sequencing, competence tracking — maps directly
onto the educational problem.

OULAD, with its longitudinal click-stream data and structured assessments,
exercises the full Orchid pipeline:

- Competence estimates update after every VLE interaction.
- Difficulty is computed from historical success rates with recency decay.
- The stretch zone adapts as the user progresses through a presentation.
- Side-information features (prior attempts, credit load, activity velocity)
  enrich the two-tower model's user representations.

That said, the same approach generalizes. The [MovieLens-1M](movielens-1m.md)
benchmark demonstrates that progression-aware scoring improves long-term
engagement even in a domain (movie recommendations) where "competence" maps
to evolving taste rather than knowledge. A planned
[music discovery benchmark](../roadmap/IMPLEMENTATION_PLAN.md) will extend
the story further. The thesis is not "Orchid is an education tool" — it is
"Orchid is a progression-aware recommender that started in education because
that is where the need is most legible."

## Reproducibility

```bash
# Full adaptive benchmark (deterministic under fixed seed)
PYTHONPATH=src python benchmarks/run_agentic_adaptive.py

# With explicit round/user/item counts
PYTHONPATH=src python benchmarks/run_agentic_adaptive.py --rounds 80 --users 16 --items 64 --top-k 6
```

Results are written as JSONL logs under `runs/`. The configuration is fully
specified in `configs/oulad.yaml`; all seeds are fixed for reproducibility.

## References

- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning
  Analytics dataset.* Scientific Data, 4, 170171.

## See also

- [Case study: EdNet](ednet.md) — the companion education case study, focused
  on adaptive question sequencing.
- [MovieLens-1M benchmark](movielens-1m.md) — generalization to movie
  recommendations.
- [Benchmarking guide](../benchmarking.md) — how to run all benchmarks.
