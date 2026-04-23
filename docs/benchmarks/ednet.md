# Case study: Learning platforms — EdNet

> **Domain:** Education (AI-based tutoring).
> EdNet is the second flagship case study alongside [OULAD](oulad.md).
> Together they demonstrate Orchid's progression-aware features on real
> educational data at two different scales.

## Dataset overview

[EdNet](https://github.com/riiid/ednet) is a large-scale interaction dataset
collected from *Santa*, an AI-powered English-proficiency tutoring application
developed by Riiid. It is one of the largest publicly available educational
interaction datasets.

| Dimension | Value |
|-----------|-------|
| Users | ~800,000 |
| Questions | ~13,000 |
| Interaction type | Timestamped question-response logs (correct/incorrect + elapsed time) |
| Metadata | Question part (exam section 1--7), content tags, platform, source |
| Side information | Per-user accuracy aggregates, activity velocity; per-item difficulty and tag diversity |

Each interaction records a user answering a specific question at a specific
time, with the response outcome and elapsed time. The combination of
fine-grained timestamps, correctness labels, and rich item metadata makes
EdNet ideal for evaluating adaptive question sequencing.

## How Orchid applies

Where OULAD exercises Orchid's ability to rank diverse VLE activities, EdNet
focuses on the tighter problem of **adaptive question sequencing with
competence tracking**:

1. **Competence estimation per category.** Each question belongs to one or
   more content tags and an exam part (1--7). Orchid's outcome-tracing model
   maintains per-category competence estimates that update with every
   response. The `mean_accuracy`, `decayed_accuracy`, and
   `recent_accuracy_15d` user features from the config feed the initial
   competence prior; the model refines it online.

2. **Stretch-zone-aware question selection.** Given a user's current
   competence profile, Orchid selects questions whose difficulty falls in
   the stretch zone — hard enough to be informative, easy enough to avoid
   frustration. The `difficulty` and `difficulty_recency` item features
   provide the difficulty signal, computed from historical correctness
   rates with recency decay.

3. **Temporal adaptation.** EdNet's timestamps span months of user activity.
   Orchid's recency-weighted features (`decayed_accuracy`,
   `recent_15d_interactions`, `avg_inter_event_ms`) allow the model to
   distinguish between a user who answered 50 questions yesterday and one
   who answered 50 questions six months ago. The competence estimates
   naturally decay and refresh.

4. **Next-question prediction.** Beyond ranking, Orchid can be evaluated on
   the knowledge-tracing task: given a user's history, predict which question
   they should see next to maximize long-term competence growth. This goes
   beyond standard next-item accuracy — it optimizes for the *right* next
   item, not just one the user is likely to click on.

## Orchid configuration

The reference configuration lives at `configs/ednet.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `datasets.ednet.users` | 2 categorical features (platform preference, source preference) + 10 numeric features (accuracy signals, activity span, velocity, response-time stats) |
| `datasets.ednet.items` | 4 categorical features (part, top tag, source, platform) + 7 numeric features (tag diversity, response time, difficulty, recency signals) |
| `datasets.ednet.sensitive` | Privacy controls — `platform_pref` and `source_pref` flagged as quasi-identifiers for sanitization |
| `datasets.ednet.difficulty` | Difficulty computed from correctness and elapsed-time aggregates with a 45-day halflife |

Run parameters default to 50 rounds of adaptive interaction.

## Expected metrics

| Metric | What it captures |
|--------|-----------------|
| **NDCG@10 (next-question)** | Ranking quality for next-question prediction on held-out interactions. Measures whether Orchid can identify the most appropriate next question. |
| **Competence estimation accuracy** | Correlation between Orchid's competence estimates and observed correctness rates on held-out questions. Validates the outcome-tracing model. |
| **Stretch-zone adherence** | Fraction of served questions whose difficulty falls within the user's estimated stretch zone. The key progression-aware metric. |
| **Engagement retention (Surv@N)** | Fraction of simulated sessions surviving at least N steps. Less central than for OULAD (EdNet sessions are assignment-driven), but still informative. |
| **Coverage** | Fraction of the question bank served to at least one user. Low coverage may indicate the model is stuck in a narrow difficulty band. |

## Why EdNet matters

EdNet complements OULAD in several ways:

| Dimension | OULAD | EdNet |
|-----------|-------|-------|
| Scale | ~32K users | ~800K users |
| Item type | Diverse VLE activities | Homogeneous questions |
| Interaction signal | Clicks (implicit) | Correct/incorrect (explicit outcome) |
| Temporal density | Weeks between interactions | Seconds to minutes |
| Primary challenge | Activity diversity | Difficulty calibration |

Together, the two datasets validate that Orchid's progression-aware approach
works across both ends of the education spectrum: sparse, heterogeneous
activity streams (OULAD) and dense, homogeneous question-response logs
(EdNet).

More importantly, the features that make Orchid effective on EdNet — competence
tracking, stretch-zone targeting, temporal adaptation — are not
education-specific. A music recommendation system faces an analogous problem:
users have evolving taste (competence), songs have complexity
(difficulty), and the optimal playlist balances familiarity with discovery
(stretch zone). The [MovieLens-1M benchmark](movielens-1m.md) explores this
generalization for movies; a planned
[music discovery benchmark](../roadmap/IMPLEMENTATION_PLAN.md) will test it
on listening data.

## Reproducibility

```bash
# Full adaptive benchmark (deterministic under fixed seed)
PYTHONPATH=src python benchmarks/run_agentic_adaptive.py

# With explicit parameters
PYTHONPATH=src python benchmarks/run_agentic_adaptive.py --rounds 80 --users 16 --items 64 --top-k 6
```

Results are written as JSONL logs under `runs/`. The configuration is fully
specified in `configs/ednet.yaml`; all seeds are fixed for reproducibility.

## References

- Choi, Y., Lee, Y., Shin, D., Cho, J., Park, S., Lee, S., ... & Heo, J.
  (2020). *EdNet: A Large-Scale Hierarchical Dataset in Education.*
  International Conference on Artificial Intelligence in Education (AIED).

## See also

- [Case study: OULAD](oulad.md) — the companion education case study, focused
  on VLE activity ranking.
- [MovieLens-1M benchmark](movielens-1m.md) — generalization to movie
  recommendations.
- [Benchmarking guide](../benchmarking.md) — how to run all benchmarks.
