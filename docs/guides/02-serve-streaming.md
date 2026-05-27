# Guide 2: Serve Adaptive Recommendations

Adaptive learning changes after every outcome. Serve Orchid by keeping a fitted
`AdaptiveRanker` in memory, logging every decision, and feeding observed
outcomes back through `observe(...)`.

## Serve A Candidate Set

```python
from orchid_ranker import LoggedDecision, stable_context_hash

learner_id = "learner-42"
concept_goal = "fractions"
candidates = ["frac-01", "frac-02", "frac-03"]
context_hash = stable_context_hash(learner_id, concept_goal)

ranked = ranker.recommend(
    learner_id,
    candidates,
    top_k=3,
    context_hash=context_hash,
    concept_goal=concept_goal,
)
```

## Log The Decision

Policy learning and OPE require candidate sets and propensities. Log the served
decision even when the reward arrives later.

```python
decision = LoggedDecision(
    learner_id=learner_id,
    ts=123456,
    candidate_item_ids=candidates,
    chosen_item_id=ranked[0].item_id,
    propensity=1.0,  # deterministic policy; use exploration probability when randomized
    policy_name="progression",
    policy_version="v1",
    scores=[rec.score for rec in ranked],
    context_hash=context_hash,
)
```

Persist the decision row in an append-only event log. When reward/outcome is
available, join it back by event ID or context/action fields before OPE.

## Observe Outcomes

```python
ranker.observe(
    learner_id=learner_id,
    ts=123500,
    item_id=ranked[0].item_id,
    concept_id=concept_goal,
    correct=1,
)
```

For SAINT+ backbones, the timestamp is forwarded into the live tracer state so
elapsed-time and lag-time features keep working after offline fit.

## Sketch Mode

For large catalogs, attach a sketch candidate generator and let Orchid shrink
the candidate set before final adaptive reranking.

```python
from orchid_ranker import ExactEmbeddingIndex, SketchCandidateGenerator

index = ExactEmbeddingIndex()
index.add("frac-01", [0.9, 0.1])
index.add("ratio-01", [0.1, 0.9])

ranker.attach_sketch_generator(SketchCandidateGenerator(ann_index=index))
ranked = ranker.recommend(
    learner_id,
    mode="sketch",
    concept_goal="fractions",
    item_query_vec=[1.0, 0.0],
    top_k=5,
)
```

## Production Loop

1. Fit offline with chronological data.
2. Serve recommendations from `AdaptiveRanker.recommend(...)`.
3. Log candidates, scores, chosen action, propensity, policy version, and context hash.
4. Call `observe(...)` after each outcome.
5. Run `ope_report(...)`, bootstrap OPE, and rollout gates before enabling a new learned policy.

Continue to [Guide 3: Operate safely in production](03-operate-safely.md).
