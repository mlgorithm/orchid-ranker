# Usage scenarios

Orchid is an adaptive-learning engine. Every supported scenario should answer:
what should this learner work on next, why is it appropriate now, and how will
we know whether it improved progress?

## Choosing In Code

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

`available_scenarios()` returns the stable adaptive-learning catalog as
`ScenarioRecipe` objects.

## Scenario 1: Adaptive-Learning Next Item

Use this when each learner outcome should affect the next recommendation:
practice questions, tutoring exercises, certification tasks, onboarding steps,
rehab exercises, or skill-based review.

**Signals:** historical learner outcomes, item concepts, item difficulty,
optional prerequisites, and live outcome events.

**Algorithm path:** AKT/SAKT/SAINT-style knowledge tracing, progression-value
policy, prerequisite gating, and optional delayed-gain policy after diagnostics
and OPE.

```python
from orchid_ranker import AdaptiveLearningEngine

rec = AdaptiveLearningEngine(
    tracer_model="akt",
    policy="auto",
    epochs=2,
).fit(
    outcomes,
    user_col="learner_id",
    item_col="item_id",
    correct_col="correct",
    timestamp_col="timestamp",
    concept_col="skill_id",
    item_difficulty_col="difficulty",
    prerequisite_by_concept={"fractions": ["number-sense"]},
)

next_items = rec.rank(
    user_id="learner-42",
    candidate_item_ids=[101, 102, 137, 144],
    top_k=5,
)
rec.observe(user_id="learner-42", item_id=next_items[0].item_id, correct=True)
```

## Scenario 2: Safe Adaptive Rollout

Use this when recommendations are user-facing and you need a clear halt path if
progression metrics degrade.

**Signals:** logged outcomes, propensities or deterministic target
probabilities, progression reward, category labels, and a reviewed fallback
policy.

**Algorithm path:** offline policy evaluation, bootstrap confidence intervals,
rollout gates, rolling progression metrics, and guardrail thresholds.

```python
from orchid_ranker import compare_logged_policies, evaluate_rollout_gate

report = compare_logged_policies(
    logged_events,
    reward_col="progression_reward",
    propensity_col="logged_probability",
    target_probability_col="candidate_probability",
    baseline_probability_col="baseline_probability",
)

gate = evaluate_rollout_gate(report, min_effect=0.0, min_effect_lower_bound=-0.02)
if gate.passed:
    ranked = adaptive_rec.rank(user_id="learner-42", candidate_item_ids=candidates, top_k=10)
else:
    ranked = reviewed_prerequisite_policy.rank("learner-42", candidates, top_k=10)
```

Use the same eligible candidate set on both paths.

## Scenario 3: Regulated Training Or Clinical Workflow

Use this when adaptive ranking needs auditability and privacy controls:
compliance training, certification, clinical rehabilitation, or other regulated
progression workflows.

**Signals:** completion or correctness outcomes, competency labels, operator
identity, deployment metadata, and audit requirements.

**Algorithm path:** adaptive-learning recommender, progression guardrail,
hashed event IDs, audit logging, RBAC, and opt-in privacy hooks.

```python
from orchid_ranker import AdaptiveLearningEngine, AuditLogger

rec = AdaptiveLearningEngine(policy="auto").fit(
    outcomes,
    correct_col="completed",
    concept_col="competency",
    item_difficulty_col="difficulty",
)
logger = AuditLogger.from_env(path="orchid_audit.jsonl")

ranked = rec.rank(user_id="learner-42", candidate_item_ids=eligible_items, top_k=5)
logger.log_event(
    "recommendation_served",
    actor="training-service",
    payload={
        "learner_id": "learner-42",
        "items": [item.item_id for item in ranked],
        "policy": rec.policy_name_,
    },
)
```

## Scenario 4: New Learner Or New Exercise Cold Start

Use this when learner history is sparse or a new exercise has not appeared in
KT training data yet, but catalog metadata exists.

**Signals:** exercise text, concept labels, difficulty metadata, prerequisites,
and early learner outcomes.

**Algorithm path:** semantic exercise retrieval, prerequisite gating,
target-correctness priors, and progression-aware reranking.

```python
from orchid_ranker import AdaptiveRanker

ranker = AdaptiveRanker(kt_backbone="akt", policy="auto").fit_kt(
    outcomes,
    learner_col="learner_id",
    item_col="item_id",
    correct_col="correct",
    timestamp_col="ts",
    concept_col="concept_id",
    item_difficulty_col="difficulty",
)
ranker.fit_semantic_items(
    catalog,
    item_col="item_id",
    text_col="item_text",
    metadata_cols=["concept_id", "difficulty"],
)

ranked = ranker.recommend(
    "learner-42",
    top_k=5,
    item_query_text="fraction addition with unlike denominators",
)
```

Candidates outside the KT training universe can be returned with
`policy="semantic_cold_start"` when catalog metadata is available.

## Choosing Quickly

| Need | Start with |
|------|------------|
| Adapt after every learner outcome | Scenario 1 |
| Prevent unsafe adaptive rollout | Scenario 2 |
| Add audit, privacy, or compliance controls | Scenario 3 |
| Handle sparse learner history or new exercises | Scenario 4 |
