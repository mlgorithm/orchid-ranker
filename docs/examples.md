# Use-Case Examples

These examples show where Orchid Ranker fits when the product question is
"what should this learner work on next?" The full runnable script is
[`examples/adaptive_learning_use_cases.py`](https://github.com/mlgorithm/orchid-ranker/blob/main/examples/adaptive_learning_use_cases.py).

Run it from the repository root:

```bash
python examples/adaptive_learning_use_cases.py
```

The cookbook uses only the torch-free core APIs. Install
`orchid-ranker[adaptive]` when you want `AdaptiveLearningEngine` to learn
knowledge-tracing state from historical learner outcomes.

## Compliance Training

Use Orchid when employees should move through a certification path in a valid
order. The prerequisite graph prevents the service from recommending
incident-response material before policy basics and phishing response are
complete.

```python
from orchid_ranker import DependencyGraph, ProgressionRecommender

graph = DependencyGraph([
    ("policy-basics", "secure-passwords"),
    ("policy-basics", "data-handling"),
    ("secure-passwords", "phishing-response"),
    ("data-handling", "incident-reporting"),
    ("phishing-response", "incident-reporting"),
])
difficulty = {
    "policy-basics": 0.10,
    "secure-passwords": 0.25,
    "data-handling": 0.35,
    "phishing-response": 0.45,
    "incident-reporting": 0.70,
}

recommender = ProgressionRecommender(graph, difficulty_map=difficulty)
next_modules = recommender.recommend({"policy-basics", "secure-passwords"}, n=3)
```

Use this pattern for compliance, professional certification, onboarding
checklists, and other structured training paths.

## Language-Learning Review

Use Orchid when the next item is a review, not a new lesson. The FSRS-style
scheduler ranks cards by forgetting urgency and updates memory state after a
review grade.

```python
from datetime import datetime, timedelta, timezone

from orchid_ranker import FSRSReviewState, FSRSScheduler

now = datetime.now(timezone.utc)
scheduler = FSRSScheduler(request_retention=0.90)
states = {
    "bonjour": FSRSReviewState(
        stability=2.0,
        difficulty=3.5,
        due_at=now - timedelta(hours=6),
        last_review_at=now - timedelta(days=4),
        repetitions=3,
    ),
    "se souvenir": FSRSReviewState(
        stability=1.2,
        difficulty=7.0,
        due_at=now - timedelta(days=1),
        last_review_at=now - timedelta(days=6),
        repetitions=1,
        lapses=1,
    ),
}

due_cards = scheduler.recommend_reviews(states, now=now, top_k=2)
updated_state = scheduler.review(states[due_cards[0].item_id], grade=3, now=now)
```

Use this pattern for vocabulary apps, exam prep, medical flashcards, and any
product where retention matters.

## Rehabilitation Progression

Use Orchid when a successful recommendation should be a safe stretch, not the
easiest task. `expected_progression_reward` scores candidates from predicted
success, difficulty, competence, repetition, and stretch fit.

```python
from orchid_ranker.progression_reward import expected_progression_reward

candidates = {
    "range-of-motion-review": {"p_correct": 0.95, "difficulty": 0.20, "competence": 0.55},
    "supported-step-up": {"p_correct": 0.72, "difficulty": 0.64, "competence": 0.55},
    "unassisted-balance-hop": {"p_correct": 0.26, "difficulty": 0.90, "competence": 0.55},
}

ranked = sorted(
    candidates,
    key=lambda item_id: expected_progression_reward(**candidates[item_id]).expected_reward,
    reverse=True,
)
```

Use this pattern for rehabilitation exercises, fitness progression,
instrument practice, and other skill-building workflows where "too easy" and
"too hard" are both bad recommendations.

## Onboarding Rollout Gate

Use Orchid before a new adaptive policy reaches users. Offline policy
evaluation estimates whether a proposed policy is better than a reviewed
baseline from logged randomized decisions.

```python
from orchid_ranker.ope import (
    compare_logged_policies,
    deterministic_policy_probabilities,
    evaluate_rollout_gate,
)

events["target_probability"] = deterministic_policy_probabilities(
    events["action"].tolist(),
    events["target_action"].tolist(),
)
events["baseline_probability"] = deterministic_policy_probabilities(
    events["action"].tolist(),
    events["baseline_action"].tolist(),
)

report = compare_logged_policies(
    events,
    reward_col="progression_reward",
    propensity_col="logging_probability",
    target_probability_col="target_probability",
    baseline_probability_col="baseline_probability",
)
gate = evaluate_rollout_gate(report, min_effect=0.05)
```

Use this pattern for employee onboarding, certification products, adaptive
tutoring, or any learner-facing system where a fallback policy must remain
available until the evidence is good enough.

## Which Example Should I Start With?

| Situation | Start with |
|-----------|------------|
| You have prerequisites and a structured path | Compliance training |
| You need to review old material before it is forgotten | Language-learning review |
| You need a safe stretch-zone recommendation | Rehabilitation progression |
| You need evidence before serving a new policy | Onboarding rollout gate |
| You have historical learner outcomes and want live updates | [Quickstart](quickstart.md) |
