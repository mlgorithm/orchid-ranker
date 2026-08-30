# Reference pilots

These small API scenarios are retained as implementation examples, not as
Orchid's product positioning or a template for a learning-efficacy study. For
assessed learning, start with the [quickstart](quickstart.md), then follow the
[end-to-end reference-pilot workflow](guides/05-pilot-workflow.md). In
particular, generic onboarding and content-discovery examples below do not
establish that Orchid is a feed or product recommender.

These are compact, product-shaped templates—not extra Orchid models. Each one
uses the same `AdaptiveRanker` loop and makes the application responsible for
hard eligibility rules.

Run all three:

```bash
python examples/adaptive_recommender_use_cases.py
```

Or export the completed decision records in validation-ready JSON Lines:

```bash
python examples/adaptive_recommender_use_cases.py \
  --output-dir artifacts/reference-pilots
```

The supplied histories and exported records are simulated integration examples.
They demonstrate the schema and logging workflow; they are not evidence of
product improvement. Validate only with completed decisions from a real pilot.
The runnable script uses a deliberately small training budget so it finishes
quickly; a real pilot can begin with `AdaptiveRanker()` and its own history.

## Optional offline-policy promotion

The ordinary workflow does not need an offline policy. When enough completed,
randomized decisions have accumulated, run the safety-gated example:

```bash
python examples/offline_policy_promotion.py
```

It demonstrates two chronological decision windows and calls `fit_policy()`.
Orchid evaluates the deployed adaptive-base+CQL blend on the later window and
only promotes it when the rollout gate passes. The simulated result may be a
safe rejection; use real append-only completed decisions for any decision.

## B2B onboarding

The project ranks the next eligible activation step: connect data, create a
project, or invite a teammate. Its outcome is completion of the recommended
step within 24 hours.

The application filters completed steps and enforces prompt-frequency limits
before passing candidates to Orchid. A static checklist is the natural control
for the first live experiment.

## Compliance training

The project ranks modules that an employee is eligible to take. Its outcome is
passing the recommended module on the next attempt.

Prerequisites, certification rules, legal requirements, and role restrictions
stay in the application’s candidate filter. Orchid only orders the already
eligible modules.

## Content discovery

The project ranks eligible help or educational articles. Its outcome is reading
the recommended article to completion within 24 hours—not merely clicking it.

Locale, subscription, availability, and previously seen content are filtered
before ranking. This makes the adaptive objective align with useful discovery
instead of empty engagement.

## Move a pilot to real data

1. Replace the simulated history with chronological completed outcomes.
2. Keep the domain’s eligibility filter in the application.
3. Persist `recommend_and_log` decisions before responding.
4. Link outcomes through `observe_decision`.
5. Export completed logs as JSON Lines and follow [Validate a rollout](benchmarks/credibility.md).

Each pilot has the same public shape:

```python
ranker = AdaptiveRanker().fit(history)
ranked, decision = ranker.recommend_and_log(
    user_id=user_id,
    candidate_item_ids=eligible_items,
    timestamp=timestamp,
    exploration=0.05,
)
ranker.observe_decision(decision.decision_id, outcome=outcome, timestamp=observed_at)
```
