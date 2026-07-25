# Evidence and rollout

Orchid is a library, not a hosted service. A successful example, fixture, or
offline score does not prove that it improves an application's outcome.

The credible unit of evidence is a versioned policy evaluated on its own
chronological, completed decision log against a simple static baseline.

## Capture the evidence

Use the production decision loop. It records the exact eligible candidate set,
the selected item, and the probability of selecting it.

```python
ranked, decision = ranker.recommend_and_log(
    user_id=user_id,
    candidate_item_ids=eligible_items,
    timestamp=timestamp,
    exploration=0.05,
    policy_version="onboarding-v3",
)

# Persist `decision` before returning a result.
ranker.observe_decision(decision.decision_id, outcome=outcome, timestamp=observed_at)
```

After outcomes arrive, export the completed records as JSON Lines. This format
preserves candidate lists and action probabilities without ambiguous CSV
string parsing.

```python
completed = ranker.decision_log_frame(completed_only=True)
completed.to_json("artifacts/onboarding-v3.jsonl", orient="records", lines=True)
```

Do not discard candidate sets, propensities, policy versions, or decisions with
missing outcomes. Without those fields, a policy-value claim is not supported.

## Run one reproducible comparison

From an Orchid source checkout, run:

```bash
PYTHONPATH=src python benchmarks/validate_logged_policy.py \
  --data artifacts/onboarding-v3.jsonl \
  --policy-version onboarding-v3 \
  --output artifacts/onboarding-v3-validation.json \
  --report-md artifacts/onboarding-v3-validation.md \
  --require-pass
```

The command makes a strictly chronological split. It learns a
propensity-weighted, smoothed static item-outcome baseline from the earlier
window only, then estimates both that baseline and the logged adaptive behavior
on the later window. Its uncertainty interval is bootstrapped by user, so
repeated decisions from one user are not treated as independent evidence.

The report is deliberately allowed to be **inconclusive**. The evidence gate
rejects a rollout claim when the evaluation window is too small, the baseline
lacks support in the randomized logs, propensities require excessive clipping,
or the lower bound of the uplift interval fails to clear the required effect.

## What a pass means

A pass means the log supports a **controlled rollout** of that exact policy
version for the same population, candidate construction, and outcome. It does
not prove a causal benefit in a new product, for a different policy version,
or beyond the observed time window.

Use a small randomized live experiment with a retained static control before
claiming improvement. Monitor outcome coverage, calibration, candidate-set
changes, and regressions for important user groups throughout the rollout.

## Claim discipline

Good: “On completed decisions from the stated period, `onboarding-v3` cleared
the pre-specified logged-policy evidence gate against the documented static
baseline.”

Not supported: “Orchid improves onboarding,” “the algorithm is better,” or a
claim transferred from fixture data, synthetic logs, or another application.
