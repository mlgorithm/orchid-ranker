# Offline policy evaluation

Offline policy evaluation is the gate between "this adaptive policy looks good
in a notebook" and "this policy is safe enough to test with learners." Orchid's
OPE module evaluates a candidate policy from logged randomized or exploration
traffic before the policy is served live.

Use it when you have logs with:

- the action that was shown, such as the exercise or lesson ID
- the observed reward, such as correctness, completion, retention, or gain
- the logging propensity, meaning the probability that the old policy assigned
  to the action it actually showed
- the candidate policy probability for that same logged action

## Estimators

`orchid_ranker.ope` reports:

| Estimate | Meaning | When to trust it |
|----------|---------|------------------|
| IPS | Inverse-propensity estimate | Useful when propensities are reliable and weights are not extreme |
| SNIPS | Self-normalized IPS | More stable when target-policy coverage is partial |
| Direct method | Model-predicted target-policy value | Useful as a model diagnostic, not enough by itself |
| Doubly robust | Model value plus propensity-corrected residual | Preferred when both propensities and value estimates are available |

The report also includes coverage, effective sample size, weight diagnostics,
and confidence intervals for the preferred estimate. Use
`bootstrap_logged_policy` or `bootstrap_compare_logged_policies` when normal
intervals look fragile, then pass the result to `evaluate_rollout_gate` before
serving the policy.

## Deterministic replay example

```python
import pandas as pd
from orchid_ranker.ope import compare_logged_policies, deterministic_policy_probabilities

events = pd.DataFrame(
    {
        "logged_action": ["review", "stretch", "review", "stretch"],
        "reward": [0.0, 1.0, 1.0, 0.0],
        "logging_propensity": [0.5, 0.5, 0.5, 0.5],
        "target_action": ["stretch", "stretch", "review", "review"],
        "target_value": [1.0, 1.0, 1.0, 1.0],
        "baseline_value": [0.5, 0.5, 0.5, 0.5],
        "logged_action_value": [0.0, 1.0, 1.0, 0.0],
    }
)

events["target_probability"] = deterministic_policy_probabilities(
    events["logged_action"].tolist(),
    events["target_action"].tolist(),
)
events["baseline_probability"] = 0.5

report = compare_logged_policies(
    events,
    reward_col="reward",
    propensity_col="logging_propensity",
    target_probability_col="target_probability",
    baseline_probability_col="baseline_probability",
    target_value_col="target_value",
    baseline_value_col="baseline_value",
    logged_action_value_col="logged_action_value",
)

print(report.uplift)

from orchid_ranker.ope import evaluate_rollout_gate

gate = evaluate_rollout_gate(report, min_effect=0.0, min_ess_fraction=0.05)
assert gate.allowed, gate.reasons
```

## Production checklist

- Log propensities for every served adaptive decision.
- Keep a frozen baseline policy and evaluate uplift against it.
- Track effective sample size; high raw event count does not help if the target
  policy almost never matches the logged action.
- Prefer doubly robust reports when a calibrated value model is available.
- Block rollout when the lower confidence bound is not positive, effective
  sample size is too low, coverage is too thin, or clipping is heavy.
- Do not ship a learning policy from offline AUC alone. Pair KT prediction
  quality with OPE on the recommendation action actually shown.
