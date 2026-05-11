"""Offline policy evaluation for an adaptive-learning policy.

Run with:
    PYTHONPATH=src python examples/offline_policy_evaluation_quickstart.py
"""
from __future__ import annotations

import json

import pandas as pd

from orchid_ranker.ope import compare_logged_policies, deterministic_policy_probabilities


def build_logged_events() -> pd.DataFrame:
    rows = []
    actions = ["review", "stretch"]
    for learner_id in range(60):
        needs_stretch = learner_id % 3 != 0
        target_action = "stretch" if needs_stretch else "review"
        for action in actions:
            reward = 1.0 if action == target_action else 0.0
            rows.append(
                {
                    "learner_id": learner_id,
                    "action": action,
                    "reward": reward,
                    "logging_propensity": 0.5,
                    "target_action": target_action,
                    "target_value": 1.0,
                    "baseline_value": 0.5,
                    "logged_action_value": reward,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    events = build_logged_events()
    events["target_probability"] = deterministic_policy_probabilities(
        events["action"].tolist(),
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
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
