from __future__ import annotations

import math

import pandas as pd
import pytest

from orchid_ranker.ope import (
    compare_logged_policies,
    deterministic_policy_probabilities,
    evaluate_logged_policy,
)


def _paired_uniform_log() -> pd.DataFrame:
    rows = []
    for context_id in range(40):
        target_action = "stretch" if context_id % 2 == 0 else "review"
        other_action = "review" if target_action == "stretch" else "stretch"
        for action in [target_action, other_action]:
            reward = float(action == target_action)
            rows.append(
                {
                    "context_id": context_id,
                    "action": action,
                    "reward": reward,
                    "propensity": 0.5,
                    "target_prob": float(action == target_action),
                    "baseline_prob": 0.5,
                    "target_value": 1.0,
                    "baseline_value": 0.5,
                    "logged_action_value": reward,
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_logged_policy_returns_ips_snips_and_dr_estimates():
    report = evaluate_logged_policy(
        _paired_uniform_log(),
        reward_col="reward",
        propensity_col="propensity",
        target_probability_col="target_prob",
        target_value_col="target_value",
        logged_action_value_col="logged_action_value",
    )

    assert report.n_events == 80
    assert report.logging_reward == 0.5
    assert report.coverage == 0.5
    assert report.effective_sample_size == 40.0
    assert report.ips == 1.0
    assert report.snips == 1.0
    assert report.direct_method == 1.0
    assert report.doubly_robust == 1.0
    assert report.estimator == "doubly_robust"
    assert report.value == 1.0
    assert report.to_dict()["value"] == 1.0


def test_compare_logged_policies_reports_paired_uplift():
    comparison = compare_logged_policies(
        _paired_uniform_log(),
        reward_col="reward",
        propensity_col="propensity",
        target_probability_col="target_prob",
        baseline_probability_col="baseline_prob",
        target_value_col="target_value",
        baseline_value_col="baseline_value",
        logged_action_value_col="logged_action_value",
    )

    assert comparison.estimator == "doubly_robust"
    assert comparison.target.value == 1.0
    assert comparison.baseline.value == 0.5
    assert comparison.uplift == 0.5
    assert comparison.ci_low <= comparison.uplift <= comparison.ci_high
    assert comparison.to_dict()["uplift"] == 0.5


def test_snips_is_preferred_without_value_model():
    report = evaluate_logged_policy(
        _paired_uniform_log(),
        reward_col="reward",
        propensity_col="propensity",
        target_probability_col="target_prob",
    )

    assert report.estimator == "snips"
    assert report.value == 1.0
    assert report.doubly_robust is None
    assert report.direct_method is None


def test_invalid_propensity_raises_clear_error():
    frame = _paired_uniform_log()
    frame.loc[0, "propensity"] = 0.0

    with pytest.raises(ValueError, match="propensity"):
        evaluate_logged_policy(
            frame,
            reward_col="reward",
            propensity_col="propensity",
            target_probability_col="target_prob",
        )


def test_weight_clipping_reports_clipped_fraction():
    frame = pd.DataFrame(
        {
            "reward": [1.0, 0.0, 1.0],
            "propensity": [0.01, 0.5, 0.5],
            "target_prob": [1.0, 1.0, 0.0],
        }
    )

    unclipped = evaluate_logged_policy(
        frame,
        reward_col="reward",
        propensity_col="propensity",
        target_probability_col="target_prob",
        min_propensity=0.01,
        reward_min=0.0,
        reward_max=1.0,
    )
    clipped = evaluate_logged_policy(
        frame,
        reward_col="reward",
        propensity_col="propensity",
        target_probability_col="target_prob",
        min_propensity=0.01,
        max_weight=10.0,
    )

    assert unclipped.weight_max == 100.0
    assert clipped.weight_max == 10.0
    assert math.isclose(clipped.clipped_fraction, 1 / 3)


def test_deterministic_policy_probabilities():
    probs = deterministic_policy_probabilities(
        logged_actions=["a", "b", "c"],
        policy_actions=["a", "a", "c"],
    )

    assert probs.tolist() == [1.0, 0.0, 1.0]


def test_deterministic_policy_probabilities_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        deterministic_policy_probabilities(["a"], ["a", "b"])


def test_to_dict_serializes_unsupported_snips_as_none():
    frame = pd.DataFrame(
        {
            "reward": [1.0, 0.0],
            "propensity": [0.5, 0.5],
            "target_prob": [0.0, 0.0],
        }
    )

    report = evaluate_logged_policy(
        frame,
        reward_col="reward",
        propensity_col="propensity",
        target_probability_col="target_prob",
    )

    assert math.isnan(report.snips)
    assert report.to_dict()["snips"] is None
