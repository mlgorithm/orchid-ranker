from __future__ import annotations

import subprocess
import sys

import pytest

from orchid_ranker import available_scenarios, recommend_scenarios


def test_available_scenarios_start_with_adaptive_learning_recipe():
    scenarios = available_scenarios()

    assert scenarios[0].id == "adaptive_learning_next_item"
    assert "AdaptiveLearningRecommender.rank" in scenarios[0].entrypoints
    assert "AKT/SAKT knowledge tracing" in scenarios[0].algorithms


def test_recommend_scenarios_selects_adaptive_learning_for_live_outcomes():
    matches = recommend_scenarios(
        has_outcomes=True,
        has_concepts=True,
        has_difficulty=True,
        has_prerequisites=True,
        needs_live_adaptation=True,
        use_case="adaptive learning practice",
    )

    assert matches[0].scenario.id == "adaptive_learning_next_item"
    assert matches[0].score > matches[1].score
    assert any("learner state" in reason for reason in matches[0].reasons)


def test_recommend_scenarios_selects_safe_rollout_for_guardrails():
    matches = recommend_scenarios(
        has_outcomes=True,
        needs_live_adaptation=True,
        needs_safe_rollout=True,
        use_case="guardrail rollout with fallback and OPE",
    )

    assert matches[0].scenario.id == "safe_adaptive_rollout"
    assert "evaluate_logged_policy" in matches[0].scenario.entrypoints


def test_recommend_scenarios_selects_generic_streaming_without_learning_metadata():
    matches = recommend_scenarios(
        has_interactions=True,
        needs_live_adaptation=True,
        use_case="streaming online recommender",
    )

    assert matches[0].scenario.id == "generic_streaming_recommender"
    assert any("learning metadata is missing" in reason for reason in matches[0].reasons)


def test_recommend_scenarios_validates_top_k():
    with pytest.raises(ValueError, match="top_k"):
        recommend_scenarios(top_k=0)


def test_scenario_selection_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/scenario_selection.py"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "Top scenario: adaptive_learning_next_item" in result.stdout
    assert "Scenario selection quickstart complete" in result.stdout
