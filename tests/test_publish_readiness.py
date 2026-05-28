"""Publish-readiness checks for the adaptive-learning public surface."""
from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from orchid_ranker import (
    AdaptiveLearningEngine,
    AdaptiveLearningRecommender,
    AdaptiveRanker,
    BayesianKnowledgeTracing,
    DependencyGraph,
    ProficiencyTracker,
    ProgressionRecommender,
    __version__,
    available_scenarios,
    recommend_scenarios,
)
from orchid_ranker._compat import require_torch, torch_available


def test_public_surface_is_adaptive_first() -> None:
    assert AdaptiveRanker is not None
    assert AdaptiveLearningEngine is AdaptiveLearningRecommender
    assert BayesianKnowledgeTracing().p_known >= 0.0
    assert ProgressionRecommender is not None
    assert ProficiencyTracker is not None
    assert DependencyGraph is not None


@pytest.mark.parametrize(
    "name",
    [
        "OrchidRecommender",
        "Recommendation",
        "SUPPORTED_STRATEGIES",
        "STRATEGY_GUIDE",
        "GridSearchCV",
        "RandomSearchCV",
        "save_model",
        "load_model",
    ],
)
def test_generic_recommender_names_are_not_package_root_exports(name: str) -> None:
    import orchid_ranker

    with pytest.raises(AttributeError):
        getattr(orchid_ranker, name)


@pytest.mark.parametrize(
    "module_name",
    [
        "orchid_ranker.recommender",
        "orchid_ranker.model_selection",
        "orchid_ranker.tuning",
        "orchid_ranker.serialization",
        "orchid_ranker.cold_start",
        "orchid_ranker.curated_feed",
        "orchid_ranker.taste_progression",
        "orchid_ranker.legacy",
    ],
)
def test_removed_generic_modules_do_not_import(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_version_matches_package_metadata() -> None:
    with Path("pyproject.toml").open("rb") as fh:
        metadata = tomllib.load(fh)
    assert __version__ == metadata["project"]["version"]


def test_torch_compat_contract() -> None:
    if torch_available():
        require_torch("adaptive-learning publish readiness")


def test_scenario_catalog_contains_only_adaptive_learning_paths() -> None:
    ids = {scenario.id for scenario in available_scenarios()}
    assert ids == {
        "adaptive_learning_next_item",
        "safe_adaptive_rollout",
        "regulated_training",
        "new_user_cold_start",
    }
    assert "generic_streaming_recommender" not in ids
    assert "batch_catalog_recommendation" not in ids


def test_adaptive_learning_engine_smoke() -> None:
    pytest.importorskip("torch")

    events = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [101, 201, 101, 202, 101, 201],
            "correct": [1, 0, 1, 1, 0, 1],
            "concept": ["number-sense", "fractions", "number-sense", "fractions", "number-sense", "fractions"],
            "difficulty": [0.20, 0.45, 0.20, 0.50, 0.20, 0.45],
        }
    )

    rec = AdaptiveLearningEngine(
        tracer_model="akt",
        policy="auto",
        epochs=1,
        d_model=8,
        n_heads=2,
        batch_size=4,
        device="cpu",
    ).fit(
        events,
        correct_col="correct",
        concept_col="concept",
        item_difficulty_col="difficulty",
        prerequisite_by_concept={"fractions": ["number-sense"]},
    )

    ranked = rec.rank(user_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
    assert ranked
    rec.observe(user_id=1, item_id=ranked[0].item_id, correct=True)


def test_scenario_selection_smoke() -> None:
    matches = recommend_scenarios(
        has_outcomes=True,
        has_concepts=True,
        has_difficulty=True,
        needs_live_adaptation=True,
        use_case="adaptive practice",
    )
    assert matches[0].scenario.id == "adaptive_learning_next_item"
