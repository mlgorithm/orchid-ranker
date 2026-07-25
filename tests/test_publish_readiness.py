"""Publish-readiness checks for the single adaptive-recommender surface."""
from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pandas as pd
import pytest

import orchid_ranker
from orchid_ranker import AdaptiveRanker


def test_public_surface_is_one_adaptive_ranker() -> None:
    assert orchid_ranker.__all__ == ["AdaptiveRanker"]
    assert AdaptiveRanker is orchid_ranker.AdaptiveRanker


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
        "orchid_ranker.agents",
        "orchid_ranker.connectors",
        "orchid_ranker.streaming",
        "orchid_ranker.security",
        "orchid_ranker.cli",
        "orchid_ranker.curriculum",
        "orchid_ranker.pykt_bridge",
    ],
)
def test_removed_generic_modules_do_not_import(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_version_matches_package_metadata() -> None:
    with Path("pyproject.toml").open("rb") as fh:
        metadata = tomllib.load(fh)
    assert orchid_ranker.__version__ == metadata["project"]["version"]


def test_torch_core_dependency_is_available() -> None:
    import torch

    assert torch.__version__


def test_adaptive_ranker_smoke() -> None:
    pytest.importorskip("torch")

    events = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [101, 201, 101, 202, 101, 201],
            "outcome": [1, 0, 1, 1, 0, 1],
            "timestamp": [1, 2, 1, 2, 1, 2],
        }
    )

    ranker = AdaptiveRanker(
        kt_backbone="sakt",
        epochs=1,
        d_model=8,
        n_heads=2,
        batch_size=4,
        device="cpu",
    ).fit(events)

    ranked = ranker.recommend(user_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
    assert ranked
    ranker.observe(
        user_id=1,
        item_id=ranked[0].item_id,
        outcome=1,
        timestamp=3,
    )
