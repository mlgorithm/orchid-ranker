"""Tests for the adaptive-learning data readiness and pilot fallback path."""
from __future__ import annotations

import pandas as pd
import pytest

from orchid_ranker import AdaptiveRanker
from orchid_ranker.adaptive_learning import AdaptiveLearningRecommender
from orchid_ranker.empirical import EmpiricalTracer


def _pilot_events() -> pd.DataFrame:
    rows = []
    for user_id in range(4):
        for timestamp, (item_id, skill_id, outcome) in enumerate(
            ((101, "basics", 1), (102, "basics", 0), (201, "advanced", user_id % 2))
        ):
            rows.append(
                {
                    "user_id": f"learner-{user_id}",
                    "item_id": item_id,
                    "skill_id": skill_id,
                    "outcome": outcome,
                    "timestamp": timestamp,
                }
            )
    return pd.DataFrame(rows)


def test_sparse_pilot_defaults_to_the_empirical_adaptive_baseline() -> None:
    ranker = AdaptiveRanker().fit(_pilot_events(), category_col="skill_id")

    readiness = ranker.learning_readiness()
    recommendations = ranker.recommend("learner-0", [101, 102, 201], top_k=3)

    assert readiness["knowledge_tracing_ready"] is False
    assert readiness["active_tracer"] == "empirical"
    assert readiness["reasons"]
    assert readiness["recommendations"]
    assert len(recommendations) == 3
    assert all(0.0 <= rec.outcome_probability <= 1.0 for rec in recommendations)
    assert ranker.diagnostics()["active_tracer"] == "empirical"


def test_empirical_tracer_adapts_immediately_to_a_completed_attempt() -> None:
    interactions = _pilot_events().rename(columns={"outcome": "correct"})
    tracer = EmpiricalTracer().fit(interactions, timestamp_col="timestamp")

    before = tracer.predict_correct("new-learner", 101)
    tracer.observe("new-learner", 101, 1)
    after = tracer.predict_correct("new-learner", 101)

    assert after > before
    assert tracer.predict_many("new-learner", [101, 102])[101] == after


def test_readiness_can_confirm_kt_support_without_training_a_neural_model() -> None:
    rows = []
    for user_id in range(50):
        for timestamp in range(10):
            rows.append(
                {
                    "user_id": f"learner-{user_id}",
                    "item_id": timestamp % 10,
                    "correct": int((user_id + timestamp) % 3 != 0),
                    "timestamp": timestamp,
                }
            )
    rec = AdaptiveLearningRecommender(tracer_model="empirical").fit(pd.DataFrame(rows), timestamp_col="timestamp")

    report = rec.readiness_report()

    assert report.knowledge_tracing_ready is True
    assert report.active_tracer == "empirical"
    assert report.outcome_entropy > 0.0


def test_catalog_metadata_drives_history_and_registers_new_exercises() -> None:
    catalog = pd.DataFrame(
        {
            "item_id": [101, 102, 201, 301],
            "category_id": ["basics", "basics", "advanced", "advanced"],
            "difficulty": [0.2, 0.3, 0.7, 0.8],
        }
    )
    ranker = AdaptiveRanker().fit(
        _pilot_events().drop(columns=["skill_id"]),
        catalog=catalog,
    )

    recommendations = ranker.recommend("learner-0", [101, 301], top_k=2)

    new_item = next(rec for rec in recommendations if rec.item_id == 301)
    assert new_item.category_id == "advanced"
    assert new_item.difficulty == 0.8
    assert new_item.feedback_supported is True
    assert ranker.diagnostics()["n_concepts"] == 2


def test_catalog_rejects_missing_or_conflicting_historical_metadata() -> None:
    incomplete = pd.DataFrame(
        {
            "item_id": [101, 102],
            "category_id": ["basics", "basics"],
            "difficulty": [0.2, 0.3],
        }
    )
    duplicate = pd.DataFrame(
        {
            "item_id": [101, 101, 102, 201],
            "category_id": ["basics", "basics", "basics", "advanced"],
            "difficulty": [0.2, 0.2, 0.3, 0.7],
        }
    )

    with pytest.raises(ValueError, match="missing historical"):
        AdaptiveRanker().fit(_pilot_events(), catalog=incomplete)
    with pytest.raises(ValueError, match="one canonical row"):
        AdaptiveRanker().fit(_pilot_events(), catalog=duplicate)
