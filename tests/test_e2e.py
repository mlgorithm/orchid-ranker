"""End-to-end adaptive-learning workflows."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


def _events() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    rows = []
    catalog = [
        (101, "number-sense", 0.20),
        (102, "number-sense", 0.25),
        (201, "fractions", 0.45),
        (202, "fractions", 0.52),
        (301, "ratios", 0.65),
    ]
    for learner in range(8):
        ability = 0.35 + learner * 0.07
        for item_id, concept, difficulty in catalog:
            p_correct = float(np.clip(0.65 + ability - difficulty, 0.05, 0.95))
            rows.append(
                {
                    "learner_id": f"learner-{learner}",
                    "item_id": item_id,
                    "correct": int(rng.binomial(1, p_correct)),
                    "ts": len(rows),
                    "concept_id": concept,
                    "difficulty": difficulty,
                    "item_text": f"{concept} exercise {item_id}",
                }
            )
    return pd.DataFrame(rows)


def test_adaptive_ranker_fit_recommend_observe_and_diagnostics() -> None:
    pytest.importorskip("torch")
    from orchid_ranker import AdaptiveRanker

    events = _events()
    catalog = events.drop_duplicates("item_id")[["item_id", "concept_id", "difficulty", "item_text"]]
    ranker = AdaptiveRanker(
        kt_backbone="akt",
        policy="auto",
        epochs=1,
        d_model=8,
        n_heads=2,
        batch_size=8,
        device="cpu",
    ).fit_kt(
        events,
        learner_col="learner_id",
        item_col="item_id",
        correct_col="correct",
        timestamp_col="ts",
        concept_col="concept_id",
        item_difficulty_col="difficulty",
        prerequisite_by_concept={"fractions": ["number-sense"], "ratios": ["fractions"]},
    )
    ranker.fit_semantic_items(catalog, text_col="item_text", metadata_cols=["concept_id", "difficulty"])

    ranked = ranker.recommend("learner-1", [101, 201, 202, 301], top_k=3)
    assert ranked
    observed_item = catalog.set_index("item_id").loc[ranked[0].item_id]
    ranker.observe(
        learner_id="learner-1",
        ts=99,
        item_id=ranked[0].item_id,
        concept_id=observed_item["concept_id"],
        correct=1,
    )

    diagnostics = ranker.diagnostics()
    assert diagnostics["policy"] == "progression"
    assert diagnostics["adaptive_ranker"]["kt_backbone"] == "akt"


def test_knowledge_tracing_to_progression_metrics_pipeline() -> None:
    from orchid_ranker import (
        BayesianKnowledgeTracing,
        ForgettingCurve,
        ProficiencyTracker,
        category_coverage,
        progression_gain,
        stretch_fit,
    )

    tracker = ProficiencyTracker(skills=["number-sense", "fractions", "ratios"])
    for skill, correct in [
        ("number-sense", True),
        ("number-sense", True),
        ("fractions", False),
        ("fractions", True),
        ("ratios", False),
    ]:
        tracker.update(skill, correct=correct)

    mastered = tracker.succeeded()
    assert isinstance(mastered, (set, list, frozenset))
    assert 0.0 <= category_coverage(set(mastered), {"number-sense", "fractions", "ratios"}) <= 1.0
    assert progression_gain(pre_score=0.3, post_score=0.7) > 0.0
    assert 0.0 <= stretch_fit([0.45, 0.55], 0.50, 0.20) <= 1.0

    bkt = BayesianKnowledgeTracing(p_init=0.05, p_transit=0.05, p_slip=0.15, p_guess=0.25)
    for correct in [True, False, True, True, False, True] * 5:
        bkt.update(correct=correct)
    assert 0.0 <= bkt.p_known <= 1.0
    assert not math.isnan(bkt.p_known)

    fc = ForgettingCurve()
    fc.review()
    assert 0.0 <= fc.retention_at(3600.0) <= 1.0


def test_curriculum_pipeline() -> None:
    from orchid_ranker import DependencyGraph, ProgressionRecommender

    graph = DependencyGraph()
    graph.add_edge("number-sense", "fractions")
    graph.add_edge("fractions", "ratios")

    recommender = ProgressionRecommender(
        graph=graph,
        difficulty_map={"number-sense": 0.2, "fractions": 0.45, "ratios": 0.65},
    )
    assert recommender.recommend(set(), n=3) == ["number-sense"]
    assert "fractions" in recommender.recommend({"number-sense"}, n=3)

    restored = DependencyGraph.from_dict(graph.to_dict())
    assert restored.topological_order() == graph.topological_order()
