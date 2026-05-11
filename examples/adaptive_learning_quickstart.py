#!/usr/bin/env python3
"""Adaptive-learning quickstart.

Run with: python examples/adaptive_learning_quickstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchid_ranker import AdaptiveLearningRecommender, DependencyGraph, ProgressionRecommender


def build_catalog() -> pd.DataFrame:
    """Small exercise catalog with prerequisite-aware concepts."""
    return pd.DataFrame(
        [
            {"item_id": 101, "concept": "number-sense", "difficulty": 0.20, "title": "Compare whole numbers"},
            {"item_id": 102, "concept": "number-sense", "difficulty": 0.25, "title": "Place value review"},
            {"item_id": 201, "concept": "fractions", "difficulty": 0.42, "title": "Equivalent fractions"},
            {"item_id": 202, "concept": "fractions", "difficulty": 0.48, "title": "Add unlike fractions"},
            {"item_id": 301, "concept": "ratios", "difficulty": 0.62, "title": "Unit rates"},
            {"item_id": 302, "concept": "ratios", "difficulty": 0.68, "title": "Scale drawings"},
            {"item_id": 401, "concept": "linear-equations", "difficulty": 0.78, "title": "One-step equations"},
            {"item_id": 402, "concept": "linear-equations", "difficulty": 0.84, "title": "Two-step equations"},
        ]
    )


def build_history(catalog: pd.DataFrame) -> pd.DataFrame:
    """Synthetic historical outcomes: user_id, item_id, correct."""
    import numpy as np

    rng = np.random.RandomState(7)
    learner_ability = {
        42: 0.52,
        1001: 0.35,
        1002: 0.45,
        1003: 0.58,
        1004: 0.72,
        1005: 0.82,
    }
    rows = []
    for user_id, ability in learner_ability.items():
        for row in catalog.itertuples(index=False):
            # More able learners are more likely to answer hard items correctly.
            p_correct = float(np.clip(0.65 + ability - row.difficulty, 0.05, 0.95))
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": int(row.item_id),
                    "correct": int(rng.binomial(1, p_correct)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    catalog = build_catalog()
    interactions = build_history(catalog)
    events = interactions.merge(catalog[["item_id", "concept", "difficulty"]], on="item_id")

    graph = DependencyGraph(
        [
            ("number-sense", "fractions"),
            ("fractions", "ratios"),
            ("ratios", "linear-equations"),
        ]
    )
    concept_difficulty = catalog.groupby("concept")["difficulty"].mean().to_dict()
    progression = ProgressionRecommender(graph, difficulty_map=concept_difficulty)

    learner_rec = AdaptiveLearningRecommender(
        tracer_model="akt",
        policy="auto",
        max_seq_len=4,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=8,
        device="cpu",
        random_state=7,
        reward_model_max_examples=100,
        reward_model_cross_fit_folds=1,
    ).fit(
        events,
        correct_col="correct",
        concept_col="concept",
        item_difficulty_col="difficulty",
        prerequisite_by_concept={
            "fractions": ["number-sense"],
            "ratios": ["fractions"],
            "linear-equations": ["ratios"],
        },
    )

    learner_id = 42
    completed_concepts = {"number-sense"}
    eligible_concepts = progression.recommend(completed_concepts, n=2)
    candidate_items = catalog[catalog["concept"].isin(eligible_concepts)]["item_id"].tolist()

    before = learner_rec.rank(learner_id, candidate_items, top_k=3)
    print(f"Eligible concepts: {eligible_concepts}")
    print(f"Resolved policy: {learner_rec.policy_name_}")
    print(f"Before live outcome: {before}")

    history_length = learner_rec.observe(user_id=learner_id, item_id=201, correct=False)
    after = learner_rec.rank(learner_id, candidate_items, top_k=3)

    print(f"Fractions competence after outcome: {learner_rec.competence_for(learner_id, 'fractions'):.3f}")
    print(f"Learner history length after outcome: {history_length}")
    print(f"After live outcome: {after}")
    print("Adaptive learning quickstart complete.")


if __name__ == "__main__":
    main()
