"""End-to-end checks for the one supported product loop."""
from __future__ import annotations

import numpy as np
import pandas as pd

from orchid_ranker import AdaptiveRanker


def _events() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for user_id, ability in enumerate((0.35, 0.5, 0.65, 0.8)):
        for timestamp, (item_id, category_id, difficulty) in enumerate(
            ((101, "start", 0.2), (102, "start", 0.3), (201, "advance", 0.6), (202, "advance", 0.7))
        ):
            rows.append(
                {
                    "user_id": f"user-{user_id}",
                    "item_id": item_id,
                    "category_id": category_id,
                    "difficulty": difficulty,
                    "outcome": int(rng.random() < 0.7 + ability - difficulty),
                    "timestamp": timestamp,
                }
            )
    return pd.DataFrame(rows)


def test_fit_serve_log_observe_and_report():
    ranker = AdaptiveRanker(epochs=1, d_model=8, n_heads=2, batch_size=4, device="cpu").fit(
        _events(),
        category_col="category_id",
        difficulty_col="difficulty",
    )

    ranked, decision = ranker.recommend_and_log(
        user_id="user-1",
        candidate_item_ids=[101, 102, 201, 202],
        timestamp=10,
        top_k=3,
    )
    ranker.observe_decision(decision.decision_id, outcome=1, timestamp=11)
    report = ranker.shadow_report(cluster_bootstrap_samples=0)

    assert ranked
    assert report.n_decisions == 1
    assert report.n_outcomes == 1
    assert report.unique_users == 1
    assert report.outcome_mean == 1.0
    assert ranker.diagnostics()["adaptive_ranker"]["linked_outcomes"] == 1
