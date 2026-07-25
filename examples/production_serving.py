#!/usr/bin/env python3
"""Production-style serving loop using the single public API.

Run with: python examples/production_serving.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchid_ranker import AdaptiveRanker  # noqa: E402


def build_events() -> pd.DataFrame:
    catalog = [
        (101, "number-sense", "seq-basics", 0.20),
        (102, "number-sense", "seq-basics", 0.25),
        (201, "fractions", "seq-fractions", 0.42),
        (202, "fractions", "seq-fractions", 0.48),
        (203, "fractions", "seq-fractions", 0.55),
        (301, "ratios", "seq-ratios", 0.64),
    ]
    rows = []
    for learner_idx, ability in enumerate([0.35, 0.45, 0.55, 0.70, 0.82]):
        for step, (item_id, concept_id, sequence_id, difficulty) in enumerate(catalog):
            rows.append(
                {
                    "user_id": f"user-{learner_idx}",
                    "item_id": item_id,
                    "category_id": concept_id,
                    "sequence_id": sequence_id,
                    "difficulty": difficulty,
                    "outcome": int((0.65 + ability - difficulty) >= 0.55),
                    "timestamp": learner_idx * 100 + step,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    events = build_events()
    ranker = AdaptiveRanker().fit(
        events,
        category_col="category_id",
        difficulty_col="difficulty",
    )

    user_id = "user-1"
    sequence_id = "seq-fractions"
    candidate_items = events.loc[events["sequence_id"] == sequence_id, "item_id"].drop_duplicates().tolist()
    ranked, decision = ranker.recommend_and_log(
        user_id=user_id,
        candidate_item_ids=candidate_items,
        timestamp=1_234,
        top_k=len(candidate_items),
        exploration=0.10,
        min_item_support=3,
    )
    chosen = ranked[0]
    ranker.observe_decision(
        decision.decision_id,
        outcome=1,
        timestamp=1_235,
    )
    decision_log = ranker.decision_log_frame(completed_only=True)
    shadow = ranker.shadow_report(cluster_bootstrap_samples=0)
    reranked = ranker.recommend(user_id, candidate_items, top_k=3)

    print(f"Served item: {chosen.item_id}")
    print(f"Chosen propensity: {decision.propensity:.3f}")
    print(f"Logged decisions: {len(decision_log)}")
    print(f"Outcome coverage: {shadow.outcome_coverage:.1%}")
    print(f"After observe: {[rec.item_id for rec in reranked]}")


if __name__ == "__main__":
    main()
