#!/usr/bin/env python3
"""Safety pattern: progression monitor + guardrail + reviewed fallback.

Run with: python examples/custom_guardrail.py
Requires: pip install orchid-ranker[adaptive]
"""
from __future__ import annotations

import pandas as pd

from orchid_ranker import AdaptiveLearningEngine
from orchid_ranker.live_metrics import (
    GuardrailConfig,
    ProgressionGuardrail,
    RollingProgressionMonitor,
)


def build_outcomes() -> pd.DataFrame:
    rows = []
    for learner in range(12):
        for item_id, concept, difficulty in [
            (101, "number-sense", 0.20),
            (201, "fractions", 0.45),
            (202, "fractions", 0.52),
            (301, "ratios", 0.65),
        ]:
            rows.append(
                {
                    "user_id": learner,
                    "item_id": item_id,
                    "correct": int((learner / 12) + 0.55 > difficulty),
                    "concept": concept,
                    "difficulty": difficulty,
                }
            )
    return pd.DataFrame(rows)


def reviewed_fallback(candidates: list[int], difficulty: dict[int, float], top_k: int) -> list[int]:
    """Simple reviewed policy: easiest eligible items first."""
    return sorted(candidates, key=lambda item_id: (difficulty[item_id], item_id))[:top_k]


def main() -> None:
    outcomes = build_outcomes()
    difficulty = outcomes.drop_duplicates("item_id").set_index("item_id")["difficulty"].to_dict()
    rec = AdaptiveLearningEngine(
        tracer_model="akt",
        policy="auto",
        epochs=1,
        d_model=16,
        n_heads=2,
        batch_size=8,
        device="cpu",
    ).fit(
        outcomes,
        correct_col="correct",
        concept_col="concept",
        item_difficulty_col="difficulty",
        prerequisite_by_concept={"fractions": ["number-sense"], "ratios": ["fractions"]},
    )

    monitor = RollingProgressionMonitor(
        window_size=500,
        total_categories={"number-sense", "fractions", "ratios"},
        emit_prometheus=False,
    )
    guardrail = ProgressionGuardrail(
        monitor,
        cfg=GuardrailConfig(
            min_progression_gain=0.0,
            min_accept_rate=0.3,
            warmup_samples=10,
            consecutive_violations=2,
        ),
    )

    for i in range(20):
        monitor.record(
            user_id=0,
            item_id=201 + (i % 2),
            correct=False,
            category="fractions",
            pre_competence=0.3,
            post_competence=0.28,
            difficulty=0.8,
        )

    candidates = [101, 201, 202, 301]
    allowed = guardrail.evaluate()
    print(f"Adaptive allowed: {allowed}")
    print(f"Halted: {guardrail.is_halted}  Reason: {guardrail.halt_reason}")

    if guardrail.should_allow_adaptive():
        top = rec.rank(user_id=0, candidate_item_ids=candidates, top_k=3)
        print("\nAdaptive policy is clear to serve:")
        for item in top:
            print(f"  item {item.item_id}: score {item.score:.4f} policy={item.policy}")
    else:
        top = reviewed_fallback(candidates, difficulty, top_k=3)
        print("\nGuardrail fired -- falling back to reviewed prerequisite/difficulty policy:")
        for item_id in top:
            print(f"  item {item_id}: difficulty {difficulty[item_id]:.2f}")


if __name__ == "__main__":
    main()
