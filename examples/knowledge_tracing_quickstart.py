#!/usr/bin/env python3
"""Knowledge tracing quickstart.

Run with: python examples/knowledge_tracing_quickstart.py
"""
from __future__ import annotations

import pandas as pd

from orchid_ranker.kt import SAKTTracer


def build_events() -> pd.DataFrame:
    rows = []
    catalog = [
        (101, "number-sense", 0.20),
        (102, "number-sense", 0.30),
        (201, "fractions", 0.45),
        (202, "fractions", 0.55),
        (301, "ratios", 0.70),
    ]
    abilities = {7: 0.48, 8: 0.35, 9: 0.62, 10: 0.80}
    for user_id, ability in abilities.items():
        for step, (item_id, concept, difficulty) in enumerate(catalog):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "concept": concept,
                    "correct": int(ability + 0.10 >= difficulty),
                    "timestamp": step,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    events = build_events()
    tracer = SAKTTracer(
        max_seq_len=4,
        d_model=16,
        n_heads=2,
        epochs=2,
        batch_size=4,
        random_state=42,
        device="cpu",
    ).fit(events, timestamp_col="timestamp")

    learner_id = "new-session"
    candidates = [201, 202, 301]

    before = tracer.recommend_practice(learner_id, candidates, top_k=3, target_correct=0.70)
    print("Before live outcome:")
    for rec in before:
        print(f"  item={rec.item_id} p_correct={rec.p_correct:.3f} stretch_score={rec.score:.3f}")

    tracer.observe(learner_id, 201, correct=False)
    after = tracer.recommend_practice(learner_id, candidates, top_k=3, target_correct=0.70)

    print("After incorrect fractions outcome:")
    for rec in after:
        print(f"  item={rec.item_id} p_correct={rec.p_correct:.3f} stretch_score={rec.score:.3f}")

    print("Knowledge tracing quickstart complete.")


if __name__ == "__main__":
    main()
