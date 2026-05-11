#!/usr/bin/env python3
"""AKT-inspired knowledge tracing quickstart.

Run with: python examples/akt_quickstart.py
"""
from __future__ import annotations

from knowledge_tracing_quickstart import build_events

from orchid_ranker.kt import AKTTracer
from orchid_ranker.learning_policy import KTValuePolicy


def main() -> None:
    events = build_events()
    difficulty = {
        101: 0.20,
        102: 0.30,
        201: 0.45,
        202: 0.55,
        301: 0.70,
    }
    events["difficulty"] = events["item_id"].map(difficulty)

    tracer = AKTTracer(
        max_seq_len=4,
        d_model=16,
        epochs=2,
        batch_size=4,
        random_state=42,
        device="cpu",
    ).fit(events, timestamp_col="timestamp", item_difficulty_col="difficulty")

    policy = KTValuePolicy(tracer, target_correct=0.70, difficulty_by_item=difficulty)
    learner_id = "akt-live-learner"
    candidates = [201, 202, 301]

    print("AKT-inspired policy ranking:")
    for rec in policy.rank(learner_id, candidates, top_k=3):
        print(
            f"  item={rec.item_id} difficulty={rec.difficulty:.2f} "
            f"p_correct={rec.p_correct:.3f} score={rec.score:.3f}"
        )

    policy.observe(learner_id, 201, correct=False)
    print("After observing an incorrect fractions answer:")
    for rec in policy.rank(learner_id, candidates, top_k=3):
        print(
            f"  item={rec.item_id} difficulty={rec.difficulty:.2f} "
            f"p_correct={rec.p_correct:.3f} score={rec.score:.3f}"
        )

    print("AKT quickstart complete.")


if __name__ == "__main__":
    main()
