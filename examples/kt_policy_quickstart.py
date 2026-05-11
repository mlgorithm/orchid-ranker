#!/usr/bin/env python3
"""KT-guided next-item policy quickstart.

Run with: python examples/kt_policy_quickstart.py
"""
from __future__ import annotations

from knowledge_tracing_quickstart import build_events

from orchid_ranker.kt import SAKTTracer
from orchid_ranker.learning_policy import KTValuePolicy


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

    policy = KTValuePolicy(tracer, target_correct=0.70)
    learner_id = "live-learner"
    candidates = [201, 202, 301]

    print("Initial policy ranking:")
    for rec in policy.rank(learner_id, candidates, top_k=3):
        print(
            f"  item={rec.item_id} score={rec.score:.3f} "
            f"p_correct={rec.p_correct:.3f} stretch={rec.stretch_fit:.3f}"
        )

    policy.observe(learner_id, 201, correct=False)
    print("After observing an incorrect fractions answer:")
    for rec in policy.rank(learner_id, candidates, top_k=3):
        print(
            f"  item={rec.item_id} score={rec.score:.3f} "
            f"p_correct={rec.p_correct:.3f} stretch={rec.stretch_fit:.3f}"
        )

    print("KT policy quickstart complete.")


if __name__ == "__main__":
    main()
