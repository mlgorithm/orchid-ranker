#!/usr/bin/env python3
"""Evaluate a CQL overlay on a future holdout before Orchid can serve it.

Run with:
    python examples/offline_policy_promotion.py

The data below is simulated only to show the contract. Replace both log frames
with completed, append-only decisions from one real deployment. A failed gate
is an expected safe outcome, not an error to override.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchid_ranker import AdaptiveRanker  # noqa: E402, I001


ITEMS = ["welcome", "profile", "connect", "invite"]
USERS = [f"user-{index:02d}" for index in range(40)]


def historical_outcomes() -> pd.DataFrame:
    """Create chronological outcomes used only to fit this small demo ranker."""
    rows: list[dict[str, object]] = []
    for user_index, user_id in enumerate(USERS):
        for step, item_id in enumerate(ITEMS):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "outcome": int((user_index + step) % 3 != 0),
                    "timestamp": user_index * 10 + step,
                }
            )
    return pd.DataFrame(rows)


def completed_decisions(ranker: AdaptiveRanker, *, start_timestamp: int, count: int) -> pd.DataFrame:
    """Generate completed, propensity-carrying decisions for the example only."""
    for offset in range(count):
        user_id = USERS[offset % len(USERS)]
        ranked, decision = ranker.recommend_and_log(
            user_id=user_id,
            candidate_item_ids=ITEMS,
            timestamp=start_timestamp + offset,
            top_k=1,
            exploration=0.25,
            min_item_support=0,
            min_outcome_probability=0.0,
            max_outcome_probability=1.0,
            require_prerequisites=False,
        )
        outcome = int(ranked[0].item_id in {"welcome", "profile"})
        ranker.observe_decision(
            decision.decision_id,
            outcome=outcome,
            timestamp=start_timestamp + count + offset,
        )
    return ranker.decision_log_frame(completed_only=True)


def main() -> None:
    ranker = AdaptiveRanker(epochs=1, d_model=8, n_heads=2, batch_size=32, device="cpu").fit(
        historical_outcomes()
    )
    first_window = completed_decisions(ranker, start_timestamp=1_000, count=60)
    all_decisions = completed_decisions(ranker, start_timestamp=2_000, count=60)
    future_window = all_decisions.iloc[len(first_window) :].reset_index(drop=True)

    ranker.fit_policy(
        first_window,
        evaluation_decisions=future_window,
    )
    assert ranker.last_policy_gate_ is not None
    gate = ranker.last_policy_gate_

    print(f"Evaluation events: {gate.n_events}")
    print(f"Estimated uplift: {gate.effect:.3f}")
    print(f"95% lower bound: {gate.ci_low:.3f}")
    print(f"Promoted: {gate.allowed}")
    if not gate.allowed:
        print("The base adaptive policy remains active:")
        for reason in gate.reasons:
            print(f"- {reason}")


if __name__ == "__main__":
    main()
