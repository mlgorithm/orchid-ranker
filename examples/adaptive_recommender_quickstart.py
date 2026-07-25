#!/usr/bin/env python3
"""Smallest useful Orchid Ranker example.

Run with: python examples/adaptive_recommender_quickstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchid_ranker import AdaptiveRanker


def build_history() -> pd.DataFrame:
    """Historical outcomes. These four columns are enough to start."""
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "item_id": [101, 102, 201, 101, 102, 201, 101, 102, 201],
            "outcome": [1, 1, 0, 1, 0, 0, 1, 1, 1],
            "timestamp": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        }
    )


def main() -> None:
    ranker = AdaptiveRanker().fit(build_history())

    ranked = ranker.recommend("a", [101, 102, 201], top_k=2)
    print("Recommended items:", [rec.item_id for rec in ranked])

    ranker.observe(
        user_id="a",
        item_id=ranked[0].item_id,
        outcome=0,
        timestamp=4,
    )
    reranked = ranker.recommend("a", [101, 102, 201], top_k=2)
    print("After the outcome:", [rec.item_id for rec in reranked])
    print("Adaptive recommender quickstart complete.")


if __name__ == "__main__":
    main()
