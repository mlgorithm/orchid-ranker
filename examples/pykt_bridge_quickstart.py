"""Export Orchid interactions to pyKT format and reuse pyKT predictions.

Run with:
    PYTHONPATH=src python examples/pykt_bridge_quickstart.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from orchid_ranker.learning_policy import KTValuePolicy
from orchid_ranker.pykt_bridge import PyKTPredictionAdapter, export_pykt_sequences, load_pykt_sequences


def main() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "item_id": ["q1", "q2", "q3", "q1", "q2", "q3"],
            "concept_id": ["fractions", "fractions", "ratios", "fractions", "fractions", "ratios"],
            "correct": [1, 0, 1, 0, 1, 1],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "duration": [12, 18, 20, 15, 16, 17],
        }
    )
    output = Path("/tmp/orchid_pykt_sequences.txt")
    export_pykt_sequences(
        interactions,
        output,
        concept_col="concept_id",
        timestamp_col="timestamp",
        duration_col="duration",
    )
    sequences = load_pykt_sequences(output)

    predictions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["q1", "q2", "q3"],
            "p_correct": [0.92, 0.62, 0.72],
        }
    )
    adapter = PyKTPredictionAdapter(predictions, fallback="global_mean")
    policy = KTValuePolicy(adapter, target_correct=0.70)
    recs = policy.rank("u1", ["q1", "q2", "q3"], top_k=2)

    print(f"Exported {len(sequences)} pyKT learner sequences")
    print(f"Top item from pyKT predictions: {recs[0].item_id}")
    print("pyKT bridge quickstart complete")


if __name__ == "__main__":
    main()
