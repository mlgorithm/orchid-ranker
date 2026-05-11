#!/usr/bin/env python3
"""Scenario-selection quickstart.

Run with: python examples/scenario_selection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchid_ranker import available_scenarios, recommend_scenarios


def main() -> None:
    matches = recommend_scenarios(
        has_outcomes=True,
        has_concepts=True,
        has_difficulty=True,
        has_prerequisites=True,
        needs_live_adaptation=True,
        use_case="adaptive learning recommender for math practice",
    )

    print("Available scenario count:", len(available_scenarios()))
    print("Top scenario:", matches[0].scenario.id)
    print("Why:")
    for reason in matches[0].reasons[:3]:
        print("-", reason)
    print("Entrypoints:", ", ".join(matches[0].scenario.entrypoints))
    print("Scenario selection quickstart complete.")


if __name__ == "__main__":
    main()
