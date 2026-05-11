"""Rank learning items by progression reward, not only correctness.

Run with:
    PYTHONPATH=src python examples/progression_policy_quickstart.py
"""
from __future__ import annotations

from orchid_ranker.learning_policy import ProgressionValuePolicy


class DemoTracer:
    def predict_many(self, user_id, item_ids):
        del user_id
        values = {
            "easy-review": 0.96,
            "right-stretch": 0.72,
            "too-hard": 0.30,
        }
        return {item_id: values[item_id] for item_id in item_ids}

    def observe(self, user_id, item_id, correct):
        del user_id, item_id, correct
        return None


def main() -> None:
    policy = ProgressionValuePolicy(
        DemoTracer(),
        difficulty_by_item={
            "easy-review": 0.15,
            "right-stretch": 0.65,
            "too-hard": 0.90,
        },
        concept_by_item={
            "easy-review": "fractions",
            "right-stretch": "fractions",
            "too-hard": "ratios",
        },
    )
    ranked = policy.rank("learner-1", ["easy-review", "right-stretch", "too-hard"], top_k=3)
    for rec in ranked:
        print(
            f"{rec.item_id}: score={rec.score:.3f} "
            f"p_correct={rec.p_correct:.2f} difficulty={rec.difficulty:.2f}"
        )
    print("Progression policy quickstart complete")


if __name__ == "__main__":
    main()
