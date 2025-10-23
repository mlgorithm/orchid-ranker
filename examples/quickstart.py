"""Minimal example showing how engineers can plug OrchidRecommender into a pipeline."""

import pandas as pd

from orchid_ranker import OrchidRecommender


def main() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 3],
            "item_id": [10, 12, 10, 11, 12, 13, 14],
            "label": [1, 1, 1, 0, 1, 0, 1],
        }
    )

    recommender = OrchidRecommender(strategy="als", epochs=3)
    recommender.fit(interactions, rating_col="label")
    recs = recommender.recommend(user_id=1, top_k=5)
    for rec in recs:
        print(f"item={rec.item_id} score={rec.score:.4f}")


if __name__ == "__main__":
    main()
