"""Minimal end-to-end example for Orchid Ranker."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from orchid_ranker import OrchidRecommender


def build_sample_data(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3],
        "item_id": [10, 11, 10, 13, 12, 14],
        "label":   [1, 0, 1, 1, 0, 1],
    })
    test = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [12, 13, 14],
        "label":   [1, 1, 0],
    })

    train_path = data_dir / "quickstart_train.csv"
    test_path = data_dir / "quickstart_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def main() -> None:
    train_path, test_path = build_sample_data(Path(__file__).parent / "data")

    interactions = pd.read_csv(train_path)
    rec = OrchidRecommender(strategy="als", epochs=3)
    rec.fit(interactions, rating_col="label")

    print("Top-3 for user 1:", rec.recommend(user_id=1, top_k=3))
    print("Use orchid-evaluate with the generated CSVs:")
    print(f"  orchid-evaluate --train {train_path} --test {test_path} --strategy 'als,epochs=3'")


if __name__ == "__main__":
    main()
