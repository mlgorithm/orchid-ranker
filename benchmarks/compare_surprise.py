"""Compare Orchid Ranker's ALS baseline with a Surprise model on tabular data.

The script expects pre-split CSV files containing at least ``user_id``,
``item_id`` and a rating column. Metrics are simple RMSE calculations on the
provided test split so engineers and researchers can run quick regression
checks.

Usage
-----
    python benchmarks/compare_surprise.py \
        --train data/train.csv \
        --test data/test.csv \
        --rating-col label

Surprise is optional; if it is missing the script prints an informative message
and exits with code 0 so it can be wired into CI without hard failing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from orchid_ranker import OrchidRecommender

try:
    from surprise import Dataset, Reader, SVD
except ImportError:  # pragma: no cover - optional dependency
    Dataset = None
    Reader = None
    SVD = None


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"user_id", "item_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    return frame


def _train_orchid(df: pd.DataFrame, rating_col: str) -> OrchidRecommender:
    model = OrchidRecommender(strategy="als", epochs=8)
    model.fit(df, rating_col=rating_col)
    return model


def _train_surprise(df: pd.DataFrame, rating_col: str):
    if Dataset is None:
        return None
    reader = Reader(rating_scale=(float(df[rating_col].min()), float(df[rating_col].max())))
    data = Dataset.load_from_df(df[["user_id", "item_id", rating_col]], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_epochs=10, biased=True)
    algo.fit(trainset)
    return algo


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def evaluate(models: dict[str, object], test_df: pd.DataFrame, rating_col: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, model in models.items():
        if model is None:
            metrics[name] = float("nan")
            continue
        preds = []
        for row in test_df.itertuples(index=False):
            try:
                if name == "orchid":
                    preds.append(model.predict(int(row.user_id), int(row.item_id)))
                else:
                    preds.append(model.predict(str(row.user_id), str(row.item_id)).est)
            except Exception:
                preds.append(np.nan)
        mask = ~np.isnan(preds)
        actual = test_df.loc[mask, rating_col].to_numpy()
        predicted = np.asarray(preds)[mask]
        if actual.size == 0:
            metrics[name] = float("nan")
        else:
            metrics[name] = _rmse(actual, predicted)
    return metrics


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Orchid Ranker against Surprise on held-out data.")
    parser.add_argument("--train", required=True, type=Path, help="Path to training CSV")
    parser.add_argument("--test", required=True, type=Path, help="Path to test CSV")
    parser.add_argument("--rating-col", default="label", help="Name of the rating/label column")
    parser.add_argument("--output", type=Path, help="Optional JSON file to write metrics")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    train_df = _load_frame(args.train)
    test_df = _load_frame(args.test)

    rating_col = args.rating_col
    if rating_col not in train_df.columns:
        raise ValueError(f"{rating_col!r} not found in training data")
    if rating_col not in test_df.columns:
        raise ValueError(f"{rating_col!r} not found in test data")

    orchid = _train_orchid(train_df, rating_col)
    surprise_model = _train_surprise(train_df, rating_col)
    if surprise_model is None:
        print("Surprise is not installed. Install with `pip install surprise` to enable competitor comparison.")

    metrics = evaluate({"orchid": orchid, "surprise": surprise_model}, test_df, rating_col)
    for name, value in metrics.items():
        print(f"{name:>8s} RMSE: {value:.4f}")

    if args.output:
        import json

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
