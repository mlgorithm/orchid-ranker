"""Benchmark Orchid ALS against the implicit library on a tabular dataset.

Usage
-----
    python benchmarks/compare_implicit.py \
        --train data/train.csv \
        --test data/test.csv \
        --rating-col label

Both libraries operate on the same user/item id space. Implicit prefers
implicit-feedback data, but for convenience we treat the provided ratings as
confidence scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from orchid_ranker import OrchidRecommender

try:
    import implicit
except ImportError:  # pragma: no cover - optional dependency
    implicit = None


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"user_id", "item_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    return frame


def _prepare_mappings(df: pd.DataFrame) -> Dict[str, Dict[int, int]]:
    users = sorted(df.user_id.unique())
    items = sorted(df.item_id.unique())
    u2i = {int(u): idx for idx, u in enumerate(users)}
    i2i = {int(i): idx for idx, i in enumerate(items)}
    return {"user": u2i, "item": i2i}


def _to_sparse(df: pd.DataFrame, rating_col: str, maps: Dict[str, Dict[int, int]]) -> sparse.coo_matrix:
    rows = df.user_id.map(maps["user"]).to_numpy()
    cols = df.item_id.map(maps["item"]).to_numpy()
    data = df[rating_col].astype(float).to_numpy()
    shape = (len(maps["user"]), len(maps["item"]))
    return sparse.coo_matrix((data, (rows, cols)), shape=shape)


def _train_implicit(train: sparse.coo_matrix) -> Optional[implicit.als.AlternatingLeastSquares]:
    if implicit is None:
        return None
    model = implicit.als.AlternatingLeastSquares(factors=32, iterations=15, regularization=0.1)
    # implicit expects item-user matrix
    model.fit(train.transpose().tocsr())
    return model


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    if actual.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def evaluate(
    orchid: OrchidRecommender,
    imp: Optional[implicit.als.AlternatingLeastSquares],
    test_df: pd.DataFrame,
    rating_col: str,
    maps: Dict[str, Dict[int, int]],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    # Orchid predictions
    preds = []
    for row in test_df.itertuples(index=False):
        try:
            preds.append(orchid.predict(int(row.user_id), int(row.item_id)))
        except KeyError:
            preds.append(np.nan)
    metrics["orchid_rmse"] = _rmse(test_df[rating_col].to_numpy(dtype=float), np.array(preds, dtype=float))

    if imp is None:
        metrics["implicit_rmse"] = float("nan")
        return metrics

    inv_user = maps["user"]
    inv_item = maps["item"]

    rev_user = {v: k for k, v in inv_user.items()}
    rev_item = {v: k for k, v in inv_item.items()}

    imp_preds = []
    for row in test_df.itertuples(index=False):
        uid = inv_user.get(int(row.user_id))
        iid = inv_item.get(int(row.item_id))
        if uid is None or iid is None:
            imp_preds.append(np.nan)
            continue
        score = imp.user_factors[uid] @ imp.item_factors[iid]
        imp_preds.append(score)
    metrics["implicit_rmse"] = _rmse(test_df[rating_col].to_numpy(dtype=float), np.array(imp_preds, dtype=float))
    return metrics


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Orchid ALS with the implicit library.")
    parser.add_argument("--train", required=True, type=Path, help="Training CSV path")
    parser.add_argument("--test", required=True, type=Path, help="Test CSV path")
    parser.add_argument("--rating-col", default="label", help="Column containing feedback scores")
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    train_df = _load_frame(args.train)
    test_df = _load_frame(args.test)

    rating_col = args.rating_col
    if rating_col not in train_df.columns:
        raise ValueError(f"{rating_col!r} not present in training data")
    if rating_col not in test_df.columns:
        raise ValueError(f"{rating_col!r} not present in test data")

    maps = _prepare_mappings(pd.concat([train_df, test_df], ignore_index=True))

    orchid = OrchidRecommender(strategy="als", epochs=8)
    orchid.fit(train_df, rating_col=rating_col)

    train_sparse = _to_sparse(train_df, rating_col, maps)
    imp_model = _train_implicit(train_sparse)
    if imp_model is None:
        print("implicit is not installed. Install with `pip install implicit` to enable comparison.")

    metrics = evaluate(orchid, imp_model, test_df, rating_col, maps)
    for name, value in metrics.items():
        print(f"{name:>15s}: {value:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
