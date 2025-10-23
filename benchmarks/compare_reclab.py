"""Compare Orchid ALS with ReCLaB's TopPop baseline on a synthetic environment.

Usage
-----
    python benchmarks/compare_reclab.py --env topics-static-v1-small

Requires `reclab` to be installed. The script resets the chosen environment,
converts its ratings dictionary into train/test splits, and evaluates RMSE for
Orchid and the TopPop recommender.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from orchid_ranker import OrchidRecommender

try:
    from reclab import make as make_env
    from reclab.recommenders.top_pop import TopPop
except ImportError:  # pragma: no cover - optional dependency
    make_env = None
    TopPop = None


def _frame_from_ratings(ratings: Dict[Tuple[int, int], Tuple[float, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for (user_id, item_id), (rating, _context) in ratings.items():
        rows.append((int(user_id), int(item_id), float(rating)))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "label"])
    return df


def _split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_ratio
    test = df[mask].reset_index(drop=True)
    train = df[~mask].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Train/test split produced empty partition; adjust test_ratio or seed.")
    return train, test


def _to_reclab_dict(df: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[float, np.ndarray]]:
    payload: Dict[Tuple[int, int], Tuple[float, np.ndarray]] = {}
    for row in df.itertuples(index=False):
        payload[(int(row.user_id), int(row.item_id))] = (float(row.label), np.zeros(0, dtype=float))
    return payload


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Orchid Ranker with ReCLaB baselines.")
    parser.add_argument("--env", default="topics-static-v1-small", help="ReCLaB environment name")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of ratings for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args(argv)


def evaluate(
    orchid: OrchidRecommender,
    top_pop: Optional[TopPop],
    test_df: pd.DataFrame,
    rating_scale: float,
) -> Dict[str, float]:
    actual = test_df["label"].to_numpy(dtype=float)

    # Orchid predictions
    orchid_preds = []
    for row in test_df.itertuples(index=False):
        try:
            orchid_preds.append(orchid.predict(int(row.user_id), int(row.item_id)) * rating_scale)
        except KeyError:
            orchid_preds.append(np.nan)
    metrics = {"orchid_rmse": _rmse(actual, np.array(orchid_preds, dtype=float))}

    if top_pop is not None:
        requests = [(int(r.user_id), int(r.item_id), np.zeros(0, dtype=float)) for r in test_df.itertuples(index=False)]
        preds = top_pop.predict(requests)
        metrics["top_pop_rmse"] = _rmse(actual, preds.astype(float))
    else:
        metrics["top_pop_rmse"] = float("nan")

    return metrics


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(predicted)
    if not np.any(mask):
        return float("nan")
    diff = actual[mask] - predicted[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if make_env is None or TopPop is None:
        print("reclab is not installed. Install with `pip install reclab` to enable this comparison.")
        return 0

    env = make_env(args.env)
    env.seed(args.seed)
    env.reset()

    ratings_dict = env.ratings
    frame = _frame_from_ratings(ratings_dict)
    train_df, test_df = _split(frame, test_ratio=args.test_ratio, seed=args.seed)

    rating_scale = float(train_df["label"].max()) or 1.0
    train_scaled = train_df.copy()
    train_scaled["label"] = train_scaled["label"] / rating_scale

    orchid = OrchidRecommender(strategy="als", epochs=8)
    orchid.fit(train_scaled, rating_col="label")

    top_pop = TopPop()
    top_pop.reset(users=env.users, items=env.items, ratings=_to_reclab_dict(train_df))

    metrics = evaluate(orchid, top_pop, test_df, rating_scale)
    for name, value in metrics.items():
        print(f"{name:>12s}: {value:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
