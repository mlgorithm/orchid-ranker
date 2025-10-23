"""Command line evaluation harness for Orchid Ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from orchid_ranker import OrchidRecommender
from orchid_ranker.evaluation import RankingReport, evaluate_recommendations


def _parse_strategy(spec: str) -> tuple[str, Dict[str, object]]:
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    if not parts:
        raise ValueError(f"Invalid strategy specification: {spec}")
    name = parts[0]
    params: Dict[str, object] = {}
    for token in parts[1:]:
        if "=" not in token:
            raise ValueError(f"Malformed parameter token '{token}' in '{spec}'")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            params[key] = value.lower() == "true"
        else:
            try:
                params[key] = int(value)
            except ValueError:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    return name, params


def _recommend(rec: OrchidRecommender, users: Sequence[int], top_k: int) -> Dict[int, List[int]]:
    outputs: Dict[int, List[int]] = {}
    for uid in users:
        try:
            slate = rec.recommend(int(uid), top_k=top_k)
        except KeyError:
            continue
        outputs[int(uid)] = [item.item_id for item in slate]
    return outputs


def _build_relevance(df: pd.DataFrame, user_col: str, item_col: str, rating_col: str, threshold: float) -> Dict[int, List[int]]:
    rel: Dict[int, List[int]] = {}
    for uid, group in df.groupby(user_col):
        rel_items = group[group[rating_col] > threshold][item_col].astype(int).tolist()
        if rel_items:
            rel[int(uid)] = rel_items
    return rel


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Orchid Ranker strategies on a dataset.")
    parser.add_argument("--train", required=True, type=Path, help="Training CSV path")
    parser.add_argument("--test", required=True, type=Path, help="Test CSV path")
    parser.add_argument(
        "--strategy",
        action="append",
        required=True,
        help="Strategy specification, e.g. 'als,epochs=10' or 'linucb,alpha=1.5'. Repeat for multiple entries.",
    )
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K recommendations to evaluate.")
    parser.add_argument(
        "--implicit-threshold",
        type=float,
        default=0.0,
        help="Values greater than this threshold are treated as relevant for recall/precision.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON file to write aggregated metrics.")
    args = parser.parse_args(argv)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    results: Dict[str, Dict[str, float]] = {}
    users = sorted(test_df[args.user_col].unique())
    relevance = _build_relevance(test_df, args.user_col, args.item_col, args.label_col, args.implicit_threshold)

    for spec in args.strategy:
        name, params = _parse_strategy(spec)
        print(f"\n>>> Strategy: {name} ({params})")
        rec = OrchidRecommender(strategy=name, validate_inputs=True, **params)
        rec.fit(train_df, rating_col=args.label_col)
        recs = _recommend(rec, users, args.top_k)
        report: RankingReport = evaluate_recommendations(recs, relevance)
        metrics = {
            "precision_at_5": report.precision_at_5,
            "recall_at_5": report.recall_at_5,
            "map_at_10": report.map_at_10,
            "ndcg_at_10": report.ndcg_at_10,
        }
        results[name] = metrics
        print(json.dumps(metrics, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    return 0


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
