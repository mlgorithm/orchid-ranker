"""Quick performance profiling harness for Orchid strategies.

This utility generates a synthetic implicit-feedback dataset, fits selected
strategies using `OrchidRecommender`, and records fit plus recommend
latencies. The goal is to provide a reproducible baseline that can be plugged
into CI or run manually before releases to catch performance regressions.

Usage
-----
    python benchmarks/profile_strategies.py --strategies als,neural_mf \\
        --users 500 --items 800 --interactions 20000
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from orchid_ranker import OrchidRecommender, SUPPORTED_STRATEGIES


@dataclass
class ProfileResult:
    strategy: str
    fit_seconds: float
    recommend_seconds: float
    notes: str = ""


def _synthetic_interactions(num_users: int, num_items: int, num_interactions: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    users = rng.integers(0, num_users, size=num_interactions, endpoint=False)
    items = rng.integers(0, num_items, size=num_interactions, endpoint=False)
    labels = rng.integers(0, 2, size=num_interactions)
    return pd.DataFrame({"user_id": users, "item_id": items, "label": labels}, dtype=int)


def _item_features(num_items: int, dims: int) -> np.ndarray:
    rng = np.random.default_rng(seed=11)
    return rng.random((num_items, dims), dtype=np.float32)


def _profile_strategy(
    name: str,
    data: pd.DataFrame,
    top_k: int,
    item_feats: np.ndarray | None,
    max_users_for_recommend: int = 50,
) -> ProfileResult:
    start = time.perf_counter()
    kwargs: Dict[str, object] = {}
    try:
        if name == "linucb":
            if item_feats is None:
                raise RuntimeError("linucb requires item features; supply via --feature-dim")
            kwargs["item_features"] = item_feats
        rec = OrchidRecommender(strategy=name)
        rec.fit(data, rating_col="label", item_features=kwargs.get("item_features"))
        fit_elapsed = time.perf_counter() - start

        recommend_start = time.perf_counter()
        for user in data["user_id"].unique()[:max_users_for_recommend]:
            rec.recommend(int(user), top_k=top_k)
        recommend_elapsed = time.perf_counter() - recommend_start
        return ProfileResult(name, fit_elapsed, recommend_elapsed)
    except Exception as exc:  # pragma: no cover - informative path
        return ProfileResult(name, float("nan"), float("nan"), notes=f"failed: {exc}")


def _parse_strategies(spec: str | None) -> List[str]:
    if spec:
        requested = [s.strip().lower() for s in spec.split(",") if s.strip()]
        unknown = [s for s in requested if s not in SUPPORTED_STRATEGIES]
        if unknown:
            raise SystemExit(f"Unknown strategies requested: {unknown}. Supported: {SUPPORTED_STRATEGIES}")
        return requested
    return list(SUPPORTED_STRATEGIES)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile Orchid recommender strategies")
    parser.add_argument("--strategies", help="Comma-separated list of strategies; defaults to all supported.")
    parser.add_argument("--users", type=int, default=400, help="Number of synthetic users to simulate")
    parser.add_argument("--items", type=int, default=600, help="Number of synthetic items to simulate")
    parser.add_argument("--interactions", type=int, default=20000, help="Number of implicit interactions")
    parser.add_argument("--feature-dim", type=int, default=8, help="Item feature dimension for LinUCB")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to generate per user")
    parser.add_argument("--output", type=str, help="Optional JSON file for structured results")
    parser.add_argument(
        "--recommend-users",
        type=int,
        default=50,
        help="Number of users sampled for recommendation timing (defaults to 50).",
    )
    args = parser.parse_args(argv)

    strategies = _parse_strategies(args.strategies)
    data = _synthetic_interactions(args.users, args.items, args.interactions)
    features = _item_features(args.items, args.feature_dim) if "linucb" in strategies else None

    results: List[ProfileResult] = []
    for name in strategies:
        result = _profile_strategy(name, data, args.top_k, features, max_users_for_recommend=args.recommend_users)
        results.append(result)
        fit = f"{result.fit_seconds:.3f}s" if np.isfinite(result.fit_seconds) else "n/a"
        rec = f"{result.recommend_seconds:.3f}s" if np.isfinite(result.recommend_seconds) else "n/a"
        suffix = f" ({result.notes})" if result.notes else ""
        print(f"{name:>12s} | fit={fit} | recommend={rec}{suffix}")

    if args.output:
        payload = [asdict(result) for result in results]
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
