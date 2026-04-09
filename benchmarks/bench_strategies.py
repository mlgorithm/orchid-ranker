"""Benchmark all strategies on the frozen fixture dataset.

Produces golden baseline metrics (P@10, R@10, NDCG@10, fit time, infer time)
for regression testing. Results are saved to benchmarks/golden/baselines.json.

Usage:
    python benchmarks/bench_strategies.py [--seeds 42,13,17] [--top-k 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orchid_ranker import OrchidRecommender
from orchid_ranker.evaluation import precision_at_k, recall_at_k, ndcg_at_k

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Strategies that don't need item features
BASIC_STRATEGIES = ["als", "explicit_mf", "popularity", "random", "user_knn", "neural_mf"]
# Strategies needing item features
FEATURE_STRATEGIES = ["linucb"]

# Skip implicit_als and implicit_bpr if implicit library is not installed
try:
    import implicit  # noqa: F401
    BASIC_STRATEGIES += ["implicit_als", "implicit_bpr"]
except ImportError:
    pass


def load_fixture():
    train = pd.read_csv(FIXTURES_DIR / "train.csv")
    test = pd.read_csv(FIXTURES_DIR / "test.csv")
    item_features = np.load(FIXTURES_DIR / "item_features.npy")
    return train, test, item_features


def evaluate_strategy(
    strategy: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    item_features: np.ndarray,
    top_k: int,
    seed: int,
) -> dict:
    """Fit and evaluate one strategy, returning metrics + timing."""
    np.random.seed(seed)

    kwargs = {}
    if strategy == "neural_mf":
        kwargs = {"epochs": 3, "emb_dim": 32, "hidden": (64, 32), "loss": "bpr"}
    elif strategy == "als":
        kwargs = {"emb_dim": 32, "epochs": 5}
    elif strategy == "explicit_mf":
        kwargs = {"emb_dim": 32, "epochs": 5}

    rec = OrchidRecommender(strategy=strategy, **kwargs)

    # Fit
    t0 = time.perf_counter()
    fit_kwargs = {"user_col": "user_id", "item_col": "item_id"}
    if strategy in ("explicit_mf",):
        fit_kwargs["rating_col"] = "rating"
    if strategy == "linucb":
        rec.fit(train, item_features=item_features, **fit_kwargs)
    else:
        rec.fit(train, **fit_kwargs)
    fit_time = time.perf_counter() - t0

    # Evaluate on test set
    # Build ground truth: per-user set of relevant items (label >= 1 or rating >= 3.5)
    test_relevant = test[test["label"] >= 1.0].groupby("user_id")["item_id"].apply(set).to_dict()

    precisions, recalls, ndcgs = [], [], []
    infer_times = []

    users_to_eval = [u for u in test_relevant if u in rec._user2idx][:50]  # cap for speed

    for user_id in users_to_eval:
        t1 = time.perf_counter()
        recs = rec.recommend(user_id, top_k=top_k, filter_seen=True)
        infer_times.append(time.perf_counter() - t1)

        rec_ids = [r.item_id for r in recs]
        relevant = test_relevant.get(user_id, set())

        if not relevant or not rec_ids:
            continue

        hits = sum(1 for r in rec_ids if r in relevant)
        precisions.append(hits / len(rec_ids))
        recalls.append(hits / len(relevant))

        # NDCG
        dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(rec_ids) if r in relevant)
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), len(rec_ids))))
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

    return {
        "strategy": strategy,
        "seed": seed,
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "fit_time_sec": round(fit_time, 3),
        "infer_p50_ms": round(float(np.percentile(infer_times, 50)) * 1000, 2) if infer_times else 0.0,
        "infer_p95_ms": round(float(np.percentile(infer_times, 95)) * 1000, 2) if infer_times else 0.0,
        "users_evaluated": len(users_to_eval),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark all strategies on fixture data")
    parser.add_argument("--seeds", default="42,13,17", help="Comma-separated seeds")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-golden", action="store_true", help="Save results as golden baselines")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    train, test, item_features = load_fixture()

    print(f"Fixture: {len(train)} train, {len(test)} test")
    print(f"Strategies: {BASIC_STRATEGIES + FEATURE_STRATEGIES}")
    print(f"Seeds: {seeds}, top_k: {args.top_k}")
    print("=" * 70)

    all_results = []

    for strategy in BASIC_STRATEGIES + FEATURE_STRATEGIES:
        for seed in seeds:
            print(f"  {strategy} (seed={seed})...", end=" ", flush=True)
            try:
                result = evaluate_strategy(strategy, train, test, item_features, args.top_k, seed)
                all_results.append(result)
                print(f"P@{args.top_k}={result['precision_at_k']:.3f} "
                      f"NDCG@{args.top_k}={result['ndcg_at_k']:.3f} "
                      f"fit={result['fit_time_sec']}s "
                      f"infer_p50={result['infer_p50_ms']}ms")
            except Exception as e:
                print(f"FAILED: {e}")
                all_results.append({"strategy": strategy, "seed": seed, "error": str(e)})

    # Aggregate across seeds
    aggregated = {}
    for strategy in set(r["strategy"] for r in all_results if "error" not in r):
        runs = [r for r in all_results if r["strategy"] == strategy and "error" not in r]
        if runs:
            aggregated[strategy] = {
                "precision_at_k_mean": round(float(np.mean([r["precision_at_k"] for r in runs])), 4),
                "precision_at_k_std": round(float(np.std([r["precision_at_k"] for r in runs])), 4),
                "ndcg_at_k_mean": round(float(np.mean([r["ndcg_at_k"] for r in runs])), 4),
                "ndcg_at_k_std": round(float(np.std([r["ndcg_at_k"] for r in runs])), 4),
                "recall_at_k_mean": round(float(np.mean([r["recall_at_k"] for r in runs])), 4),
                "fit_time_mean_sec": round(float(np.mean([r["fit_time_sec"] for r in runs])), 3),
                "infer_p50_mean_ms": round(float(np.mean([r["infer_p50_ms"] for r in runs])), 2),
                "infer_p95_mean_ms": round(float(np.mean([r["infer_p95_ms"] for r in runs])), 2),
            }

    output = {
        "meta": {
            "top_k": args.top_k,
            "seeds": seeds,
            "fixture": "benchmarks/fixtures/",
            "num_train": len(train),
            "num_test": len(test),
        },
        "per_run": all_results,
        "aggregated": aggregated,
    }

    print("\n" + "=" * 70)
    print("Aggregated results:")
    for strat, metrics in sorted(aggregated.items()):
        print(f"  {strat:20s}  P@k={metrics['precision_at_k_mean']:.4f}±{metrics['precision_at_k_std']:.4f}  "
              f"NDCG={metrics['ndcg_at_k_mean']:.4f}  fit={metrics['fit_time_mean_sec']}s")

    if args.save_golden:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path = GOLDEN_DIR / "baselines.json"
        with open(golden_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nGolden baselines saved to {golden_path}")

    return output


if __name__ == "__main__":
    main()
