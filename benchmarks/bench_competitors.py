"""Cross-library benchmark: Orchid Ranker vs Surprise vs implicit vs LightFM.

Runs all libraries on the same frozen fixture dataset and reports:
- P@10, R@10, NDCG@10 (quality)
- Training time, inference latency p50/p95 (performance)
- Memory usage (resource)

Usage:
    python benchmarks/bench_competitors.py [--top-k 10] [--seeds 42,13,17] [--output results.json]

Requirements:
    pip install scikit-surprise implicit lightfm orchid-ranker
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orchid_ranker import OrchidRecommender

# Guard imports for optional dependencies
try:
    from surprise import Dataset, Reader, SVD, NMF, KNNBasic
    HAS_SURPRISE = True
except ImportError:
    HAS_SURPRISE = False

try:
    from implicit.als import AlternatingLeastSquares as ImplicitALS
    from implicit.bpr import BayesianPersonalizedRanking
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False

try:
    from lightfm import LightFM
    from lightfm.data import Dataset as LightFMDataset
    HAS_LIGHTFM = True
except ImportError:
    HAS_LIGHTFM = False

try:
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load_fixture() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load frozen fixture data: train.csv, test.csv, item_features.npy.

    Returns:
        train: DataFrame with columns [user_id, item_id, rating, label]
        test: DataFrame with columns [user_id, item_id, rating, label]
        item_features: (300, 8) numpy array
    """
    train = pd.read_csv(FIXTURES_DIR / "train.csv")
    test = pd.read_csv(FIXTURES_DIR / "test.csv")
    item_features = np.load(FIXTURES_DIR / "item_features.npy")
    return train, test, item_features


def build_ground_truth(test: pd.DataFrame) -> Dict[int, set]:
    """Build ground truth: {user_id: set(relevant_item_ids)}.

    Uses label >= 1.0 as relevance threshold.
    """
    return test[test["label"] >= 1.0].groupby("user_id")["item_id"].apply(set).to_dict()


def measure_memory(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Measure peak memory usage during function execution.

    Args:
        func: callable to measure
        *args, **kwargs: arguments to pass to func

    Returns:
        (result, peak_memory_mb): function result and peak memory in MB
    """
    tracemalloc.start()
    try:
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        return result, peak_mb
    finally:
        tracemalloc.stop()


def compute_metrics(
    recs_list: List[List[int]],
    ground_truth: Dict[int, set],
    users_to_eval: List[int],
) -> Dict[str, float]:
    """Compute P@K, R@K, NDCG@K from recommendation lists.

    Args:
        recs_list: list of recommendation lists (top-k item ids per user)
        ground_truth: {user_id: set(relevant_items)}
        users_to_eval: list of users that were evaluated (for alignment)

    Returns:
        dict with precision_at_k, recall_at_k, ndcg_at_k (all floats, 0.0 if empty)
    """
    precisions, recalls, ndcgs = [], [], []

    for user_id, rec_ids in zip(users_to_eval, recs_list):
        relevant = ground_truth.get(user_id, set())

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
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


# ==============================================================================
# ORCHID RANKER
# ==============================================================================

def evaluate_orchid(
    strategy: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate Orchid Ranker with specified strategy.

    Args:
        strategy: e.g., "als", "neural_mf", "popularity", "user_knn"
        train: training DataFrame
        test: test DataFrame
        top_k: number of recommendations per user
        seed: random seed

    Returns:
        dict with metrics (precision, recall, ndcg, timing, memory)
    """
    np.random.seed(seed)

    # Configure strategy with tuned hyperparameters
    kwargs = {}
    if strategy == "neural_mf":
        kwargs = {"epochs": 15, "emb_dim": 64, "hidden": (128, 64), "loss": "bpr", "batch_size": 512, "lr": 0.001}
    elif strategy == "als":
        kwargs = {"emb_dim": 64, "epochs": 15, "lr": 0.01}
    elif strategy == "explicit_mf":
        kwargs = {"emb_dim": 100, "epochs": 30, "lr": 0.003}
    elif strategy == "auto":
        kwargs = {}  # auto will pick explicit_mf with good defaults

    rec = OrchidRecommender(strategy=strategy, **kwargs)

    # Determine whether to pass explicit ratings
    use_ratings = strategy in ("explicit_mf", "auto")

    # Fit with memory and timing measurement
    def fit_model():
        fit_kwargs = dict(user_col="user_id", item_col="item_id")
        if use_ratings and "rating" in train.columns:
            fit_kwargs["rating_col"] = "rating"
        rec.fit(train, **fit_kwargs)

    t0 = time.perf_counter()
    _, fit_memory_mb = measure_memory(fit_model)
    fit_time = time.perf_counter() - t0

    # Build ground truth
    ground_truth = build_ground_truth(test)

    # Inference on subset of users
    users_to_eval = [u for u in ground_truth if u in rec._user2idx][:50]
    infer_times = []
    recs_list = []

    for user_id in users_to_eval:
        t1 = time.perf_counter()
        recs = rec.recommend(user_id, top_k=top_k, filter_seen=True)
        infer_times.append((time.perf_counter() - t1) * 1000)  # ms
        recs_list.append([r.item_id for r in recs])

    metrics = compute_metrics(recs_list, ground_truth, users_to_eval)

    return {
        "library": "orchid-ranker",
        "algorithm": strategy,
        "seed": seed,
        "precision_at_k": metrics["precision_at_k"],
        "recall_at_k": metrics["recall_at_k"],
        "ndcg_at_k": metrics["ndcg_at_k"],
        "fit_time_sec": round(fit_time, 3),
        "infer_p50_ms": round(float(np.percentile(infer_times, 50)), 2) if infer_times else 0.0,
        "infer_p95_ms": round(float(np.percentile(infer_times, 95)), 2) if infer_times else 0.0,
        "peak_memory_mb": round(fit_memory_mb, 1),
        "users_evaluated": len(users_to_eval),
    }


# ==============================================================================
# SURPRISE
# ==============================================================================

def evaluate_surprise(
    algorithm: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate Surprise library algorithm.

    Args:
        algorithm: "svd", "nmf", or "knnbasic"
        train: training DataFrame
        test: test DataFrame
        top_k: number of recommendations per user
        seed: random seed

    Returns:
        dict with metrics or error info
    """
    if not HAS_SURPRISE:
        return {
            "library": "surprise",
            "algorithm": algorithm,
            "seed": seed,
            "error": "surprise not installed",
        }

    np.random.seed(seed)

    try:
        # Prepare Surprise dataset
        rating_min = float(train["rating"].min())
        rating_max = float(train["rating"].max())
        reader = Reader(rating_scale=(rating_min, rating_max))
        data = Dataset.load_from_df(train[["user_id", "item_id", "rating"]], reader)
        trainset = data.build_full_trainset()

        # Fit algorithm
        if algorithm == "svd":
            algo = SVD(n_epochs=10, biased=True, random_state=seed)
        elif algorithm == "nmf":
            algo = NMF(n_epochs=10, random_state=seed)
        elif algorithm == "knnbasic":
            algo = KNNBasic(random_state=seed)
        else:
            return {"library": "surprise", "algorithm": algorithm, "seed": seed, "error": f"unknown algorithm {algorithm}"}

        def fit_algo():
            algo.fit(trainset)

        t0 = time.perf_counter()
        _, fit_memory_mb = measure_memory(fit_algo)
        fit_time = time.perf_counter() - t0

        # Build ground truth
        ground_truth = build_ground_truth(test)

        # Get all item ids
        all_items = sorted(train["item_id"].unique())

        # Inference: evaluate users present in both ground truth and training set
        known_users = {trainset.to_raw_uid(iid) for iid in trainset.all_users()}
        users_to_eval = [u for u in ground_truth if u in known_users][:50]
        infer_times = []
        recs_list = []

        for user_id in users_to_eval:
            t1 = time.perf_counter()
            # Score all items
            scores = []
            for item_id in all_items:
                try:
                    pred = algo.predict(str(user_id), str(item_id))
                    scores.append((pred.est, item_id))
                except Exception:
                    scores.append((float("-inf"), item_id))
            infer_times.append((time.perf_counter() - t1) * 1000)  # ms

            # Top-k
            scores.sort(key=lambda x: x[0], reverse=True)
            recs_list.append([item_id for _, item_id in scores[:top_k]])

        metrics = compute_metrics(recs_list, ground_truth, users_to_eval)

        return {
            "library": "surprise",
            "algorithm": algorithm,
            "seed": seed,
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "fit_time_sec": round(fit_time, 3),
            "infer_p50_ms": round(float(np.percentile(infer_times, 50)), 2) if infer_times else 0.0,
            "infer_p95_ms": round(float(np.percentile(infer_times, 95)), 2) if infer_times else 0.0,
            "peak_memory_mb": round(fit_memory_mb, 1),
            "users_evaluated": len(users_to_eval),
        }
    except Exception as e:
        return {
            "library": "surprise",
            "algorithm": algorithm,
            "seed": seed,
            "error": str(e),
        }


# ==============================================================================
# IMPLICIT
# ==============================================================================

def evaluate_implicit(
    algorithm: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate implicit library algorithm.

    Args:
        algorithm: "als" or "bpr"
        train: training DataFrame
        test: test DataFrame
        top_k: number of recommendations per user
        seed: random seed

    Returns:
        dict with metrics or error info
    """
    if not HAS_IMPLICIT or not HAS_SCIPY:
        return {
            "library": "implicit",
            "algorithm": algorithm,
            "seed": seed,
            "error": "implicit or scipy not installed",
        }

    np.random.seed(seed)

    try:
        # Build sparse user-item matrix
        user_ids = train["user_id"].unique()
        item_ids = train["item_id"].unique()
        user_to_idx = {u: i for i, u in enumerate(sorted(user_ids))}
        item_to_idx = {i: j for j, i in enumerate(sorted(item_ids))}

        rows = [user_to_idx[u] for u in train["user_id"]]
        cols = [item_to_idx[i] for i in train["item_id"]]
        data = train["label"].values  # use binary label (0/1) for implicit

        user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_to_idx), len(item_to_idx))
        )

        # Fit model
        if algorithm == "als":
            model = ImplicitALS(factors=32, iterations=15, random_state=seed)
        elif algorithm == "bpr":
            model = BayesianPersonalizedRanking(factors=32, iterations=15, random_state=seed)
        else:
            return {"library": "implicit", "algorithm": algorithm, "seed": seed, "error": f"unknown algorithm {algorithm}"}

        def fit_model_fn():
            model.fit(user_item_matrix)

        t0 = time.perf_counter()
        _, fit_memory_mb = measure_memory(fit_model_fn)
        fit_time = time.perf_counter() - t0

        # Build ground truth
        ground_truth = build_ground_truth(test)

        # Inference
        users_to_eval = [u for u in ground_truth if u in user_to_idx][:50]
        infer_times = []
        recs_list = []

        for user_id in users_to_eval:
            user_idx = user_to_idx[user_id]
            t1 = time.perf_counter()
            # Get recommendations
            recs, scores = model.recommend(user_idx, user_item_matrix[user_idx], N=top_k, filter_items=None)
            infer_times.append((time.perf_counter() - t1) * 1000)  # ms
            # Map back to item ids
            idx_to_item = {j: i for i, j in item_to_idx.items()}
            recs_list.append([idx_to_item[r] for r in recs])

        metrics = compute_metrics(recs_list, ground_truth, users_to_eval)

        return {
            "library": "implicit",
            "algorithm": algorithm,
            "seed": seed,
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "fit_time_sec": round(fit_time, 3),
            "infer_p50_ms": round(float(np.percentile(infer_times, 50)), 2) if infer_times else 0.0,
            "infer_p95_ms": round(float(np.percentile(infer_times, 95)), 2) if infer_times else 0.0,
            "peak_memory_mb": round(fit_memory_mb, 1),
            "users_evaluated": len(users_to_eval),
        }
    except Exception as e:
        return {
            "library": "implicit",
            "algorithm": algorithm,
            "seed": seed,
            "error": str(e),
        }


# ==============================================================================
# LIGHTFM
# ==============================================================================

def evaluate_lightfm(
    model_type: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    top_k: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate LightFM model.

    Args:
        model_type: "warp" or "bpr"
        train: training DataFrame
        test: test DataFrame
        top_k: number of recommendations per user
        seed: random seed

    Returns:
        dict with metrics or error info
    """
    if not HAS_LIGHTFM or not HAS_SCIPY:
        return {
            "library": "lightfm",
            "algorithm": model_type,
            "seed": seed,
            "error": "lightfm or scipy not installed",
        }

    np.random.seed(seed)

    try:
        # Build LightFM dataset
        dataset = LightFMDataset()
        dataset.fit(train["user_id"], train["item_id"])

        # Create interaction matrix
        (interactions, weights) = dataset.build_interactions(
            [(u, i) for u, i in zip(train["user_id"], train["item_id"])]
        )

        # Fit model
        if model_type == "warp":
            model = LightFM(loss="warp", random_state=seed)
        elif model_type == "bpr":
            model = LightFM(loss="bpr", random_state=seed)
        else:
            return {"library": "lightfm", "algorithm": model_type, "seed": seed, "error": f"unknown model {model_type}"}

        def fit_model_fn():
            model.fit(interactions, epochs=10, num_threads=1)

        t0 = time.perf_counter()
        _, fit_memory_mb = measure_memory(fit_model_fn)
        fit_time = time.perf_counter() - t0

        # Build ground truth
        ground_truth = build_ground_truth(test)

        # Get user/item id mappings
        user_ids_unique = sorted(train["user_id"].unique())
        item_ids_unique = sorted(train["item_id"].unique())
        user_to_idx = {u: i for i, u in enumerate(user_ids_unique)}
        item_to_idx = {i: j for j, i in enumerate(item_ids_unique)}

        # Inference
        users_to_eval = [u for u in ground_truth if u in user_to_idx][:50]
        infer_times = []
        recs_list = []

        for user_id in users_to_eval:
            user_idx = user_to_idx[user_id]
            t1 = time.perf_counter()
            # Predict scores for all items
            item_indices = np.array([item_to_idx[i] for i in item_ids_unique])
            scores = model.predict(user_idx, item_indices)
            infer_times.append((time.perf_counter() - t1) * 1000)  # ms
            # Top-k
            top_indices = np.argsort(-scores)[:top_k]
            recs_list.append([item_ids_unique[i] for i in top_indices])

        metrics = compute_metrics(recs_list, ground_truth, users_to_eval)

        return {
            "library": "lightfm",
            "algorithm": model_type,
            "seed": seed,
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "fit_time_sec": round(fit_time, 3),
            "infer_p50_ms": round(float(np.percentile(infer_times, 50)), 2) if infer_times else 0.0,
            "infer_p95_ms": round(float(np.percentile(infer_times, 95)), 2) if infer_times else 0.0,
            "peak_memory_mb": round(fit_memory_mb, 1),
            "users_evaluated": len(users_to_eval),
        }
    except Exception as e:
        return {
            "library": "lightfm",
            "algorithm": model_type,
            "seed": seed,
            "error": str(e),
        }


# ==============================================================================
# MARKDOWN TABLE GENERATION
# ==============================================================================

def generate_markdown_table(results: Dict[str, List[Dict]]) -> str:
    """Generate markdown table from aggregated results.

    Args:
        results: dict with library -> list of run dicts

    Returns:
        markdown string suitable for docs
    """
    lines = [
        "# Benchmark Results: Cross-Library Comparison",
        "",
        "| Library | Algorithm | P@10 | R@10 | NDCG@10 | Fit (s) | Infer p50 (ms) | Infer p95 (ms) | Memory (MB) |",
        "|---------|-----------|------|------|---------|---------|----------------|----------------|------------|",
    ]

    for lib_algo in sorted(results.keys()):
        runs = results[lib_algo]
        if not runs or any("error" in r for r in runs):
            continue

        p = np.mean([r["precision_at_k"] for r in runs])
        r = np.mean([r["recall_at_k"] for r in runs])
        n = np.mean([r["ndcg_at_k"] for r in runs])
        ft = np.mean([r["fit_time_sec"] for r in runs])
        ip50 = np.mean([r["infer_p50_ms"] for r in runs])
        ip95 = np.mean([r["infer_p95_ms"] for r in runs])
        mem = np.mean([r["peak_memory_mb"] for r in runs])

        lib, algo = lib_algo.split(" / ")
        lines.append(
            f"| {lib} | {algo} | {p:.4f} | {r:.4f} | {n:.4f} | {ft:.3f} | {ip50:.2f} | {ip95:.2f} | {mem:.1f} |"
        )

    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-library benchmark: Orchid vs Surprise vs implicit vs LightFM"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for metrics")
    parser.add_argument("--seeds", default="42,13,17", help="Comma-separated seeds")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/bench_competitors_results.json"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    train, test, item_features = load_fixture()

    print("=" * 80)
    print("Cross-Library Recommender Benchmark")
    print("=" * 80)
    print(f"Fixture: {len(train)} train, {len(test)} test")
    print(f"Top-K: {args.top_k}, Seeds: {seeds}")
    print(f"Orchid: {'available' if True else 'missing'}")
    print(f"Surprise: {'available' if HAS_SURPRISE else 'missing'}")
    print(f"implicit: {'available' if HAS_IMPLICIT else 'missing'}")
    print(f"LightFM: {'available' if HAS_LIGHTFM else 'missing'}")
    print("=" * 80)

    all_results = []

    # Orchid strategies
    orchid_strategies = ["explicit_mf", "auto", "als", "neural_mf", "popularity", "user_knn"]
    for strategy in orchid_strategies:
        for seed in seeds:
            print(f"Orchid {strategy:12s} (seed={seed})...", end=" ", flush=True)
            try:
                result = evaluate_orchid(strategy, train, test, args.top_k, seed)
                all_results.append(result)
                print(f"P@{args.top_k}={result['precision_at_k']:.3f} "
                      f"NDCG={result['ndcg_at_k']:.3f} "
                      f"fit={result['fit_time_sec']}s")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({"library": "orchid-ranker", "algorithm": strategy, "seed": seed, "error": str(e)})

    # Surprise algorithms
    if HAS_SURPRISE:
        surprise_algos = ["svd", "nmf", "knnbasic"]
        for algo in surprise_algos:
            for seed in seeds:
                print(f"Surprise {algo:12s} (seed={seed})...", end=" ", flush=True)
                try:
                    result = evaluate_surprise(algo, train, test, args.top_k, seed)
                    all_results.append(result)
                    if "error" not in result:
                        print(f"P@{args.top_k}={result['precision_at_k']:.3f} "
                              f"NDCG={result['ndcg_at_k']:.3f} "
                              f"fit={result['fit_time_sec']}s")
                    else:
                        print(f"ERROR: {result['error']}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({"library": "surprise", "algorithm": algo, "seed": seed, "error": str(e)})
    else:
        print("SKIP: Surprise not installed")

    # implicit algorithms
    if HAS_IMPLICIT:
        implicit_algos = ["als", "bpr"]
        for algo in implicit_algos:
            for seed in seeds:
                print(f"implicit {algo:12s} (seed={seed})...", end=" ", flush=True)
                try:
                    result = evaluate_implicit(algo, train, test, args.top_k, seed)
                    all_results.append(result)
                    if "error" not in result:
                        print(f"P@{args.top_k}={result['precision_at_k']:.3f} "
                              f"NDCG={result['ndcg_at_k']:.3f} "
                              f"fit={result['fit_time_sec']}s")
                    else:
                        print(f"ERROR: {result['error']}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({"library": "implicit", "algorithm": algo, "seed": seed, "error": str(e)})
    else:
        print("SKIP: implicit not installed")

    # LightFM models
    if HAS_LIGHTFM:
        lightfm_models = ["warp", "bpr"]
        for model_type in lightfm_models:
            for seed in seeds:
                print(f"LightFM {model_type:12s} (seed={seed})...", end=" ", flush=True)
                try:
                    result = evaluate_lightfm(model_type, train, test, args.top_k, seed)
                    all_results.append(result)
                    if "error" not in result:
                        print(f"P@{args.top_k}={result['precision_at_k']:.3f} "
                              f"NDCG={result['ndcg_at_k']:.3f} "
                              f"fit={result['fit_time_sec']}s")
                    else:
                        print(f"ERROR: {result['error']}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({"library": "lightfm", "algorithm": model_type, "seed": seed, "error": str(e)})
    else:
        print("SKIP: LightFM not installed")

    # Aggregate results
    print("\n" + "=" * 80)
    print("Aggregated Results (mean ± std over seeds)")
    print("=" * 80)

    aggregated = defaultdict(list)
    for result in all_results:
        if "error" not in result:
            key = f"{result['library']} / {result['algorithm']}"
            aggregated[key].append(result)

    # Print summary
    for lib_algo in sorted(aggregated.keys()):
        runs = aggregated[lib_algo]
        if not runs:
            continue

        p_mean = np.mean([r["precision_at_k"] for r in runs])
        p_std = np.std([r["precision_at_k"] for r in runs])
        n_mean = np.mean([r["ndcg_at_k"] for r in runs])
        n_std = np.std([r["ndcg_at_k"] for r in runs])
        r_mean = np.mean([r["recall_at_k"] for r in runs])
        r_std = np.std([r["recall_at_k"] for r in runs])
        ft_mean = np.mean([r["fit_time_sec"] for r in runs])

        print(f"{lib_algo:30s} | P@10 {p_mean:.4f}±{p_std:.4f} | "
              f"R@10 {r_mean:.4f}±{r_std:.4f} | NDCG {n_mean:.4f}±{n_std:.4f} | "
              f"fit {ft_mean:.3f}s")

    # Save results
    output = {
        "meta": {
            "top_k": args.top_k,
            "seeds": seeds,
            "fixture": "benchmarks/fixtures/",
            "num_train": len(train),
            "num_test": len(test),
            "libraries": {
                "orchid-ranker": "installed",
                "surprise": "installed" if HAS_SURPRISE else "missing",
                "implicit": "installed" if HAS_IMPLICIT else "missing",
                "lightfm": "installed" if HAS_LIGHTFM else "missing",
            },
        },
        "per_run": all_results,
        "aggregated": dict(aggregated),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Generate markdown
    md_table = generate_markdown_table(dict(aggregated))
    print("\n" + md_table)


if __name__ == "__main__":
    main()
