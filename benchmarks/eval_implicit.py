"""Apples-to-apples implicit ranking benchmark on MovieLens 100K.

- Binary labels: rating >= 4 -> 1 else 0
- filter_seen=True, shared candidate set per user
- Metrics: Precision@10, Recall@10, NDCG@10 (mean±std over seeds)

Usage (CPU-safe defaults):
  PYTHONPATH=src python3 benchmarks/eval_implicit.py \
    --seeds 11 13 17 --top-users 400 --top-items 800 --k 10

"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from surprise import Dataset, Reader, SVD

from orchid_ranker import OrchidRecommender


def precision_recall_ndcg_at_k(user_relevant: Dict[int, set], user_slates: Dict[int, List[int]], k: int = 10) -> dict:
    log_den = [1.0 / math.log2(i + 2) for i in range(k)]
    P, R, N = [], [], []
    for uid, slate in user_slates.items():
        rel = user_relevant.get(uid)
        if not rel:
            continue
        hits = 0
        dcg = 0.0
        for idx, iid in enumerate(slate[:k]):
            if iid in rel:
                hits += 1
                dcg += log_den[idx]
        P.append(hits / max(1, min(k, len(slate))))
        R.append(hits / len(rel))
        ideal = min(len(rel), k)
        idcg = sum(log_den[:ideal]) if ideal > 0 else 0.0
        N.append((dcg / idcg) if idcg > 0 else 0.0)
    def avg(x):
        return float(sum(x) / len(x)) if x else float("nan")
    return {"P@10": avg(P), "R@10": avg(R), "NDCG@10": avg(N)}


def build_subset(frame: pd.DataFrame, seed: int, top_users: int, top_items: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(frame)) < 0.8
    train_df = frame[mask].copy()
    test_df = frame[~mask].copy()
    uids = train_df.groupby("user_id").size().sort_values(ascending=False).head(top_users).index
    iids = train_df.groupby("item_id").size().sort_values(ascending=False).head(top_items).index
    train_df = train_df[train_df.user_id.isin(uids) & train_df.item_id.isin(iids)].reset_index(drop=True)
    test_df = test_df[test_df.user_id.isin(train_df.user_id.unique()) & test_df.item_id.isin(train_df.item_id.unique())].reset_index(drop=True)
    return train_df, test_df


def evaluate_once(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 10) -> dict:
    # Binary labels
    train = train_df.copy(); test = test_df.copy()
    train["label"] = (train["rating"] >= 4.0).astype(float)
    test["label"] = (test["rating"] >= 4.0).astype(float)

    items = sorted(train.item_id.unique())
    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    user_rel = {
        int(u): set(g.loc[g["label"] >= 0.5, "item_id"].astype(int).tolist())
        for u, g in test.groupby("user_id")
        if (g["label"] >= 0.5).any()
    }

    results = {}

    # Popularity
    pop = train.groupby("item_id")["label"].mean().to_dict()
    slates = {}
    for u in user_rel.keys():
        cands = [i for i in items if i not in seen.get(u, set())]
        cands.sort(key=lambda i: pop.get(i, 0.0), reverse=True)
        slates[u] = cands[:k]
    results["popularity"] = precision_recall_ndcg_at_k(user_rel, slates, k)

    # Surprise SVD ranking (trained on binary labels)
    reader = Reader(rating_scale=(0.0, 1.0))
    svd_data = Dataset.load_from_df(train[["user_id", "item_id", "label"]], reader)
    svd_train = svd_data.build_full_trainset()
    svd = SVD(n_epochs=10, biased=True)
    svd.fit(svd_train)
    slates = {}
    for u in user_rel.keys():
        cands = [i for i in items if i not in seen.get(u, set())]
        scores = [(svd.predict(str(u), str(i)).est, i) for i in cands]
        scores.sort(key=lambda x: x[0], reverse=True)
        slates[u] = [i for _, i in scores[:k]]
    results["svd_ranking"] = precision_recall_ndcg_at_k(user_rel, slates, k)

    # Orchid implicit ALS
    als = OrchidRecommender(strategy="implicit_als", factors=32, iterations=12)
    als.fit(train[["user_id", "item_id", "label"]], rating_col="label")
    slates = {}
    for u in user_rel.keys():
        cands = [i for i in items if i not in seen.get(u, set())]
        scores = [(als.predict(u, i), i) for i in cands]
        scores.sort(key=lambda x: x[0], reverse=True)
        slates[u] = [i for _, i in scores[:k]]
    results["implicit_als"] = precision_recall_ndcg_at_k(user_rel, slates, k)

    # Orchid implicit BPR
    bpr = OrchidRecommender(strategy="implicit_bpr", factors=32, iterations=30)
    bpr.fit(train[["user_id", "item_id", "label"]], rating_col="label")
    slates = {}
    for u in user_rel.keys():
        cands = [i for i in items if i not in seen.get(u, set())]
        scores = [(bpr.predict(u, i), i) for i in cands]
        scores.sort(key=lambda x: x[0], reverse=True)
        slates[u] = [i for _, i in scores[:k]]
    results["implicit_bpr"] = precision_recall_ndcg_at_k(user_rel, slates, k)

    # Orchid Neural MF with BPR (two light configs)
    nmf3 = OrchidRecommender(strategy="neural_mf", loss="bpr", epochs=3, emb_dim=48, hidden=(128, 64))
    nmf3.fit(train[["user_id", "item_id", "label"]], rating_col="label")
    slates = {}
    for u in user_rel.keys():
        cands = [i for i in items if i not in seen.get(u, set())]
        scores = [(nmf3.predict(u, i), i) for i in cands]
        scores.sort(key=lambda x: x[0], reverse=True)
        slates[u] = [i for _, i in scores[:k]]
    results["neural_mf_bpr_e3"] = precision_recall_ndcg_at_k(user_rel, slates, k)

    nmf5 = OrchidRecommender(strategy="neural_mf", loss="bpr", epochs=5, emb_dim=64, hidden=(128, 64))
    nmf5.fit(train[["user_id", "item_id", "label"]], rating_col="label")
    slates = {}
    for u in user_rel.keys():
        cands = [i for i in items if i not in seen.get(u, set())]
        scores = [(nmf5.predict(u, i), i) for i in cands]
        scores.sort(key=lambda x: x[0], reverse=True)
        slates[u] = [i for _, i in scores[:k]]
    results["neural_mf_bpr_e5"] = precision_recall_ndcg_at_k(user_rel, slates, k)

    # Orchid Neural MF with sampled softmax (stronger implicit baseline)
    nmf_sm = OrchidRecommender(strategy="neural_mf", loss="softmax", neg_k=20, epochs=5, emb_dim=64, hidden=(128, 64))
    nmf_sm.fit(train[["user_id", "item_id", "label"]], rating_col="label")
    slates = {}
    for u in user_rel.keys():
        cands = [i for i in items if i not in seen.get(u, set())]
        scores = [(nmf_sm.predict(u, i), i) for i in cands]
        scores.sort(key=lambda x: x[0], reverse=True)
        slates[u] = [i for _, i in scores[:k]]
    results["neural_mf_softmax_e5_k20"] = precision_recall_ndcg_at_k(user_rel, slates, k)

    return results


def aggregate(measurements: List[dict]) -> dict:
    # model -> metric -> [values]
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for res in measurements:
        for name, m in res.items():
            for k, v in m.items():
                acc[name][k].append(float(v))
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for name, metrics in acc.items():
        out[name] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Implicit top-K apples-to-apples benchmark on ML-100K")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 13, 17], help="Random seeds for 80/20 split")
    p.add_argument("--top-users", type=int, default=400, help="Top-N users by frequency in train")
    p.add_argument("--top-items", type=int, default=800, help="Top-N items by frequency in train")
    p.add_argument("--k", type=int, default=10, help="K for P@K/Recall@K/NDCG@K")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = Dataset.load_builtin("ml-100k", prompt=False)
    raw = data.raw_ratings
    frame = pd.DataFrame(raw, columns=["user_id", "item_id", "rating", "timestamp"]).astype({"user_id": int, "item_id": int, "rating": float})

    runs = []
    for seed in args.seeds:
        train_df, test_df = build_subset(frame, seed, args.top_users, args.top_items)
        runs.append(evaluate_once(train_df, test_df, k=args.k))

    agg = aggregate(runs)
    print("\nImplicit (binary >=4) — mean±std over seeds:")
    for name, m in sorted(agg.items()):
        p, r, n = m["P@10"], m["R@10"], m["NDCG@10"]
        print(f"{name:18s} | P@10 {p[0]:.4f}±{p[1]:.4f} | R@10 {r[0]:.4f}±{r[1]:.4f} | NDCG@10 {n[0]:.4f}±{n[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
