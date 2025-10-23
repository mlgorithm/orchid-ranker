"""Evaluation utilities for Orchid Ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def precision_at_k(recommended: Sequence[int], relevant: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    recommended = list(recommended)[:k]
    relevant_set = set(int(r) for r in relevant)
    hits = sum(1 for item in recommended if item in relevant_set)
    return hits / float(k)


def recall_at_k(recommended: Sequence[int], relevant: Sequence[int], k: int) -> float:
    relevant_set = set(int(r) for r in relevant)
    if not relevant_set:
        return 0.0
    recommended = list(recommended)[:k]
    hits = sum(1 for item in recommended if item in relevant_set)
    return hits / float(len(relevant_set))


def ndcg_at_k(recommended: Sequence[int], relevant: Dict[int, float], k: int) -> float:
    if k <= 0:
        return 0.0
    recommended = list(recommended)[:k]
    gains = []
    for rank, item in enumerate(recommended, start=1):
        rel = float(relevant.get(int(item), 0.0))
        gain = (2 ** rel - 1) / np.log2(rank + 1)
        gains.append(gain)
    dcg = float(np.sum(gains))
    if not relevant:
        return 0.0
    ideal = sorted((float(v) for v in relevant.values()), reverse=True)
    ideal = ideal[:k]
    ideal_gains = [(2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal)]
    idcg = float(np.sum(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(recommended: Sequence[int], relevant: Sequence[int], k: int) -> float:
    relevant_set = set(int(r) for r in relevant)
    if not relevant_set:
        return 0.0
    recommended = list(recommended)[:k]
    score = 0.0
    hits = 0
    for idx, item in enumerate(recommended, start=1):
        if item in relevant_set:
            hits += 1
            score += hits / idx
    return score / len(relevant_set)


# ---------------------------------------------------------------------------
# Calibration & engagement metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(preds: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if preds.size == 0:
        return 0.0
    assert preds.shape == labels.shape
    bins = max(1, int(bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (preds >= lo) & (preds < hi)
        if not mask.any():
            continue
        bucket_acc = labels[mask].mean()
        bucket_conf = preds[mask].mean()
        total += mask.mean() * abs(bucket_acc - bucket_conf)
    return float(total)


# ---------------------------------------------------------------------------
# Summary container
# ---------------------------------------------------------------------------

@dataclass
class RankingReport:
    precision_at_5: float
    recall_at_5: float
    map_at_10: float
    ndcg_at_10: float


def evaluate_recommendations(
    recommendations: Dict[int, Sequence[int]],
    relevant: Dict[int, Sequence[int]],
    *,
    k_prec: int = 5,
    k_rec: int = 5,
    k_map: int = 10,
    k_ndcg: int = 10,
) -> RankingReport:
    prec = []
    rec = []
    ap = []
    ndcg = []
    relevant_dict = {uid: set(map(int, items)) for uid, items in relevant.items()}
    for user_id, slate in recommendations.items():
        rel_items = relevant_dict.get(user_id, set())
        prec.append(precision_at_k(slate, rel_items, k_prec))
        rec.append(recall_at_k(slate, rel_items, k_rec))
        ap.append(average_precision(slate, rel_items, k_map))
        rel_scores = {item: 1.0 for item in rel_items}
        ndcg.append(ndcg_at_k(slate, rel_scores, k_ndcg))
    return RankingReport(
        precision_at_5=float(np.mean(prec) if prec else 0.0),
        recall_at_5=float(np.mean(rec) if rec else 0.0),
        map_at_10=float(np.mean(ap) if ap else 0.0),
        ndcg_at_10=float(np.mean(ndcg) if ndcg else 0.0),
    )


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "average_precision",
    "expected_calibration_error",
    "RankingReport",
    "evaluate_recommendations",
]
