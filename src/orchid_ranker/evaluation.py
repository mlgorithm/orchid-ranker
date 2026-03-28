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
    """Compute Precision@k metric.

    Measures the fraction of top-k recommended items that are relevant.

    Parameters
    ----------
    recommended : Sequence[int]
        List of recommended item IDs.
    relevant : Sequence[int]
        List of relevant/ground-truth item IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Precision@k, in [0, 1].
    """
    if k <= 0:
        return 0.0
    recommended = list(recommended)[:k]
    relevant_set = set(int(r) for r in relevant)
    hits = sum(1 for item in recommended if item in relevant_set)
    return hits / float(k)


def recall_at_k(recommended: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Compute Recall@k metric.

    Measures the fraction of ground-truth relevant items that appear in top-k recommendations.

    Parameters
    ----------
    recommended : Sequence[int]
        List of recommended item IDs.
    relevant : Sequence[int]
        List of relevant/ground-truth item IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Recall@k, in [0, 1].
    """
    relevant_set = set(int(r) for r in relevant)
    if not relevant_set:
        return 0.0
    recommended = list(recommended)[:k]
    hits = sum(1 for item in recommended if item in relevant_set)
    return hits / float(len(relevant_set))


def ndcg_at_k(recommended: Sequence[int], relevant: Dict[int, float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@k).

    Measures ranking quality considering both relevance scores and position discounts.

    Parameters
    ----------
    recommended : Sequence[int]
        List of recommended item IDs.
    relevant : dict
        Mapping from item ID to relevance score (e.g., {item_id: 1.0 or 0.0}).
    k : int
        Cut-off rank.

    Returns
    -------
    float
        NDCG@k, in [0, 1].
    """
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
    """Compute Mean Average Precision (MAP) for binary relevance.

    Measures ranking quality by averaging precision at each position where a
    relevant item is found.

    Parameters
    ----------
    recommended : Sequence[int]
        List of recommended item IDs.
    relevant : Sequence[int]
        List of relevant/ground-truth item IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Average precision, in [0, 1].
    """
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
    """Compute Expected Calibration Error (ECE).

    Measures the gap between predicted confidence and actual accuracy across
    confidence bins. Lower ECE indicates better calibration.

    Parameters
    ----------
    preds : np.ndarray
        Predicted probabilities, in [0, 1].
    labels : np.ndarray
        Binary labels {0, 1}, same shape as preds.
    bins : int, optional
        Number of confidence bins (default: 10).

    Returns
    -------
    float
        Expected calibration error, in [0, 1].
    """
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
    """Container for ranking evaluation metrics.

    Attributes
    ----------
    precision_at_5 : float
        Mean Precision@5 across users.
    recall_at_5 : float
        Mean Recall@5 across users.
    map_at_10 : float
        Mean Average Precision@10 across users.
    ndcg_at_10 : float
        Mean NDCG@10 across users.
    """

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
    """Evaluate recommendation lists against ground-truth relevant items.

    Computes a comprehensive ranking report with precision, recall, MAP, and NDCG metrics.

    Parameters
    ----------
    recommendations : dict
        Mapping from user ID to list of recommended item IDs.
    relevant : dict
        Mapping from user ID to list of relevant/ground-truth item IDs.
    k_prec : int, optional
        Cut-off for precision computation (default: 5).
    k_rec : int, optional
        Cut-off for recall computation (default: 5).
    k_map : int, optional
        Cut-off for MAP computation (default: 10).
    k_ndcg : int, optional
        Cut-off for NDCG computation (default: 10).

    Returns
    -------
    RankingReport
        Report with averaged metrics across all users.
    """
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


# ---------------------------------------------------------------------------
# Educational metrics
# ---------------------------------------------------------------------------


def learning_gain(pre_score: float, post_score: float) -> float:
    """Compute normalized learning gain.

    Measures the improvement from pre-test to post-test, normalized by the
    maximum possible gain. Returns 0.0 if pre-test score is already perfect.

    Parameters
    ----------
    pre_score : float
        Pre-test score, typically in [0, 1] where 1 is perfect.
    post_score : float
        Post-test score after instruction, typically in [0, 1].

    Returns
    -------
    float
        Normalized learning gain (post - pre) / (1 - pre). Returns 0.0 if pre == 1.0.
    """
    pre_score = float(pre_score)
    post_score = float(post_score)
    if pre_score >= 1.0:
        return 0.0
    return (post_score - pre_score) / (1.0 - pre_score)


def knowledge_coverage(mastered_skills: set, total_skills: set) -> float:
    """Compute fraction of total skills mastered.

    Measures the breadth of a student's competency across a skill domain.

    Parameters
    ----------
    mastered_skills : set
        Set of skill identifiers the student has demonstrated mastery of.
    total_skills : set
        Set of all skills in the domain.

    Returns
    -------
    float
        Fraction of skills mastered, in [0, 1]. Returns 0.0 if total_skills is empty.
    """
    total_skills = set(total_skills)
    if not total_skills:
        return 0.0
    mastered_skills = set(mastered_skills)
    return float(len(mastered_skills & total_skills)) / float(len(total_skills))


def curriculum_adherence(
    recommended_items: Sequence[int],
    prerequisite_graph: Dict[int, set],
    mastered: set,
) -> float:
    """Compute fraction of recommendations whose prerequisites are met.

    Ensures that the recommender respects the logical structure of the curriculum
    by only recommending items when their prerequisites have been mastered.

    Parameters
    ----------
    recommended_items : Sequence[int]
        Recommended item IDs in order.
    prerequisite_graph : dict
        Mapping from item ID to set of prerequisite item IDs.
        Example: {3: {1, 2}} means item 3 requires items 1 and 2.
    mastered : set
        Set of item IDs (prerequisites) already mastered.

    Returns
    -------
    float
        Fraction of recommendations with all prerequisites met, in [0, 1].
        Returns 1.0 if recommended_items is empty.
    """
    if not recommended_items:
        return 1.0
    mastered = set(mastered)
    adherent = 0
    for item_id in recommended_items:
        prereqs = prerequisite_graph.get(item_id, set())
        if prereqs.issubset(mastered):
            adherent += 1
    return float(adherent) / float(len(recommended_items))


def difficulty_appropriateness(
    recommended_difficulties: Sequence[float],
    student_ability: float,
    zpd_width: float = 0.25,
) -> float:
    """Compute fraction of recommendations within the student's Zone of Proximal Development.

    Implements Vygotsky's ZPD: recommendations should be slightly above current ability
    but not so far as to be discouraging. Items are in the ZPD if their difficulty is
    in [student_ability, student_ability + zpd_width].

    Parameters
    ----------
    recommended_difficulties : Sequence[float]
        Difficulty scores of recommended items, typically in [0, 1].
    student_ability : float
        Current ability level of the student, typically in [0, 1].
    zpd_width : float, optional
        Width of the zone of proximal development (default: 0.25).
        Items with difficulty in [ability, ability + width] are considered appropriate.

    Returns
    -------
    float
        Fraction of items within the ZPD, in [0, 1].
        Returns 1.0 if recommended_difficulties is empty.
    """
    if not recommended_difficulties:
        return 1.0
    student_ability = float(student_ability)
    zpd_width = float(zpd_width)
    zpd_min = student_ability
    zpd_max = student_ability + zpd_width
    in_zpd = sum(
        1 for diff in recommended_difficulties
        if zpd_min <= float(diff) <= zpd_max
    )
    return float(in_zpd) / float(len(recommended_difficulties))


def engagement_score(interactions: Sequence, total_available: int) -> float:
    """Compute ratio of items interacted with to items recommended.

    Measures student engagement as a fraction of total recommendations.

    Parameters
    ----------
    interactions : Sequence
        List or set of items the student interacted with.
    total_available : int
        Total number of items recommended.

    Returns
    -------
    float
        Fraction of recommended items that received interaction, in [0, 1].
        Returns 0.0 if total_available <= 0.
    """
    if total_available <= 0:
        return 0.0
    interactions = set(interactions) if not isinstance(interactions, set) else interactions
    return float(len(interactions)) / float(total_available)


@dataclass
class EducationalReport:
    """Container for educational evaluation metrics.

    Attributes
    ----------
    learning_gain : float
        Normalized learning gain: (post - pre) / (1 - pre).
    coverage : float
        Fraction of total skills mastered.
    adherence : float
        Fraction of recommendations with satisfied prerequisites.
    difficulty_fit : float
        Fraction of recommendations within student's ZPD.
    engagement : float
        Ratio of items interacted with to items recommended.
    """

    learning_gain: float
    coverage: float
    adherence: float
    difficulty_fit: float
    engagement: float


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "average_precision",
    "expected_calibration_error",
    "RankingReport",
    "evaluate_recommendations",
    "learning_gain",
    "knowledge_coverage",
    "curriculum_adherence",
    "difficulty_appropriateness",
    "engagement_score",
    "EducationalReport",
]
