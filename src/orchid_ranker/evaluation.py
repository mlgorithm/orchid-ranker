"""Evaluation utilities for Orchid Ranker.

Provides ranking metrics, calibration helpers, and progression-domain metrics
(stretch fit, category coverage, sequence adherence, engagement) used to
assess adaptive recommendation quality for any outcome-tracing domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
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


def recall_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
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


def average_precision(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Compute Average Precision (AP@k) for a single ranking.

    Averages precision at each rank position where a relevant item appears.
    To get Mean Average Precision (MAP), average AP across multiple users.

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
    if preds.shape != labels.shape:
        raise ValueError(f"preds and labels must have the same shape, got {preds.shape} vs {labels.shape}")
    bins = max(1, int(bins))
    # Vectorized binning with numpy.digitize instead of Python loop
    bin_indices = np.digitize(preds, np.linspace(0.0, 1.0, bins + 1)) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    total = 0.0
    n = len(preds)
    for b in range(bins):
        mask = bin_indices == b
        count = mask.sum()
        if count == 0:
            continue
        bucket_acc = labels[mask].mean()
        bucket_conf = preds[mask].mean()
        total += (count / n) * abs(bucket_acc - bucket_conf)
    return float(total)


# ---------------------------------------------------------------------------
# Summary container
# ---------------------------------------------------------------------------

@dataclass
class RankingReport:
    """Container for ranking evaluation metrics.

    Holds aggregated ranking metrics computed across multiple users.
    Field names reflect the default k values; use ``evaluate_recommendations()``
    keyword arguments to customize the cut-offs.

    Attributes
    ----------
    precision : float
        Mean Precision@k across users.
    recall : float
        Mean Recall@k across users.
    map : float
        Mean Average Precision@k across users.
    ndcg : float
        Mean NDCG@k across users.
    """

    precision: float
    recall: float
    map: float
    ndcg: float

    # Backward-compatible aliases for old field names
    @property
    def precision_at_5(self) -> float:
        return self.precision

    @property
    def recall_at_5(self) -> float:
        return self.recall

    @property
    def map_at_10(self) -> float:
        return self.map

    @property
    def ndcg_at_10(self) -> float:
        return self.ndcg


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
    relevant_dict = {int(uid): set(map(int, items)) for uid, items in relevant.items()}

    # Average over the full relevant-user set so omitted users contribute zero
    # instead of disappearing from the denominator.
    for user_id, rel_items in relevant_dict.items():
        slate = recommendations.get(user_id, [])
        prec.append(precision_at_k(slate, rel_items, k_prec))
        rec.append(recall_at_k(slate, rel_items, k_rec))
        ap.append(average_precision(slate, rel_items, k_map))
        rel_scores = {item: 1.0 for item in rel_items}
        ndcg.append(ndcg_at_k(slate, rel_scores, k_ndcg))
    return RankingReport(
        precision=float(np.mean(prec) if prec else 0.0),
        recall=float(np.mean(rec) if rec else 0.0),
        map=float(np.mean(ap) if ap else 0.0),
        ndcg=float(np.mean(ndcg) if ndcg else 0.0),
    )


# ---------------------------------------------------------------------------
# Educational metrics
# ---------------------------------------------------------------------------


def progression_gain(pre_score: float, post_score: float) -> float:
    """Compute normalized progression gain.

    Measures improvement from pre-assessment to post-assessment, normalized by
    the maximum possible gain. Works for any domain: education (learning gain),
    rehabilitation (recovery gain), fitness (performance gain), etc.

    Parameters
    ----------
    pre_score : float
        Pre-assessment score, typically in [0, 1] where 1 is perfect.
    post_score : float
        Post-assessment score, typically in [0, 1].

    Returns
    -------
    float
        Normalized gain (post - pre) / (1 - pre). Returns 0.0 if pre == 1.0.
        Can be negative if post_score < pre_score (indicating regression).
    """
    pre_score = float(pre_score)
    post_score = float(post_score)
    if pre_score >= 1.0:
        return 0.0
    gain = (post_score - pre_score) / (1.0 - pre_score)
    return gain


def category_coverage(
    successful_categories: Optional[set[Any]] = None,
    total_categories: Optional[set[Any]] = None,
    *,
    # Positional-friendly aliases (used by internal callers)
    achieved: Optional[set[Any]] = None,
    total: Optional[set[Any]] = None,
    # Deprecated aliases
    mastered_skills: Optional[set[Any]] = None,
    total_skills: Optional[set[Any]] = None,
) -> float:
    """Compute fraction of total categories achieved.

    Measures the breadth of a user's competence across a category domain.

    Parameters
    ----------
    successful_categories : set
        Set of category identifiers the user has succeeded in.
    total_categories : set
        Set of all categories in the domain.
    achieved : set, optional
        Alias for ``successful_categories`` (backward compatibility).
    total : set, optional
        Alias for ``total_categories`` (backward compatibility).
    mastered_skills : set, optional
        Deprecated alias for ``successful_categories``.
    total_skills : set, optional
        Deprecated alias for ``total_categories``.

    Returns
    -------
    float
        Fraction of categories achieved, in [0, 1]. Returns 0.0 if total is empty.
    """
    import warnings as _w

    # Resolve deprecated params
    if mastered_skills is not None:
        _w.warn(
            "Parameter 'mastered_skills' is deprecated, use 'successful_categories' instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        if successful_categories is None:
            successful_categories = mastered_skills
    if total_skills is not None:
        _w.warn(
            "Parameter 'total_skills' is deprecated, use 'total_categories' instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        if total_categories is None:
            total_categories = total_skills
    # Support old positional 'achieved' / 'total' aliases
    if successful_categories is None and achieved is not None:
        successful_categories = achieved
    if total_categories is None and total is not None:
        total_categories = total
    if successful_categories is None:
        successful_categories = set()
    if total_categories is None:
        total_categories = set()
    total_categories = set(total_categories)
    if not total_categories:
        return 0.0
    successful_categories = set(successful_categories)
    return float(len(successful_categories & total_categories)) / float(len(total_categories))


def proficiency_coverage(
    achieved: Optional[set[Any]] = None,
    total: Optional[set[Any]] = None,
    *,
    mastered_skills: Optional[set[Any]] = None,
    total_skills: Optional[set[Any]] = None,
) -> float:
    """Compute fraction of total competencies achieved.

    .. deprecated::
        Use :func:`category_coverage` instead. Will be removed in v1.0.

    Parameters
    ----------
    achieved : set
        Set of competency identifiers the user has demonstrated competence in.
    total : set
        Set of all competencies in the domain.
    mastered_skills : set, optional
        Deprecated alias for ``achieved`` (backward compatibility).
    total_skills : set, optional
        Deprecated alias for ``total`` (backward compatibility).

    Returns
    -------
    float
        Fraction of competencies achieved, in [0, 1]. Returns 0.0 if total is empty.
    """
    import warnings as _w
    _w.warn(
        "proficiency_coverage is deprecated, use category_coverage instead. "
        "Will be removed in v1.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return category_coverage(
        achieved=achieved,
        total=total,
        mastered_skills=mastered_skills,
        total_skills=total_skills,
    )


def sequence_adherence(
    recommended_items: Sequence[int],
    prerequisite_graph: Dict[int, set],
    succeeded: Optional[set] = None,
    *,
    # Deprecated alias
    mastered: Optional[set] = None,
) -> float:
    """Compute fraction of recommendations whose dependencies are met.

    Ensures that the recommender respects the logical ordering by only
    recommending items when their prerequisites have been completed.

    Parameters
    ----------
    recommended_items : Sequence[int]
        Recommended item IDs in order.
    prerequisite_graph : dict
        Mapping from item ID to set of prerequisite item IDs.
        Example: {3: {1, 2}} means item 3 requires items 1 and 2.
    succeeded : set
        Set of item IDs (prerequisites) already completed.
    mastered : set, optional
        Deprecated alias for ``succeeded``. Will be removed in v1.0.

    Returns
    -------
    float
        Fraction of recommendations with all dependencies met, in [0, 1].
        Returns 1.0 if recommended_items is empty.
    """
    if mastered is not None:
        import warnings
        warnings.warn(
            "Parameter 'mastered' is deprecated, use 'succeeded' instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        if succeeded is None:
            succeeded = mastered
    if succeeded is None:
        succeeded = set()
    if not recommended_items:
        return 1.0
    succeeded = set(succeeded)
    adherent = 0
    for item_id in recommended_items:
        prereqs = prerequisite_graph.get(item_id, set())
        if prereqs.issubset(succeeded):
            adherent += 1
    return float(adherent) / float(len(recommended_items))


def stretch_fit(
    recommended_difficulties: Sequence[float],
    user_competence: Optional[float] = None,
    stretch_width: float = 0.25,
    *,
    # Deprecated aliases
    student_ability: Optional[float] = None,
    zpd_width: Optional[float] = None,
) -> float:
    """Compute fraction of recommendations within the user's stretch zone.

    Recommendations should be slightly above current competence but not so
    far as to be discouraging.  Items are in the stretch zone if their
    difficulty is in ``[user_competence, user_competence + stretch_width]``.

    Parameters
    ----------
    recommended_difficulties : Sequence[float]
        Difficulty scores of recommended items, typically in [0, 1].
    user_competence : float
        Current competence level of the user, typically in [0, 1].
    stretch_width : float, optional
        Width of the stretch zone (default: 0.25).
    student_ability : float, optional
        Deprecated alias for ``user_competence``. Will be removed in v1.0.
    zpd_width : float, optional
        Deprecated alias for ``stretch_width``. Will be removed in v1.0.

    Returns
    -------
    float
        Fraction of items within the stretch zone, in [0, 1].
        Returns 1.0 if recommended_difficulties is empty.
    """
    import warnings as _w

    if student_ability is not None:
        _w.warn(
            "Parameter 'student_ability' is deprecated, use 'user_competence' instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        if user_competence is None:
            user_competence = student_ability
    if user_competence is None:
        raise TypeError("user_competence is required")
    if zpd_width is not None:
        _w.warn(
            "Parameter 'zpd_width' is deprecated, use 'stretch_width' instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        stretch_width = zpd_width

    if not recommended_difficulties:
        return 1.0
    user_competence = float(user_competence)
    stretch_width = float(stretch_width)
    zone_min = user_competence
    zone_max = user_competence + stretch_width
    in_zone = sum(
        1 for diff in recommended_difficulties
        if zone_min <= float(diff) <= zone_max
    )
    return float(in_zone) / float(len(recommended_difficulties))


def difficulty_appropriateness(
    recommended_difficulties: Sequence[float],
    student_ability: Optional[float] = None,
    zpd_width: float = 0.25,
    *,
    user_competence: Optional[float] = None,
    stretch_width: Optional[float] = None,
) -> float:
    """Compute fraction of recommendations within the user's stretch zone.

    .. deprecated::
        Use :func:`stretch_fit` instead. Will be removed in v1.0.

    Parameters
    ----------
    recommended_difficulties : Sequence[float]
        Difficulty scores of recommended items, typically in [0, 1].
    student_ability : float, optional
        Deprecated alias for ``user_competence``.
    zpd_width : float, optional
        Deprecated alias for ``stretch_width``.
    user_competence : float, optional
        Current competence level.
    stretch_width : float, optional
        Width of the stretch zone.

    Returns
    -------
    float
        Fraction of items within the stretch zone, in [0, 1].
        Returns 1.0 if recommended_difficulties is empty.
    """
    import warnings as _w
    _w.warn(
        "difficulty_appropriateness is deprecated, use stretch_fit instead. "
        "Will be removed in v1.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Merge old positional into keyword form for stretch_fit
    uc = user_competence if user_competence is not None else student_ability
    sw = stretch_width if stretch_width is not None else zpd_width
    return stretch_fit(
        recommended_difficulties,
        user_competence=uc,
        stretch_width=sw if sw is not None else 0.25,
    )


def engagement_score(
    interacted_items: Optional[Sequence[Any]] = None,
    total_recommended: Optional[int] = None,
    *,
    interactions: Optional[Sequence[Any]] = None,
    total_available: Optional[int] = None,
) -> float:
    """Compute ratio of items interacted with to items recommended.

    Measures user engagement as a fraction of total recommendations.

    Parameters
    ----------
    interacted_items : Sequence
        Items the user actually interacted with (clicked, answered, etc.).
    total_recommended : int
        Total number of items that were recommended to the user.
    interactions : Sequence
        Deprecated alias for interacted_items (backward compatibility).
    total_available : int
        Deprecated alias for total_recommended (backward compatibility).

    Returns
    -------
    float
        Fraction of recommended items that received interaction, in [0, 1].
        Returns 0.0 if total_recommended <= 0.
    """
    # Support old parameter names for backward compatibility
    if interacted_items is None and interactions is not None:
        interacted_items = interactions
    if total_recommended is None and total_available is not None:
        total_recommended = total_available
    if interacted_items is None:
        interacted_items = []
    if total_recommended is None:
        total_recommended = 0

    if total_recommended <= 0:
        return 0.0
    items = set(interacted_items) if not isinstance(interacted_items, set) else interacted_items
    return float(len(items)) / float(total_recommended)


@dataclass
class ProgressionReport:
    """Container for progression evaluation metrics.

    Aggregates metrics for assessing progression effectiveness, sequencing
    quality, and user engagement across any adaptive domain.

    Attributes
    ----------
    progression_gain : float
        Normalized gain: (post - pre) / (1 - pre). Measures improvement
        from pre-assessment to post-assessment.
    category_coverage : float
        Fraction of total categories achieved. Measures breadth of competence.
    adherence : float
        Fraction of recommendations with satisfied dependencies. Measures
        sequencing validity of recommendations.
    stretch_fit : float
        Fraction of recommendations within user's stretch zone.
        Measures appropriateness of difficulty level.
    engagement : float
        Ratio of items interacted with to items recommended. Measures user
        engagement and participation.
    """

    progression_gain: float
    category_coverage: float
    adherence: float
    stretch_fit: float
    engagement: float

    # Backward-compatible aliases for old field names
    @property
    def coverage(self) -> float:
        """Deprecated alias for ``category_coverage``."""
        return self.category_coverage

    @property
    def difficulty_fit(self) -> float:
        """Deprecated alias for ``stretch_fit``."""
        return self.stretch_fit


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "average_precision",
    "expected_calibration_error",
    "RankingReport",
    "evaluate_recommendations",
    # Primary names (domain-neutral)
    "progression_gain",
    "category_coverage",
    "stretch_fit",
    "sequence_adherence",
    "engagement_score",
    "ProgressionReport",
    # Deprecated aliases (still importable, emit warnings)
    "proficiency_coverage",
    "difficulty_appropriateness",
    "learning_gain",
    "knowledge_coverage",
    "curriculum_adherence",
    "EducationalReport",
]


# --- Deprecation handling for renamed symbols (PEP 562) ---
_DEPRECATED_NAMES = {
    "learning_gain": "progression_gain",
    "knowledge_coverage": "proficiency_coverage",
    "curriculum_adherence": "sequence_adherence",
    "EducationalReport": "ProgressionReport",
}


def __getattr__(name: str):
    if name in _DEPRECATED_NAMES:
        import warnings
        warnings.warn(
            f"{name} is deprecated, use {_DEPRECATED_NAMES[name]} instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[_DEPRECATED_NAMES[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
