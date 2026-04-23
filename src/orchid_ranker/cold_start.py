"""Cold-start handling for new users and new items.

Orchid Ranker needs interaction history to model user trajectories. This
module bridges the gap for cold-start scenarios:

* :class:`ItemFeatureIndex` -- precomputed content-based item similarity
  from feature vectors, enabling "users who have no history get items
  similar to a seed set."
* :class:`PopularityPrior` -- global and per-segment popularity fallback.
* :class:`ColdStartBridge` -- automatic routing that blends cold-start
  scores with Orchid scores based on how many interactions a user has.
  New users get popularity + content; experienced users get pure Orchid;
  users in between get a smooth linear blend.

Usage::

    from orchid_ranker.cold_start import ColdStartBridge, ColdStartConfig

    bridge = ColdStartBridge(
        recommender=rec,
        item_features=item_feature_matrix,
        config=ColdStartConfig(min_interactions=5, blend_until=20),
    )

    # Works for brand-new users (0 interactions) AND experienced users
    recs = bridge.recommend(user_id=new_user_id, top_k=10)

    # As the user interacts, the bridge automatically transitions
    bridge.observe(user_id=new_user_id, item_id=42, outcome=1.0)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ColdStartConfig",
    "ItemFeatureIndex",
    "PopularityPrior",
    "ColdStartBridge",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ColdStartConfig:
    """Configuration for cold-start handling.

    Parameters
    ----------
    min_interactions : int
        Below this count, the user is fully cold-start (Orchid weight = 0).
    blend_until : int
        Between ``min_interactions`` and this count, Orchid is blended in
        linearly.  At ``blend_until`` interactions, the user is fully warm
        (Orchid weight = 1.0).
    popularity_weight : float
        Weight of popularity score in the cold-start blend. The content
        similarity score gets ``1 - popularity_weight``.
    diversity_penalty : float
        MMR-style penalty discounting items similar to those already in
        the slate.  0.0 disables diversity; 1.0 is maximum penalty.
    seed_item_count : int
        Number of "seed" items used to build a content profile for a new
        user from stated preferences.  If the user has a few interactions
        already, the most recent ``seed_item_count`` interactions are used.
    """

    min_interactions: int = 3
    blend_until: int = 20
    popularity_weight: float = 0.3
    diversity_penalty: float = 0.2
    seed_item_count: int = 5


# ---------------------------------------------------------------------------
# Content-based item similarity
# ---------------------------------------------------------------------------
class ItemFeatureIndex:
    """Precomputed content-based item similarity from feature vectors.

    At construction time the feature matrix is L2-normalised per item,
    enabling fast cosine-similarity lookups via matrix multiplication.

    Parameters
    ----------
    item_features : np.ndarray
        Shape ``(num_items, feature_dim)``.  Each row is a feature vector
        for one item.  Items are indexed ``0 … num_items-1``.
    metric : str
        Similarity metric.  Currently only ``"cosine"`` is supported.

    Examples
    --------
    >>> idx = ItemFeatureIndex(features)
    >>> similar = idx.similar_items(item_id=42, top_k=5)
    >>> # [(17, 0.93), (88, 0.91), ...]
    """

    def __init__(
        self,
        item_features: np.ndarray,
        *,
        metric: str = "cosine",
    ) -> None:
        if metric != "cosine":
            raise ValueError(f"Unsupported metric '{metric}'; only 'cosine' is supported.")
        feats = np.asarray(item_features, dtype=np.float32)
        if feats.ndim != 2:
            raise ValueError(f"item_features must be 2-D, got shape {feats.shape}")

        # L2-normalise each row so dot product = cosine similarity
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # avoid division by zero
        self._normed: np.ndarray = feats / norms
        self._num_items = int(feats.shape[0])
        self._feature_dim = int(feats.shape[1])

    @property
    def num_items(self) -> int:
        """Number of items in the index."""
        return self._num_items

    def similarity(self, item_a: int, item_b: int) -> float:
        """Cosine similarity between two items."""
        return float(self._normed[item_a] @ self._normed[item_b])

    def similar_items(
        self,
        item_id: int,
        top_k: int = 10,
        *,
        exclude: Optional[set] = None,
    ) -> List[Tuple[int, float]]:
        """Return the ``top_k`` most similar items to ``item_id``.

        Parameters
        ----------
        item_id : int
            Query item.
        top_k : int
            Number of similar items to return.
        exclude : set of int, optional
            Item IDs to exclude from results (e.g. items the user already
            interacted with, or the query item itself).

        Returns
        -------
        list of (item_id, similarity)
            Sorted by similarity descending.
        """
        sims = self._normed @ self._normed[item_id]  # (num_items,)
        sims[item_id] = -1.0  # exclude self
        if exclude:
            for eid in exclude:
                if 0 <= eid < self._num_items:
                    sims[eid] = -1.0
        # argpartition is O(n) vs O(n log n) for argsort
        k = min(top_k, self._num_items - 1)
        if k <= 0:
            return []
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(int(i), float(sims[i])) for i in top_idx if sims[i] > -1.0]

    def user_profile_scores(
        self,
        seed_item_ids: Sequence[int],
        *,
        exclude: Optional[set] = None,
    ) -> np.ndarray:
        """Score all items against a user profile built from seed items.

        The user profile is the mean of the normalised feature vectors for
        the seed items.  Scores are cosine similarities in ``[-1, 1]``.

        Parameters
        ----------
        seed_item_ids : sequence of int
            Items the user has interacted with or stated as preferences.
        exclude : set of int, optional
            Items to mask out (returns -inf for these).

        Returns
        -------
        np.ndarray
            Shape ``(num_items,)`` similarity scores.
        """
        if not seed_item_ids:
            return np.zeros(self._num_items, dtype=np.float32)

        valid_ids = [i for i in seed_item_ids if 0 <= i < self._num_items]
        if not valid_ids:
            return np.zeros(self._num_items, dtype=np.float32)

        profile = np.mean(self._normed[valid_ids], axis=0)
        # re-normalise the profile
        pnorm = np.linalg.norm(profile)
        if pnorm > 1e-12:
            profile /= pnorm

        scores = self._normed @ profile  # (num_items,)
        # Guard against NaN/Inf from degenerate feature vectors
        np.nan_to_num(scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if exclude:
            for eid in exclude:
                if 0 <= eid < self._num_items:
                    scores[eid] = -np.inf
        return scores


# ---------------------------------------------------------------------------
# Popularity prior
# ---------------------------------------------------------------------------
class PopularityPrior:
    """Global and per-segment item popularity scores.

    Popularity is computed as normalised interaction frequency: the count
    of interactions per item divided by the maximum count across all items,
    yielding scores in ``[0, 1]``.

    Optionally supports *segment-level* popularity (e.g. popularity among
    users in the "enterprise" segment vs. "consumer" segment).

    Parameters
    ----------
    smoothing : float
        Laplace smoothing added to each item's count to prevent zero scores
        for items with no interactions.  Default is 1.0.
    """

    def __init__(self, *, smoothing: float = 1.0) -> None:
        self._smoothing = float(smoothing)
        self._global_counts: Dict[int, float] = defaultdict(float)
        self._segment_counts: Dict[str, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._fitted = False

    def fit(
        self,
        item_ids: Sequence[int],
        *,
        segments: Optional[Sequence[str]] = None,
    ) -> "PopularityPrior":
        """Fit popularity from a sequence of observed item interactions.

        Parameters
        ----------
        item_ids : sequence of int
            Flat list of item IDs from the interaction log (one entry per
            interaction, duplicates expected and counted).
        segments : sequence of str, optional
            If provided, must be same length as ``item_ids``.  Each entry
            is the segment label of the user who generated that interaction.

        Returns
        -------
        self
        """
        self._global_counts = defaultdict(float)
        self._segment_counts = defaultdict(lambda: defaultdict(float))

        for idx, iid in enumerate(item_ids):
            self._global_counts[int(iid)] += 1.0
            if segments is not None:
                seg = str(segments[idx])
                self._segment_counts[seg][int(iid)] += 1.0

        self._fitted = True
        logger.info(
            "PopularityPrior fitted: %d unique items, %d segments",
            len(self._global_counts),
            len(self._segment_counts),
        )
        return self

    def scores(
        self,
        candidate_ids: Sequence[int],
        *,
        segment: Optional[str] = None,
    ) -> np.ndarray:
        """Return normalised popularity scores for candidate items.

        Parameters
        ----------
        candidate_ids : sequence of int
            Items to score.
        segment : str, optional
            If provided, scores are computed from the segment-specific
            counts instead of global counts.

        Returns
        -------
        np.ndarray
            Shape ``(len(candidate_ids),)`` scores in ``[0, 1]``.
        """
        if segment is not None and segment in self._segment_counts:
            counts = self._segment_counts[segment]
        else:
            counts = self._global_counts

        raw = np.array(
            [counts.get(int(cid), 0.0) + self._smoothing for cid in candidate_ids],
            dtype=np.float32,
        )
        max_val = raw.max()
        if max_val > 0:
            raw /= max_val
        return raw

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Cold-start bridge
# ---------------------------------------------------------------------------
class ColdStartBridge:
    """Automatic routing between cold-start and Orchid recommendations.

    For users with zero or few interactions the bridge returns
    popularity + content-based recommendations.  As the user accumulates
    interactions, Orchid scores are blended in linearly until the user
    is fully warm.

    The blending formula for a user with *n* interactions:

    * ``n < min_interactions``: pure cold-start
      ``score = pop_weight * popularity + (1 - pop_weight) * content_sim``
    * ``min_interactions <= n < blend_until``: linear blend
      ``α = (n - min) / (blend_until - min)``
      ``score = α * orchid_score + (1 - α) * cold_start_score``
    * ``n >= blend_until``: pure Orchid
      ``score = orchid_score``

    Parameters
    ----------
    recommender : object
        A fitted :class:`OrchidRecommender` (or any object with a
        ``recommend(user_id, top_k, candidate_item_ids)`` method that
        returns a list of ``Recommendation(item_id, score)``).
    item_features : np.ndarray
        Shape ``(num_items, feature_dim)`` for content-based scoring.
    popularity_prior : PopularityPrior, optional
        Pre-fitted popularity model.  If ``None``, a uniform prior is used.
    config : ColdStartConfig, optional
        Configuration.  Uses defaults if not provided.
    """

    def __init__(
        self,
        recommender: object,
        item_features: np.ndarray,
        *,
        popularity_prior: Optional[PopularityPrior] = None,
        config: Optional[ColdStartConfig] = None,
    ) -> None:
        self._rec = recommender
        self._index = ItemFeatureIndex(item_features)
        self._pop = popularity_prior
        self._cfg = config or ColdStartConfig()

        # Per-user interaction tracking (lightweight — just counts + recent items)
        self._user_interactions: Dict[int, List[int]] = defaultdict(list)
        self._user_outcomes: Dict[int, List[float]] = defaultdict(list)

        if self._cfg.min_interactions > self._cfg.blend_until:
            raise ValueError(
                f"min_interactions ({self._cfg.min_interactions}) must be "
                f"<= blend_until ({self._cfg.blend_until})"
            )

    def observe(
        self,
        user_id: int,
        item_id: int,
        outcome: float = 1.0,
    ) -> None:
        """Record a user interaction for cold-start tracking.

        Parameters
        ----------
        user_id : int
            User identifier.
        item_id : int
            Item the user interacted with.
        outcome : float
            Interaction outcome in ``[0, 1]``.  1.0 = positive (correct,
            purchased, liked), 0.0 = negative.
        """
        self._user_interactions[int(user_id)].append(int(item_id))
        self._user_outcomes[int(user_id)].append(float(outcome))

    def interaction_count(self, user_id: int) -> int:
        """Number of recorded interactions for a user."""
        return len(self._user_interactions.get(int(user_id), []))

    def warmth(self, user_id: int) -> float:
        """Orchid blending weight for a user, in ``[0.0, 1.0]``.

        * 0.0 = fully cold (pure cold-start scores)
        * 1.0 = fully warm (pure Orchid scores)
        """
        n = self.interaction_count(user_id)
        if n < self._cfg.min_interactions:
            return 0.0
        if n >= self._cfg.blend_until:
            return 1.0
        span = max(self._cfg.blend_until - self._cfg.min_interactions, 1)
        return float((n - self._cfg.min_interactions) / span)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        *,
        candidate_item_ids: Optional[Sequence[int]] = None,
        segment: Optional[str] = None,
        seed_item_ids: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Recommend items, automatically routing based on interaction count.

        Parameters
        ----------
        user_id : int
            User identifier.
        top_k : int
            Number of items to return.
        candidate_item_ids : sequence of int, optional
            Restrict to these candidates.  Defaults to all items.
        segment : str, optional
            User segment for segment-level popularity scoring.
        seed_item_ids : sequence of int, optional
            Override seed items for content-based profile.  Defaults to
            the user's most recent interactions.

        Returns
        -------
        list of (item_id, score)
            Sorted by blended score descending.
        """
        alpha = self.warmth(user_id)

        if candidate_item_ids is None:
            candidates = list(range(self._index.num_items))
        else:
            candidates = [int(c) for c in candidate_item_ids]

        if not candidates:
            return []

        # --- Cold-start scores ---
        cold_scores = self._cold_start_scores(
            user_id, candidates, segment=segment, seed_item_ids=seed_item_ids,
        )

        # --- Orchid scores (if user is warm enough) ---
        if alpha > 0.0:
            orchid_scores = self._orchid_scores(user_id, candidates, top_k)
        else:
            orchid_scores = np.zeros(len(candidates), dtype=np.float32)

        # --- Blend ---
        blended = alpha * orchid_scores + (1.0 - alpha) * cold_scores
        np.nan_to_num(blended, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Top-k selection ---
        k = min(top_k, len(candidates))
        top_idx = np.argpartition(blended, -k)[-k:]
        top_idx = top_idx[np.argsort(blended[top_idx])[::-1]]

        return [(candidates[i], float(blended[i])) for i in top_idx]

    # --- internals ---

    def _cold_start_scores(
        self,
        user_id: int,
        candidates: List[int],
        *,
        segment: Optional[str] = None,
        seed_item_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Compute cold-start scores: popularity + content similarity."""
        pw = self._cfg.popularity_weight
        cw = 1.0 - pw

        # Popularity component
        if self._pop is not None and self._pop.is_fitted:
            pop_scores = self._pop.scores(candidates, segment=segment)
        else:
            pop_scores = np.ones(len(candidates), dtype=np.float32)

        # Content similarity component
        seeds = seed_item_ids
        if seeds is None:
            history = self._user_interactions.get(int(user_id), [])
            seeds = history[-self._cfg.seed_item_count:] if history else []

        if seeds:
            # Score all items, then pick candidates
            all_scores = self._index.user_profile_scores(seeds)
            content_scores = np.array(
                [all_scores[c] if 0 <= c < len(all_scores) else 0.0 for c in candidates],
                dtype=np.float32,
            )
            # Normalise content scores to [0, 1]
            cmin, cmax = content_scores.min(), content_scores.max()
            if cmax > cmin:
                content_scores = (content_scores - cmin) / (cmax - cmin)
            else:
                content_scores = np.zeros_like(content_scores)
        else:
            content_scores = np.zeros(len(candidates), dtype=np.float32)

        combined = pw * pop_scores + cw * content_scores
        # Guard against NaN/Inf from degenerate feature vectors
        np.nan_to_num(combined, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return combined

    def _orchid_scores(
        self,
        user_id: int,
        candidates: List[int],
        top_k: int,
    ) -> np.ndarray:
        """Get Orchid recommender scores for candidates.

        Supports two recommender APIs:

        1. ``recommend(user_id, top_k, candidate_item_ids=...)`` — custom
           recommenders that accept candidate filtering.
        2. ``recommend(user_id, top_k, filter_seen=True)`` —
           :class:`OrchidRecommender` standard API.

        Falls back gracefully if the user is not in training data.
        """
        # Use NaN to distinguish "model has no opinion" from "model
        # scored this item at zero".  After normalisation, items the
        # model didn't cover get the mean of covered items' scores
        # (neutral) rather than 0 (implicit penalty).
        scores = np.full(len(candidates), np.nan, dtype=np.float32)

        try:
            # Try the candidate_item_ids API first
            try:
                recs = self._rec.recommend(
                    user_id=user_id,
                    top_k=len(candidates),
                    candidate_item_ids=candidates,
                )
            except TypeError:
                # Fall back to the standard OrchidRecommender API
                recs = self._rec.recommend(
                    user_id=user_id,
                    top_k=len(candidates),
                )

            # Map recommended item scores back to candidate positions
            cand_to_idx = {c: i for i, c in enumerate(candidates)}
            for rec in recs:
                iid = rec.item_id if hasattr(rec, "item_id") else rec[0]
                sc = rec.score if hasattr(rec, "score") else rec[1]
                if iid in cand_to_idx:
                    scores[cand_to_idx[iid]] = float(sc)

            # Identify covered vs uncovered items
            covered = ~np.isnan(scores)
            if not np.any(covered):
                return np.zeros(len(candidates), dtype=np.float32)

            # Clean NaN/Inf in covered scores only
            covered_vals = scores[covered]
            np.nan_to_num(covered_vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            scores[covered] = covered_vals

            # Normalise covered scores to [0, 1].
            # Variance guard: if the score range is negligible (< 0.01),
            # the model isn't meaningfully discriminating — fall back.
            smin, smax = covered_vals.min(), covered_vals.max()
            if (smax - smin) >= 0.01:
                scores[covered] = (scores[covered] - smin) / (smax - smin)
                # Uncovered items get the mean of covered scores (neutral).
                mean_score = float(np.mean(scores[covered]))
                scores[~covered] = mean_score
            else:
                scores[:] = 0.0
        except (KeyError, RuntimeError) as exc:
            # User not in Orchid's training data — fall back gracefully
            logger.debug(
                "Orchid scoring failed for user %d (likely cold-start): %s",
                user_id, exc,
            )
            scores = np.zeros(len(candidates), dtype=np.float32)

        return scores

    def __repr__(self) -> str:
        n_users = len(self._user_interactions)
        return (
            f"ColdStartBridge(users_tracked={n_users}, "
            f"min_interactions={self._cfg.min_interactions}, "
            f"blend_until={self._cfg.blend_until})"
        )
