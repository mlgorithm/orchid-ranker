"""Taste-progression module for expertise-driven product domains.

This module applies to product domains where users develop expertise and
taste over time.  It does NOT apply to commodity e-commerce where every
purchase is transactional (paper towels, batteries, USB cables).

Domains with genuine progression trajectories:

* **Wine**: table wine → regional varietals → reserve bottles → rare vintages
* **Photography**: kit lens → prime lens → L-series → specialised macro/tilt-shift
* **Coffee**: drip → pour-over → espresso → single-origin → competition-grade
* **Fashion**: fast fashion → contemporary → designer → bespoke
* **Cooking**: basic knife set → Japanese knives → Damascus steel → custom forged

In these domains the user develops *taste* and *expertise* over time.
Orchid's BKT (outcome tracing) can model this evolution by reinterpreting:

* **"correct" → kept** (purchased AND not returned, or purchased AND rated ≥ 4)
* **"category" → product category** (wine, photography, running shoes)
* **"difficulty" → sophistication** (price tier, expert rating, feature complexity)
* **"stretch zone" → next tier** (recommend a step up but not two tiers ahead)

This module provides:

* :class:`SophisticationMapper` -- maps items to a 0–1 sophistication score.
* :class:`TasteProfile` -- per-user, per-category taste evolution via BKT.
* :class:`TasteProgressionRanker` -- wraps an Orchid recommender and
  re-ranks by taste-trajectory fit.

Usage::

    from orchid_ranker.taste_progression import TasteProgressionRanker, TasteConfig

    ranker = TasteProgressionRanker(
        recommender=rec,
        sophistication_scores={42: 0.3, 137: 0.7, 99: 0.9},
        config=TasteConfig(stretch_width=0.15),
    )

    # Observe a purchase outcome
    ranker.observe(user_id=7, item_id=42, purchased=True, returned=False,
                   category="wine", rating=4.5)

    # Recommend items matching user's taste trajectory
    recs = ranker.recommend(user_id=7, top_k=10,
                            candidate_item_ids=[42, 137, 99, 200])
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from orchid_ranker.knowledge_tracing import BayesianKnowledgeTracing

logger = logging.getLogger(__name__)

__all__ = [
    "TasteConfig",
    "SophisticationMapper",
    "TasteProfile",
    "TasteProgressionRanker",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TasteConfig:
    """Configuration for e-commerce taste-progression.

    Parameters
    ----------
    stretch_width : float
        Half-width of the "stretch zone" around the user's taste level.
        Items within ``[taste - stretch_width, taste + stretch_width]``
        score highest.  Default 0.15 (15% of the 0–1 sophistication scale).
    keep_threshold : float
        Rating threshold above which a purchase counts as "kept" (positive
        outcome).  Default 4.0 on a 1–5 scale.
    return_penalty : float
        Outcome value assigned to returned items.  0.0 = fully negative
        signal.  Default 0.0.
    min_rating_for_positive : float
        Minimum rating to count as a positive signal when rating is
        available.  Default 3.5.
    bkt_p_init : float
        Initial taste-level prior.  Lower values mean the system starts
        conservative (recommends simpler products first).  Default 0.2.
    bkt_p_transit : float
        Per-interaction probability of taste advancement.  Default 0.08
        (slower than educational BKT because taste evolves gradually).
    category_weight : float
        Weight of category-specific taste vs global taste.  1.0 = fully
        per-category; 0.0 = global only.  Default 0.7.
    """

    stretch_width: float = 0.15
    keep_threshold: float = 4.0
    return_penalty: float = 0.0
    min_rating_for_positive: float = 3.5
    bkt_p_init: float = 0.2
    bkt_p_transit: float = 0.08
    category_weight: float = 0.7


# ---------------------------------------------------------------------------
# Sophistication scoring
# ---------------------------------------------------------------------------
class SophisticationMapper:
    """Maps items to a 0–1 sophistication score.

    Sophistication can be derived from price tier, expert rating, feature
    complexity, brand positioning, or any domain-specific signal.

    Supports three modes:

    1. **Explicit scores** -- pass a ``{item_id: score}`` dictionary.
    2. **Price-based** -- derive from price using quantile normalisation.
    3. **Feature-based** -- derive from a weighted combination of features.

    Parameters
    ----------
    scores : dict of int to float, optional
        Pre-computed sophistication scores in ``[0, 1]``.
    default_score : float
        Score for items not in the dictionary.  Default 0.5 (mid-tier).
    """

    def __init__(
        self,
        scores: Optional[Dict[int, float]] = None,
        *,
        default_score: float = 0.5,
    ) -> None:
        self._scores: Dict[int, float] = {}
        if scores:
            for k, v in scores.items():
                self._scores[int(k)] = float(max(0.0, min(1.0, v)))
        self._default = float(default_score)

    @classmethod
    def from_prices(
        cls,
        prices: Dict[int, float],
        *,
        default_score: float = 0.5,
    ) -> "SophisticationMapper":
        """Create from item prices using quantile normalisation.

        Items are ranked by price and mapped to ``[0, 1]`` by their
        percentile rank.  The cheapest item gets 0.0, the most expensive
        gets 1.0.

        Parameters
        ----------
        prices : dict of int to float
            Mapping from item ID to price.
        default_score : float
            Score for items not in the price dictionary.
        """
        if not prices:
            return cls(default_score=default_score)

        sorted_items = sorted(prices.items(), key=lambda x: x[1])
        n = len(sorted_items)
        scores = {}
        for rank, (item_id, _price) in enumerate(sorted_items):
            scores[item_id] = rank / max(n - 1, 1)
        return cls(scores, default_score=default_score)

    def __getitem__(self, item_id: int) -> float:
        return self._scores.get(int(item_id), self._default)

    def __contains__(self, item_id: int) -> bool:
        return int(item_id) in self._scores

    def score_batch(self, item_ids: Sequence[int]) -> List[float]:
        """Return sophistication scores for a batch of items."""
        return [self[iid] for iid in item_ids]


# ---------------------------------------------------------------------------
# Per-user taste evolution
# ---------------------------------------------------------------------------
class TasteProfile:
    """Tracks a user's evolving taste level across product categories.

    Uses BKT (outcome tracing) to model how a user's taste sophistication
    develops over time.  A "correct" observation means the user purchased
    AND kept an item at a given sophistication level (demonstrating
    readiness for that tier).  An "incorrect" observation means they
    returned it or rated it poorly (not ready for that tier).

    Parameters
    ----------
    config : TasteConfig
        Taste progression configuration.
    """

    def __init__(self, config: Optional[TasteConfig] = None) -> None:
        self._cfg = config or TasteConfig()
        # category -> BKT tracker
        self._trackers: Dict[str, BayesianKnowledgeTracing] = {}
        # global tracker (across all categories)
        self._global_tracker = BayesianKnowledgeTracing(
            p_init=self._cfg.bkt_p_init,
            p_transit=self._cfg.bkt_p_transit,
        )
        self._interaction_count = 0
        # EMA tracking of consumed item sophistication (per-category and global).
        # BKT p_known is non-linear and doesn't map directly to a linear
        # sophistication scale.  The EMA tracks the *actual* sophistication
        # level the user is consuming, providing a more stable signal for
        # stretch-zone computation.
        self._ema_alpha = 0.15  # EMA smoothing factor
        self._cat_ema: Dict[str, float] = {}   # category -> EMA sophistication
        self._global_ema: float = 0.0
        self._global_ema_init: bool = False

    def _get_tracker(self, category: str) -> BayesianKnowledgeTracing:
        if category not in self._trackers:
            self._trackers[category] = BayesianKnowledgeTracing(
                p_init=self._cfg.bkt_p_init,
                p_transit=self._cfg.bkt_p_transit,
            )
        return self._trackers[category]

    def observe(
        self,
        category: str,
        positive: bool,
        sophistication: Optional[float] = None,
    ) -> float:
        """Record one purchase outcome and return updated taste level.

        Parameters
        ----------
        category : str
            Product category (e.g. "wine", "cameras").
        positive : bool
            True if the user kept the item and was satisfied.
        sophistication : float, optional
            Sophistication score of the consumed item (0–1).  When provided,
            updates an EMA of consumed sophistication that supplements BKT
            for more accurate stretch-zone computation.

        Returns
        -------
        float
            Updated taste level for the category, in ``[0, 1]``.
        """
        cat_tracker = self._get_tracker(category)
        cat_tracker.update(positive)
        self._global_tracker.update(positive)
        self._interaction_count += 1

        # Update EMA of consumed sophistication when available
        if sophistication is not None:
            alpha = self._ema_alpha
            # Category EMA
            if category in self._cat_ema:
                self._cat_ema[category] = (
                    alpha * sophistication + (1.0 - alpha) * self._cat_ema[category]
                )
            else:
                self._cat_ema[category] = sophistication
            # Global EMA
            if self._global_ema_init:
                self._global_ema = (
                    alpha * sophistication + (1.0 - alpha) * self._global_ema
                )
            else:
                self._global_ema = sophistication
                self._global_ema_init = True

        return self.taste_level(category)

    def taste_level(self, category: Optional[str] = None) -> float:
        """Current taste sophistication level for a category.

        Blends BKT p_known with an EMA of consumed item sophistication
        when available.  The EMA provides a linear signal that tracks
        actual consumption level, while BKT captures the confidence
        that the user has "mastered" the current tier.

        Parameters
        ----------
        category : str, optional
            Product category.  If None, returns the global taste level.

        Returns
        -------
        float
            Taste level in ``[0, 1]``.  Higher means more sophisticated.
        """
        global_bkt = float(self._global_tracker.p_known)
        # Blend BKT with EMA when EMA is available (60% EMA, 40% BKT).
        # EMA tracks *where* the user is on the sophistication scale;
        # BKT tracks *confidence* they've mastered that level.
        if self._global_ema_init:
            global_level = 0.6 * self._global_ema + 0.4 * global_bkt
        else:
            global_level = global_bkt

        if category is None:
            return global_level

        cw = self._cfg.category_weight
        if category in self._trackers:
            cat_bkt = float(self._trackers[category].p_known)
            if category in self._cat_ema:
                cat_level = 0.6 * self._cat_ema[category] + 0.4 * cat_bkt
            else:
                cat_level = cat_bkt
            return cw * cat_level + (1.0 - cw) * global_level
        return global_level

    def stretch_zone(self, category: Optional[str] = None) -> Tuple[float, float]:
        """Return the ideal sophistication range for recommendations.

        Items within this range are a comfortable "step up" — not too
        basic (boring repeat) and not too advanced (sticker shock / buyer's
        remorse).

        Parameters
        ----------
        category : str, optional
            Product category.

        Returns
        -------
        (low, high) : tuple of float
            Lower and upper bounds of the stretch zone, clamped to ``[0, 1]``.
        """
        level = self.taste_level(category)
        w = self._cfg.stretch_width
        return (max(0.0, level - w), min(1.0, level + w))

    @property
    def categories(self) -> List[str]:
        """Categories the user has interacted with."""
        return list(self._trackers.keys())

    @property
    def interaction_count(self) -> int:
        return self._interaction_count


# ---------------------------------------------------------------------------
# Taste-progression ranker
# ---------------------------------------------------------------------------
class TasteProgressionRanker:
    """Wraps an Orchid recommender with e-commerce taste-trajectory scoring.

    Combines Orchid's collaborative filtering scores with:

    1. **Stretch-zone fit** -- how well the item's sophistication matches
       the user's current taste level.
    2. **Taste momentum** -- bonus for items slightly above the user's level
       (encouraging discovery), penalty for items far above.
    3. **Category exploration** -- mild diversity bonus for categories the
       user hasn't explored yet.

    Parameters
    ----------
    recommender : object, optional
        A fitted :class:`OrchidRecommender` (or any object with a
        ``recommend(user_id, top_k, candidate_item_ids)`` method).
        If None, only taste-progression scoring is used (no collaborative
        filtering component).
    sophistication_scores : dict of int to float, optional
        Pre-computed ``{item_id: sophistication}`` scores.  Alternatively
        pass a :class:`SophisticationMapper`.
    config : TasteConfig, optional
        Configuration.
    w_relevance : float
        Weight for collaborative filtering score.  Default 0.4.
    w_stretch : float
        Weight for stretch-zone fit.  Default 0.35.
    w_momentum : float
        Weight for upward-taste-momentum bonus.  Default 0.15.
    w_exploration : float
        Weight for category exploration bonus.  Default 0.10.
    """

    def __init__(
        self,
        recommender: Optional[object] = None,
        sophistication_scores: Optional[Dict[int, float] | SophisticationMapper] = None,
        *,
        config: Optional[TasteConfig] = None,
        w_relevance: float = 0.4,
        w_stretch: float = 0.35,
        w_momentum: float = 0.15,
        w_exploration: float = 0.10,
    ) -> None:
        self._rec = recommender
        self._cfg = config or TasteConfig()

        if isinstance(sophistication_scores, SophisticationMapper):
            self._soph = sophistication_scores
        else:
            self._soph = SophisticationMapper(sophistication_scores or {})

        self._w_relevance = float(w_relevance)
        self._w_stretch = float(w_stretch)
        self._w_momentum = float(w_momentum)
        self._w_exploration = float(w_exploration)

        # Per-user taste profiles
        self._profiles: Dict[int, TasteProfile] = {}

        # Item-to-category mapping (populated via observe or set_item_categories)
        self._item_categories: Dict[int, str] = {}

    def set_item_categories(self, mapping: Dict[int, str]) -> None:
        """Set the item-to-category mapping.

        Parameters
        ----------
        mapping : dict of int to str
            Maps item ID to its product category.
        """
        self._item_categories = {int(k): str(v) for k, v in mapping.items()}

    def _get_profile(self, user_id: int) -> TasteProfile:
        uid = int(user_id)
        if uid not in self._profiles:
            self._profiles[uid] = TasteProfile(self._cfg)
        return self._profiles[uid]

    def observe(
        self,
        user_id: int,
        item_id: int,
        *,
        purchased: bool = True,
        returned: bool = False,
        rating: Optional[float] = None,
        category: Optional[str] = None,
    ) -> Dict[str, float]:
        """Record a purchase outcome.

        Parameters
        ----------
        user_id : int
            User identifier.
        item_id : int
            Item identifier.
        purchased : bool
            Whether the user purchased the item.
        returned : bool
            Whether the user returned the item.
        rating : float, optional
            User's rating for the item (e.g. 1–5 scale).
        category : str, optional
            Product category.  If None, looked up from ``_item_categories``
            or defaults to ``"__general__"``.

        Returns
        -------
        dict
            ``{"taste_level": float, "category": str, "positive": bool}``
        """
        cat = category or self._item_categories.get(int(item_id), "__general__")
        if category is not None:
            self._item_categories[int(item_id)] = cat

        # Determine if this is a positive signal
        if returned:
            positive = False
        elif not purchased:
            positive = False
        elif rating is not None:
            positive = rating >= self._cfg.min_rating_for_positive
        else:
            positive = True  # purchased and not returned, no rating info

        # Look up the item's sophistication score so the profile EMA
        # tracks actual consumed sophistication level.
        soph: Optional[float] = None
        iid = int(item_id)
        if iid in self._soph:
            soph = self._soph[iid]

        profile = self._get_profile(user_id)
        taste_level = profile.observe(cat, positive, sophistication=soph)

        return {
            "taste_level": taste_level,
            "category": cat,
            "positive": positive,
        }

    def taste_level(
        self, user_id: int, category: Optional[str] = None,
    ) -> float:
        """Current taste level for a user (optionally per-category)."""
        return self._get_profile(user_id).taste_level(category)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        *,
        candidate_item_ids: Optional[Sequence[int]] = None,
        category: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """Recommend items scored by taste-progression fit.

        Parameters
        ----------
        user_id : int
            User identifier.
        top_k : int
            Number of items to return.
        candidate_item_ids : sequence of int, optional
            Candidate items.  Required if no recommender is provided.
        category : str, optional
            Filter to a specific product category.

        Returns
        -------
        list of (item_id, score)
            Sorted by blended score descending.
        """
        if candidate_item_ids is None and self._rec is None:
            raise ValueError(
                "candidate_item_ids is required when no recommender is set"
            )

        candidates: List[int]
        if candidate_item_ids is not None:
            candidates = [int(c) for c in candidate_item_ids]
        else:
            # Use a large candidate set from the recommender
            candidates = list(range(1000))  # placeholder

        if category is not None:
            # Filter to items in the requested category
            candidates = [
                c for c in candidates
                if self._item_categories.get(c, "__general__") == category
                or c not in self._item_categories  # include uncategorised
            ]

        if not candidates:
            return []

        profile = self._get_profile(user_id)

        # 1. Relevance scores (from Orchid recommender)
        relevance = self._relevance_scores(user_id, candidates, top_k)

        # 2. Stretch-zone fit scores
        stretch = self._stretch_scores(profile, candidates, category)

        # 3. Momentum scores (bonus for slight step-up)
        momentum = self._momentum_scores(profile, candidates, category)

        # 4. Exploration scores (bonus for new categories)
        exploration = self._exploration_scores(profile, candidates)

        # Blend
        blended = (
            self._w_relevance * relevance
            + self._w_stretch * stretch
            + self._w_momentum * momentum
            + self._w_exploration * exploration
        )

        # Top-k

        k = min(top_k, len(candidates))
        top_idx = np.argpartition(blended, -k)[-k:]
        top_idx = top_idx[np.argsort(blended[top_idx])[::-1]]

        return [(candidates[i], float(blended[i])) for i in top_idx]

    # --- scoring components ---

    def _relevance_scores(
        self, user_id: int, candidates: List[int], top_k: int,
    ) -> "np.ndarray":

        scores = np.zeros(len(candidates), dtype=np.float32)

        if self._rec is None:
            return scores

        try:
            recs = self._rec.recommend(
                user_id=user_id,
                top_k=len(candidates),
                candidate_item_ids=candidates,
            )
            cand_idx = {c: i for i, c in enumerate(candidates)}
            for rec in recs:
                iid = rec.item_id if hasattr(rec, "item_id") else rec[0]
                sc = rec.score if hasattr(rec, "score") else rec[1]
                if iid in cand_idx:
                    scores[cand_idx[iid]] = float(sc)
            # Normalise to [0, 1]
            smin, smax = scores.min(), scores.max()
            if smax > smin:
                scores = (scores - smin) / (smax - smin)
        except (KeyError, RuntimeError):
            pass

        return scores

    def _stretch_scores(
        self,
        profile: TasteProfile,
        candidates: List[int],
        category: Optional[str],
    ) -> "np.ndarray":
        """Gaussian-shaped score centred on the user's taste level."""

        level = profile.taste_level(category)
        w = max(self._cfg.stretch_width, 0.01)
        sigma = w  # stretch_width maps to 1 sigma

        scores = np.zeros(len(candidates), dtype=np.float32)
        for i, cid in enumerate(candidates):
            soph = self._soph[cid]
            diff = abs(soph - level)
            # Gaussian: max=1.0 at diff=0, drops off with sigma
            scores[i] = math.exp(-0.5 * (diff / sigma) ** 2)

        return scores

    def _momentum_scores(
        self,
        profile: TasteProfile,
        candidates: List[int],
        category: Optional[str],
    ) -> "np.ndarray":
        """Bonus for items slightly above user's taste level (upward discovery)."""

        level = profile.taste_level(category)
        w = self._cfg.stretch_width

        scores = np.zeros(len(candidates), dtype=np.float32)
        for i, cid in enumerate(candidates):
            soph = self._soph[cid]
            delta = soph - level  # positive = more sophisticated

            if 0 < delta <= w:
                # Sweet spot: slightly above — maximum momentum bonus
                scores[i] = 1.0 - (delta / w) * 0.3  # 1.0 → 0.7 as delta → w
            elif delta > w:
                # Too far above — rapid decay
                scores[i] = max(0.0, 0.7 * math.exp(-(delta - w) / w))
            elif -w <= delta <= 0:
                # Slightly below — mild score (repeat territory)
                scores[i] = 0.3 + 0.3 * (1.0 + delta / w)  # 0.3 → 0.6
            else:
                # Far below — minimal momentum
                scores[i] = 0.1

        return scores

    def _exploration_scores(
        self,
        profile: TasteProfile,
        candidates: List[int],
    ) -> "np.ndarray":
        """Bonus for items in categories the user hasn't explored."""

        explored = set(profile.categories)
        scores = np.zeros(len(candidates), dtype=np.float32)
        for i, cid in enumerate(candidates):
            cat = self._item_categories.get(cid, "__general__")
            if cat not in explored and cat != "__general__":
                scores[i] = 1.0  # full bonus for unexplored
            elif cat in explored:
                scores[i] = 0.3  # mild bonus for familiar categories
            else:
                scores[i] = 0.5  # neutral for uncategorised

        return scores

    def __repr__(self) -> str:
        n_users = len(self._profiles)
        n_items = len(self._item_categories)
        return (
            f"TasteProgressionRanker(users={n_users}, "
            f"categorised_items={n_items}, "
            f"weights=[rel={self._w_relevance}, stretch={self._w_stretch}, "
            f"mom={self._w_momentum}, expl={self._w_exploration}])"
        )
