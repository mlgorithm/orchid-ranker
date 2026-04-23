"""Curated feed ranking with freshness, topic progression, and diversity.

For editorially curated publications where readers have a learning arc
through a body of content -- specialist newsletters (Stratechery, Matt
Levine), industry press, technical publications, course marketplaces,
podcast series with topical depth.

This module does NOT target engagement-driven social feeds (Meta, TikTok,
Reddit). Those are CTR-optimization problems where Orchid has no edge.

Standard recommenders rank by predicted relevance alone. For curated
content feeds, additional signals matter:

* **Freshness** -- recent content is more valuable.
* **Topic progression** -- users develop understanding of topics over time;
  recommend content at the right complexity level.
* **Diversity** -- avoid topic monocultures in the feed.

This module provides composable scorers and a :class:`FeedRanker` that
combines them with a base recommender's relevance scores.

Usage::

    from orchid_ranker.curated_feed import FeedRanker, FeedItem, FreshnessScorer

    ranker = FeedRanker(
        freshness=FreshnessScorer(halflife_hours=12),
        w_freshness=0.3,
    )
    ranked = ranker.rank(user_id=42, candidates=items, top_k=20)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from orchid_ranker.knowledge_tracing import CompetencyTracker

__all__ = [
    "FeedItem",
    "ScoredFeedItem",
    "FreshnessScorer",
    "TopicTracker",
    "ReadingLevelEstimator",
    "FeedRanker",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedItem:
    """A candidate content item for feed ranking.

    Parameters
    ----------
    item_id : int
        Unique identifier for the content item.
    topic : str
        Topic or category label (e.g. ``"AI basics"``, ``"economics"``).
    difficulty : float
        Reading difficulty on a 0--1 scale where 0.0 is easiest and 1.0
        is most advanced.
    timestamp : float
        Publication time as Unix epoch seconds.
    metadata : dict or None, optional
        Arbitrary extra features the caller may want to pass through
        (e.g. source, author, language). Not used by the ranker itself.
    """

    item_id: int
    topic: str
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    timestamp: float  # Unix epoch seconds (publication time)
    metadata: dict | None = None  # optional extra features


@dataclass
class ScoredFeedItem:
    """A ranked feed item with score breakdown.

    Attributes
    ----------
    item : FeedItem
        The underlying content item.
    total_score : float
        Weighted combination of all component scores.
    relevance_score : float
        Base recommender relevance component.
    freshness_score : float
        Time-decay freshness component.
    stretch_score : float
        How well the item's difficulty matches the user's stretch zone.
    diversity_score : float
        MMR-style topic diversity component.
    competence_score : float
        User's topic competence readiness component.
    """

    item: FeedItem
    total_score: float
    relevance_score: float
    freshness_score: float
    stretch_score: float
    diversity_score: float
    competence_score: float


# ---------------------------------------------------------------------------
# FreshnessScorer
# ---------------------------------------------------------------------------


class FreshnessScorer:
    """Time-decay scoring for content items.

    Computes a freshness multiplier in (0, 1] based on item age.
    Uses exponential decay: ``score = exp(-age * ln(2) / halflife)``.

    Parameters
    ----------
    halflife_hours : float
        Time in hours for the freshness score to decay to 0.5.
        Default 24.0 (daily news cycle).
    min_score : float
        Floor value -- even very old items get at least this score.
        Default 0.01.

    Raises
    ------
    ValueError
        If *halflife_hours* is not positive or *min_score* is not in [0, 1].

    Examples
    --------
    >>> scorer = FreshnessScorer(halflife_hours=24.0)
    >>> scorer.score(time.time())  # just published
    1.0
    >>> 0.49 < scorer.score(time.time() - 24 * 3600) < 0.51
    True
    """

    def __init__(
        self,
        halflife_hours: float = 24.0,
        min_score: float = 0.01,
    ) -> None:
        if halflife_hours <= 0:
            raise ValueError(
                f"halflife_hours must be positive, got {halflife_hours}"
            )
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {min_score}")

        self.halflife_hours = halflife_hours
        self.min_score = min_score
        # Pre-compute decay constant: lambda such that exp(-lambda * halflife) = 0.5
        self._decay = math.log(2) / (halflife_hours * 3600.0)

    def score(
        self,
        item_timestamp: float,
        now: float | None = None,
    ) -> float:
        """Freshness score for a single item.

        Parameters
        ----------
        item_timestamp : float
            Publication time as Unix epoch seconds.
        now : float or None, optional
            Current time as Unix epoch seconds. Defaults to
            ``time.time()`` when *None*.

        Returns
        -------
        float
            Freshness multiplier in [min_score, 1.0].
        """
        if now is None:
            now = time.time()
        age_seconds = max(0.0, now - item_timestamp)
        raw = math.exp(-self._decay * age_seconds)
        return max(self.min_score, raw)

    def scores_batch(
        self,
        item_timestamps: np.ndarray,
        now: float | None = None,
    ) -> np.ndarray:
        """Vectorized freshness scores.

        Parameters
        ----------
        item_timestamps : numpy.ndarray
            1-D array of publication times (Unix epoch seconds).
        now : float or None, optional
            Current time as Unix epoch seconds. Defaults to
            ``time.time()`` when *None*.

        Returns
        -------
        numpy.ndarray
            1-D array of freshness multipliers in [min_score, 1.0].
        """
        if now is None:
            now = time.time()
        ages = np.maximum(0.0, now - item_timestamps)
        raw = np.exp(-self._decay * ages)
        return np.maximum(self.min_score, raw)

    def __repr__(self) -> str:
        return (
            f"FreshnessScorer(halflife_hours={self.halflife_hours}, "
            f"min_score={self.min_score})"
        )


# ---------------------------------------------------------------------------
# TopicTracker
# ---------------------------------------------------------------------------


class TopicTracker:
    """Per-user topic competence tracker using outcome tracing (BKT).

    Tracks how well a user understands each topic.  "Correct" in this
    context means *engaged meaningfully* (read > 60 % of article, spent
    > median dwell time, etc.) -- the caller defines what "engaged"
    means.

    This extends BKT to content domains where "mastery" means
    "sophisticated understanding" rather than "can solve problems."

    Internally, each user gets a :class:`CompetencyTracker` from
    :mod:`orchid_ranker.knowledge_tracing`.

    Parameters
    ----------
    success_threshold : float
        Competence probability above which a topic is considered
        "mastered" for coverage calculations.  Default 0.8.
    p_init : float
        Prior probability of knowing a topic initially.  Default 0.3.
    p_transit : float
        Per-observation learning probability.  Default 0.1.
    p_slip : float
        Probability of disengagement despite understanding.  Default 0.05.
    p_guess : float
        Probability of engagement despite not understanding.  Default 0.2.

    Notes
    -----
    Thread-safe: all mutable user state is guarded by a
    :class:`threading.Lock`.
    """

    def __init__(
        self,
        *,
        success_threshold: float = 0.8,
        p_init: float = 0.3,
        p_transit: float = 0.1,
        p_slip: float = 0.05,
        p_guess: float = 0.2,
    ) -> None:
        self.success_threshold = success_threshold
        self._bkt_defaults: dict[str, float] = {
            "p_init": p_init,
            "p_transit": p_transit,
            "p_slip": p_slip,
            "p_guess": p_guess,
        }
        # {user_id: CompetencyTracker}
        self._trackers: dict[int, CompetencyTracker] = {}
        # {user_id: set_of_known_topics} -- tracks which topics have been
        # registered on each user's CompetencyTracker so we know when to
        # create a fresh tracker with the expanded topic list.
        self._user_topics: dict[int, set[str]] = {}
        self._lock = threading.Lock()

    # -- internal helpers ---------------------------------------------------

    def _ensure_tracker(self, user_id: int, topic: str) -> CompetencyTracker:
        """Return a ``CompetencyTracker`` for *user_id* that includes *topic*.

        If the user has no tracker yet, one is created with a single
        competency.  If the user already has a tracker but the topic is
        new, a fresh tracker is created that includes all previously
        known topics plus the new one.  Existing BKT posteriors are
        preserved by re-applying the observation-count trick (we store
        the raw ``p_known`` directly on the underlying BKT model).
        """
        if user_id not in self._trackers:
            tracker = CompetencyTracker(
                competencies=[topic],
                default_params=self._bkt_defaults,
                success_threshold=self.success_threshold,
            )
            self._trackers[user_id] = tracker
            self._user_topics[user_id] = {topic}
            return tracker

        known = self._user_topics[user_id]
        if topic in known:
            return self._trackers[user_id]

        # Need to expand the competency list.  Save current posteriors.
        old_tracker = self._trackers[user_id]
        old_mastery = old_tracker.get_mastery()

        new_topics = sorted(known | {topic})
        new_tracker = CompetencyTracker(
            competencies=new_topics,
            default_params=self._bkt_defaults,
            success_threshold=self.success_threshold,
        )
        # Restore posteriors by directly patching the internal BKT objects.
        for t, p in old_mastery.items():
            new_tracker._trackers[t]._p_known = p

        self._trackers[user_id] = new_tracker
        self._user_topics[user_id] = set(new_topics)
        return new_tracker

    # -- public API ---------------------------------------------------------

    def observe(
        self,
        user_id: int,
        topic: str,
        engaged: bool,
        timestamp: float | None = None,
    ) -> float:
        """Record an engagement observation.

        Parameters
        ----------
        user_id : int
            User identifier.
        topic : str
            Topic label of the content item.
        engaged : bool
            Whether the user engaged meaningfully (caller-defined).
        timestamp : float or None, optional
            Observation time (Unix epoch seconds).  Currently stored for
            future time-decay extensions but not used in the BKT update.

        Returns
        -------
        float
            Updated competence estimate for (*user_id*, *topic*),
            in [0, 1].
        """
        with self._lock:
            tracker = self._ensure_tracker(user_id, topic)
            updated = tracker.update(topic, engaged)
            logger.debug(
                "TopicTracker: user=%d topic=%r engaged=%s -> competence=%.3f",
                user_id,
                topic,
                engaged,
                updated,
            )
            return updated

    def competence(self, user_id: int, topic: str) -> float:
        """Current competence estimate for (*user_id*, *topic*).

        Parameters
        ----------
        user_id : int
            User identifier.
        topic : str
            Topic label.

        Returns
        -------
        float
            Competence probability in [0, 1].  Returns the BKT default
            ``p_init`` if the user has never been observed on *topic*.
        """
        with self._lock:
            if user_id not in self._trackers:
                return self._bkt_defaults["p_init"]
            known = self._user_topics[user_id]
            if topic not in known:
                return self._bkt_defaults["p_init"]
            return self._trackers[user_id].proficiency(topic)

    def user_profile(self, user_id: int) -> dict[str, float]:
        """All topic competences for a user.

        Parameters
        ----------
        user_id : int
            User identifier.

        Returns
        -------
        dict[str, float]
            Mapping ``{topic: competence}`` for every topic the user has
            been observed on.  Empty dict if user is unknown.
        """
        with self._lock:
            if user_id not in self._trackers:
                return {}
            return dict(self._trackers[user_id].get_mastery())

    def topic_coverage(self, user_id: int, all_topics: list[str]) -> float:
        """Fraction of topics where user competence exceeds the threshold.

        Parameters
        ----------
        user_id : int
            User identifier.
        all_topics : list of str
            Universe of topics to consider.

        Returns
        -------
        float
            Coverage ratio in [0.0, 1.0].  Returns 0.0 when
            *all_topics* is empty.
        """
        if not all_topics:
            return 0.0
        above = sum(
            1
            for t in all_topics
            if self.competence(user_id, t) >= self.success_threshold
        )
        return above / len(all_topics)

    def __repr__(self) -> str:
        with self._lock:
            n_users = len(self._trackers)
        return (
            f"TopicTracker(users={n_users}, "
            f"success_threshold={self.success_threshold})"
        )


# ---------------------------------------------------------------------------
# ReadingLevelEstimator
# ---------------------------------------------------------------------------


class ReadingLevelEstimator:
    """Estimates and tracks user reading level from engagement signals.

    Uses an exponential moving average of successfully-engaged item
    difficulties to estimate the user's current reading level.  The
    *stretch zone* then targets items slightly above this level to
    promote growth.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor in (0, 1].  Larger values weight recent
        observations more heavily.  Default 0.1.
    initial_level : float
        Default reading level for new users, on the same 0--1 scale as
        item difficulty.  Default 0.5.

    Raises
    ------
    ValueError
        If *alpha* is not in (0, 1] or *initial_level* is not in [0, 1].

    Notes
    -----
    Thread-safe: internal state is protected by a :class:`threading.Lock`.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        initial_level: float = 0.5,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= initial_level <= 1.0:
            raise ValueError(
                f"initial_level must be in [0, 1], got {initial_level}"
            )

        self.alpha = alpha
        self.initial_level = initial_level
        self._levels: dict[int, float] = {}
        self._lock = threading.Lock()

    def observe(
        self,
        user_id: int,
        item_difficulty: float,
        engaged: bool,
    ) -> float:
        """Update reading level estimate after an engagement signal.

        Only *engaged* interactions move the reading level.  Disengaged
        interactions are ignored to avoid penalizing users for skipping
        content that was too hard.

        Parameters
        ----------
        user_id : int
            User identifier.
        item_difficulty : float
            Difficulty of the consumed item, in [0, 1].
        engaged : bool
            Whether the user engaged meaningfully.

        Returns
        -------
        float
            Updated reading level estimate in [0, 1].
        """
        with self._lock:
            current = self._levels.get(user_id, self.initial_level)
            if engaged:
                current = (1.0 - self.alpha) * current + self.alpha * item_difficulty
                self._levels[user_id] = current
                logger.debug(
                    "ReadingLevel: user=%d difficulty=%.2f engaged -> level=%.3f",
                    user_id,
                    item_difficulty,
                    current,
                )
            return current

    def level(self, user_id: int) -> float:
        """Current reading level for *user_id*.

        Parameters
        ----------
        user_id : int
            User identifier.

        Returns
        -------
        float
            Reading level in [0, 1].  Returns *initial_level* if the
            user has never been observed.
        """
        with self._lock:
            return self._levels.get(user_id, self.initial_level)

    def stretch_zone(
        self,
        user_id: int,
        width: float = 0.15,
    ) -> tuple[float, float]:
        """Return difficulty bounds for the user's stretch zone.

        The stretch zone is a band centered *above* the user's current
        reading level.  Items whose difficulty falls inside this band
        are challenging but reachable.

        Parameters
        ----------
        user_id : int
            User identifier.
        width : float
            Half-width of the stretch zone.  Default 0.15.

        Returns
        -------
        tuple of (float, float)
            ``(lower, upper)`` difficulty bounds, each clamped to [0, 1].
        """
        lvl = self.level(user_id)
        lower = max(0.0, lvl - width)
        upper = min(1.0, lvl + width)
        return (lower, upper)

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._levels)
        return (
            f"ReadingLevelEstimator(alpha={self.alpha}, "
            f"initial_level={self.initial_level}, users={n})"
        )


# ---------------------------------------------------------------------------
# FeedRanker
# ---------------------------------------------------------------------------


class FeedRanker:
    """Content feed ranker combining progression, freshness, and diversity.

    Scores each candidate item as a weighted combination of:

    1. **Relevance** -- base recommender score (dot-product,
       collaborative filtering, etc.).
    2. **Freshness** -- exponential time-decay via :class:`FreshnessScorer`.
    3. **Stretch fit** -- how well item difficulty matches the user's
       reading level (via :class:`ReadingLevelEstimator`).
    4. **Topic diversity** -- MMR-style penalty for topics already
       selected into the ranked list.
    5. **Topic competence** -- user's readiness for the item's topic
       (via :class:`TopicTracker`).

    Final score::

        w_rel * relevance + w_fresh * freshness + w_stretch * stretch
        + w_diversity * diversity + w_competence * competence

    Parameters
    ----------
    freshness : FreshnessScorer or None, optional
        Freshness scorer.  Defaults to ``FreshnessScorer()`` with a
        24-hour half-life.
    topic_tracker : TopicTracker or None, optional
        Topic competence tracker.  Defaults to ``TopicTracker()``.
    reading_level : ReadingLevelEstimator or None, optional
        Reading level estimator.  Defaults to
        ``ReadingLevelEstimator()``.
    w_relevance : float
        Weight for the relevance component.  Default 0.3.
    w_freshness : float
        Weight for the freshness component.  Default 0.25.
    w_stretch : float
        Weight for the stretch-fit component.  Default 0.2.
    w_diversity : float
        Weight for the diversity component.  Default 0.15.
    w_competence : float
        Weight for the topic competence component.  Default 0.1.

    Notes
    -----
    Thread-safe: this class delegates to thread-safe sub-components and
    uses a lock around the greedy MMR selection loop.

    Examples
    --------
    >>> from orchid_ranker.curated_feed import FeedRanker, FeedItem, FreshnessScorer
    >>> ranker = FeedRanker(
    ...     freshness=FreshnessScorer(halflife_hours=12),
    ...     w_freshness=0.3,
    ... )
    >>> items = [
    ...     FeedItem(1, "tech", 0.4, time.time() - 3600),
    ...     FeedItem(2, "sports", 0.6, time.time() - 7200),
    ... ]
    >>> ranked = ranker.rank(user_id=42, candidates=items, top_k=2)
    >>> len(ranked)
    2
    """

    def __init__(
        self,
        *,
        freshness: FreshnessScorer | None = None,
        topic_tracker: TopicTracker | None = None,
        reading_level: ReadingLevelEstimator | None = None,
        w_relevance: float = 0.35,
        w_freshness: float = 0.25,
        w_stretch: float = 0.15,
        w_diversity: float = 0.10,
        w_competence: float = 0.15,
    ) -> None:
        self.freshness = freshness if freshness is not None else FreshnessScorer()
        self.topic_tracker = (
            topic_tracker if topic_tracker is not None else TopicTracker()
        )
        self.reading_level = (
            reading_level if reading_level is not None else ReadingLevelEstimator()
        )

        self.w_relevance = w_relevance
        self.w_freshness = w_freshness
        self.w_stretch = w_stretch
        self.w_diversity = w_diversity
        self.w_competence = w_competence

        self._lock = threading.Lock()

    # -- scoring helpers ----------------------------------------------------

    @staticmethod
    def _stretch_score(difficulty: float, lower: float, upper: float) -> float:
        """Score how well *difficulty* fits the stretch zone [lower, upper].

        Returns 1.0 when difficulty is inside the zone, decaying
        towards 0.0 as it moves away.  Uses a Gaussian-like bump
        centered on the zone midpoint.

        Parameters
        ----------
        difficulty : float
            Item difficulty in [0, 1].
        lower : float
            Lower bound of the stretch zone.
        upper : float
            Upper bound of the stretch zone.

        Returns
        -------
        float
            Stretch-fit score in (0, 1].
        """
        mid = (lower + upper) / 2.0
        half_width = max((upper - lower) / 2.0, 1e-9)
        z = (difficulty - mid) / half_width
        return math.exp(-0.5 * z * z)

    # -- public API ---------------------------------------------------------

    def rank(
        self,
        user_id: int,
        candidates: list[FeedItem],
        *,
        base_scores: np.ndarray | None = None,
        top_k: int = 20,
    ) -> list[ScoredFeedItem]:
        """Rank candidate items for a user's feed.

        Uses a greedy MMR-style selection loop: at each step, the
        candidate with the highest combined score (including a diversity
        penalty for already-selected topics) is appended to the result.

        Parameters
        ----------
        user_id : int
            User identifier.
        candidates : list of FeedItem
            Candidate content items.
        base_scores : numpy.ndarray or None, optional
            Pre-computed relevance scores from a base recommender, one
            per candidate.  If *None*, the relevance component is
            uniform (``1.0`` for every item).
        top_k : int
            Maximum number of items to return.  Default 20.

        Returns
        -------
        list of ScoredFeedItem
            Ranked items with component score breakdown, in descending
            order of ``total_score``.

        Raises
        ------
        ValueError
            If *base_scores* length does not match *candidates*.
        """
        if not candidates:
            return []

        n = len(candidates)

        if base_scores is not None:
            if len(base_scores) != n:
                raise ValueError(
                    f"base_scores length ({len(base_scores)}) != "
                    f"candidates length ({n})"
                )
            relevance = np.asarray(base_scores, dtype=np.float64)
        else:
            relevance = np.ones(n, dtype=np.float64)

        # Freshness scores (vectorised)
        timestamps = np.array(
            [c.timestamp for c in candidates], dtype=np.float64
        )
        now = time.time()
        fresh = self.freshness.scores_batch(timestamps, now=now)

        # Stretch zone
        lower, upper = self.reading_level.stretch_zone(user_id)
        stretch = np.array(
            [self._stretch_score(c.difficulty, lower, upper) for c in candidates],
            dtype=np.float64,
        )

        # Topic competence
        comp = np.array(
            [self.topic_tracker.competence(user_id, c.topic) for c in candidates],
            dtype=np.float64,
        )

        # --- Greedy MMR selection ---
        #
        # Diversity cannot be pre-computed because it depends on the
        # set of already-selected items.  We greedily pick the best
        # item at each step while updating the topic-diversity signal.

        with self._lock:
            selected: list[ScoredFeedItem] = []
            selected_topics: list[str] = []
            remaining = set(range(n))

            k = min(top_k, n)

            for _ in range(k):
                best_idx = -1
                best_score = -math.inf
                best_components: tuple[float, float, float, float, float] = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )

                for idx in remaining:
                    # Diversity: gradual decay as same-topic items accumulate.
                    # Binary 0/1 is too harsh — a second item from the same
                    # topic is fine; the third starts to feel repetitive.
                    if selected_topics:
                        same_count = sum(
                            1 for t in selected_topics
                            if t == candidates[idx].topic
                        )
                        div = max(0.0, 1.0 - 0.3 * same_count)
                    else:
                        div = 1.0

                    total = (
                        self.w_relevance * relevance[idx]
                        + self.w_freshness * fresh[idx]
                        + self.w_stretch * stretch[idx]
                        + self.w_diversity * div
                        + self.w_competence * comp[idx]
                    )

                    if total > best_score:
                        best_score = total
                        best_idx = idx
                        best_components = (
                            float(relevance[idx]),
                            float(fresh[idx]),
                            float(stretch[idx]),
                            div,
                            float(comp[idx]),
                        )

                if best_idx < 0:
                    break  # pragma: no cover — should not happen

                rel_s, fre_s, str_s, div_s, cmp_s = best_components
                selected.append(
                    ScoredFeedItem(
                        item=candidates[best_idx],
                        total_score=best_score,
                        relevance_score=rel_s,
                        freshness_score=fre_s,
                        stretch_score=str_s,
                        diversity_score=div_s,
                        competence_score=cmp_s,
                    )
                )
                selected_topics.append(candidates[best_idx].topic)
                remaining.discard(best_idx)

        logger.debug(
            "FeedRanker: user=%d candidates=%d top_k=%d -> selected=%d",
            user_id,
            n,
            top_k,
            len(selected),
        )
        return selected

    def observe(
        self,
        user_id: int,
        item: FeedItem,
        engaged: bool,
    ) -> None:
        """Record a user-item engagement for online adaptation.

        Updates both the :class:`TopicTracker` competence estimate and
        the :class:`ReadingLevelEstimator` for the user.

        Parameters
        ----------
        user_id : int
            User identifier.
        item : FeedItem
            The content item that was shown / consumed.
        engaged : bool
            Whether the user engaged meaningfully.
        """
        self.topic_tracker.observe(
            user_id, item.topic, engaged, timestamp=time.time()
        )
        self.reading_level.observe(user_id, item.difficulty, engaged)
        logger.debug(
            "FeedRanker.observe: user=%d item=%d topic=%r engaged=%s",
            user_id,
            item.item_id,
            item.topic,
            engaged,
        )

    def __repr__(self) -> str:
        return (
            f"FeedRanker("
            f"w_relevance={self.w_relevance}, "
            f"w_freshness={self.w_freshness}, "
            f"w_stretch={self.w_stretch}, "
            f"w_diversity={self.w_diversity}, "
            f"w_competence={self.w_competence})"
        )
