#!/usr/bin/env python3
"""Benchmark: end-to-end pipeline on MovieLens-1M.

Measures the full user lifecycle:
  cold-start -> warm transition -> taste progression re-ranking

Compares four systems:

1. **Popularity** — global popularity baseline (no personalisation).
2. **Orchid direct** — raw OrchidRecommender, no cold-start handling.
3. **ColdStartBridge only** — bridge that transitions to Orchid, but
   no taste re-ranking.
4. **Full pipeline** — ColdStartBridge -> Orchid -> TasteProgressionRanker.

Headline metrics:

* Session survival (Surv@5, Surv@10, Surv@20)
* Mean session length
* Warmth transition speed
* Kept-rate by phase (cold 0-5, transition 5-15, warm 15+)

Usage::

    # Smoke test (~3 min)
    PYTHONPATH=src python benchmarks/end_to_end_bench.py --smoke

    # Full run (~30 min)
    PYTHONPATH=src python benchmarks/end_to_end_bench.py

Outputs ``benchmarks/results_end_to_end.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

from movielens_1m.download import download_and_extract
from movielens_1m.preprocess import MovieLensData, preprocess
from movielens_1m.simulator import ClickSimulator

from orchid_ranker.cold_start import (
    ColdStartBridge,
    ColdStartConfig,
    ItemFeatureIndex,
    PopularityPrior,
)
from orchid_ranker.taste_progression import (
    SophisticationMapper,
    TasteConfig,
    TasteProgressionRanker,
)

logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Movie sophistication mapping
# ---------------------------------------------------------------------------

def build_sophistication_scores(
    data: MovieLensData,
    movies_df: pd.DataFrame,
) -> Dict[int, float]:
    """Build sophistication scores for MovieLens items.

    Uses inverted popularity rank as a proxy: niche movies are
    more "sophisticated".  This is imperfect but directionally
    correct -- widely-seen blockbusters are entry-level, while
    obscure arthouse films require more developed taste.

    Scores are in [0, 1] where 0 = most popular (least
    sophisticated) and 1 = least popular (most sophisticated).

    Parameters
    ----------
    data : MovieLensData
        Preprocessed MovieLens data.
    movies_df : pd.DataFrame
        Raw movies DataFrame with genre information.

    Returns
    -------
    dict of int to float
        Mapping from 0-based item index to sophistication score.
    """
    # Count interactions per item in training data
    item_counts = data.train.groupby("item_id").size()

    # Build scores for all items
    scores: Dict[int, float] = {}
    max_count = item_counts.max() if len(item_counts) > 0 else 1

    for idx in range(data.num_items):
        orig_id = data.idx_to_item_id.get(idx)
        if orig_id is None:
            scores[idx] = 0.5
            continue

        count = item_counts.get(orig_id, 0)
        # Inverted popularity: popular items get low sophistication
        # Use log scale to compress the long tail
        if count > 0:
            pop_score = np.log1p(count) / np.log1p(max_count)
            scores[idx] = 1.0 - pop_score
        else:
            scores[idx] = 0.9  # unseen items are niche

    return scores


def build_item_categories(
    data: MovieLensData,
    movies_df: pd.DataFrame,
) -> Dict[int, str]:
    """Map item indices to their primary genre (first listed genre).

    Parameters
    ----------
    data : MovieLensData
        Preprocessed MovieLens data.
    movies_df : pd.DataFrame
        Raw movies DataFrame.

    Returns
    -------
    dict of int to str
        Mapping from 0-based item index to genre string.
    """
    # Build original item_id -> primary genre
    orig_to_genre: Dict[int, str] = {}
    for _, row in movies_df.iterrows():
        genres = str(row["genres"]).split("|")
        orig_to_genre[int(row["item_id"])] = genres[0] if genres else "Unknown"

    # Map to index space
    categories: Dict[int, str] = {}
    for idx in range(data.num_items):
        orig_id = data.idx_to_item_id.get(idx)
        if orig_id is not None:
            categories[idx] = orig_to_genre.get(orig_id, "Unknown")
        else:
            categories[idx] = "Unknown"

    return categories


# ---------------------------------------------------------------------------
# Recommender adapters (for the replay interface)
# ---------------------------------------------------------------------------

class _IndexSpaceOrchidWrapper:
    """Wraps OrchidRecommender to operate in 0-based index space.

    The ColdStartBridge needs a recommender that speaks the same ID
    space as the item feature matrix (0-based indices).  This wrapper
    translates between index space and the original ID space that
    OrchidRecommender uses.
    """

    def __init__(
        self,
        rec: Any,
        idx_to_user_id: Dict[int, int],
        idx_to_item_id: Dict[int, int],
        item_id_to_idx: Dict[int, int],
        user_id_to_idx: Dict[int, int],
    ) -> None:
        self._rec = rec
        self._idx_to_uid = idx_to_user_id
        self._idx_to_iid = idx_to_item_id
        self._iid_to_idx = item_id_to_idx
        self._uid_to_idx = user_id_to_idx

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[Tuple[int, float]]:
        uid = user_id
        try:
            recs = self._rec.recommend(uid, top_k=top_k, filter_seen=True)
        except (KeyError, RuntimeError):
            return []

        candidate_idx = kwargs.get("candidate_item_ids")
        allowed: Optional[Set[int]] = set(candidate_idx) if candidate_idx is not None else None

        result = []
        for r in recs:
            orig_iid = r.item_id
            idx = self._iid_to_idx.get(orig_iid)
            if idx is not None:
                if allowed is not None and idx not in allowed:
                    continue
                result.append(type("Rec", (), {"item_id": idx, "score": r.score})())
        return result[:top_k]


class _PopularityOnlyRecommender:
    """Baseline: always returns globally popular items."""

    def __init__(self, pop_prior: PopularityPrior, num_items: int) -> None:
        self._pop = pop_prior
        self._num_items = num_items
        self._all_items = list(range(num_items))

    def recommend(
        self, user_idx: int, k: int = 10, exclude: Optional[Set[int]] = None
    ) -> List[int]:
        exclude = exclude or set()
        candidates = [i for i in self._all_items if i not in exclude]
        if not candidates:
            return []
        scores = self._pop.scores(candidates)
        top_idx = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in top_idx]

    def score(self, user_idx: int, item_idx: int) -> float:
        s = self._pop.scores([item_idx])
        return float(s[0])

    def observe(self, user_idx: int, item_idx: int, outcome: float) -> None:
        pass  # Popularity baseline does not adapt


class _OrchidDirectRecommender:
    """Wraps a raw OrchidRecommender (no cold-start handling) for replay."""

    def __init__(
        self,
        rec: Any,
        idx_to_user_id: Dict[int, int],
        idx_to_item_id: Dict[int, int],
        item_id_to_idx: Dict[int, int],
    ) -> None:
        self._rec = rec
        self._idx_to_uid = idx_to_user_id
        self._idx_to_iid = idx_to_item_id
        self._iid_to_idx = item_id_to_idx

    def recommend(
        self, user_idx: int, k: int = 10, exclude: Optional[Set[int]] = None
    ) -> List[int]:
        uid = self._idx_to_uid.get(user_idx)
        if uid is None:
            return []
        try:
            recs = self._rec.recommend(uid, top_k=k, filter_seen=True)
            items_idx = []
            for r in recs:
                idx = self._iid_to_idx.get(r.item_id)
                if idx is not None:
                    items_idx.append(idx)
            if exclude:
                items_idx = [i for i in items_idx if i not in exclude]
            return items_idx[:k]
        except (KeyError, RuntimeError):
            return []

    def score(self, user_idx: int, item_idx: int) -> float:
        uid = self._idx_to_uid.get(user_idx)
        iid = self._idx_to_iid.get(item_idx)
        if uid is None or iid is None:
            return 0.0
        try:
            return self._rec.predict(uid, iid)
        except (KeyError, RuntimeError):
            return 0.0

    def observe(self, user_idx: int, item_idx: int, outcome: float) -> None:
        pass  # Orchid direct does not adapt online


class _ColdStartBridgeRecommender:
    """Wraps ColdStartBridge for the replay interface."""

    def __init__(
        self,
        bridge: ColdStartBridge,
        idx_to_user_id: Dict[int, int],
        idx_to_item_id: Dict[int, int],
        item_id_to_idx: Dict[int, int],
        num_items: int,
    ) -> None:
        self._bridge = bridge
        self._idx_to_uid = idx_to_user_id
        self._idx_to_iid = idx_to_item_id
        self._iid_to_idx = item_id_to_idx
        self._num_items = num_items

    def recommend(
        self, user_idx: int, k: int = 10, exclude: Optional[Set[int]] = None
    ) -> List[int]:
        uid = self._idx_to_uid.get(user_idx)
        if uid is None:
            return []

        all_candidates = [
            idx for idx in range(self._num_items)
            if idx not in (exclude or set())
        ]

        try:
            recs = self._bridge.recommend(
                user_id=uid,
                top_k=k,
                candidate_item_ids=all_candidates,
            )
            return [item_id for item_id, _score in recs]
        except Exception as exc:
            logger.debug("ColdStartBridge recommend failed for user %d: %s", uid, exc)
            return []

    def score(self, user_idx: int, item_idx: int) -> float:
        uid = self._idx_to_uid.get(user_idx)
        if uid is None:
            return 0.0
        try:
            recs = self._bridge.recommend(user_id=uid, top_k=1, candidate_item_ids=[item_idx])
            if recs:
                return recs[0][1]
        except Exception:
            pass
        return 0.0

    def observe(self, user_idx: int, item_idx: int, outcome: float) -> None:
        uid = self._idx_to_uid.get(user_idx)
        if uid is not None:
            self._bridge.observe(uid, item_idx, outcome=outcome)


class _FullPipelineRecommender:
    """Composes ColdStartBridge -> TasteProgressionRanker.

    The bridge handles cold-start -> warm transition.  Once candidates
    are produced, TasteProgressionRanker re-ranks them by taste-
    trajectory fit (stretch-zone, momentum, exploration).
    """

    def __init__(
        self,
        bridge: ColdStartBridge,
        taste_ranker: TasteProgressionRanker,
        idx_to_user_id: Dict[int, int],
        idx_to_item_id: Dict[int, int],
        item_id_to_idx: Dict[int, int],
        item_categories: Dict[int, str],
        sophistication: Dict[int, float],
        num_items: int,
    ) -> None:
        self._bridge = bridge
        self._taste_ranker = taste_ranker
        self._idx_to_uid = idx_to_user_id
        self._idx_to_iid = idx_to_item_id
        self._iid_to_idx = item_id_to_idx
        self._item_categories = item_categories
        self._sophistication = sophistication
        self._num_items = num_items

    def recommend(
        self, user_idx: int, k: int = 10, exclude: Optional[Set[int]] = None
    ) -> List[int]:
        uid = self._idx_to_uid.get(user_idx)
        if uid is None:
            return []

        all_candidates = [
            idx for idx in range(self._num_items)
            if idx not in (exclude or set())
        ]

        try:
            # Step 1: Get candidates from bridge (handles cold->warm)
            # Request more candidates so taste ranker has room to re-rank
            bridge_k = min(k * 5, len(all_candidates))
            bridge_recs = self._bridge.recommend(
                user_id=uid,
                top_k=bridge_k,
                candidate_item_ids=all_candidates,
            )

            if not bridge_recs:
                return []

            candidate_ids = [item_id for item_id, _score in bridge_recs]

            # Step 2: Re-rank using taste progression
            reranked = self._taste_ranker.recommend(
                user_id=user_idx,
                top_k=k,
                candidate_item_ids=candidate_ids,
            )

            return [item_id for item_id, _score in reranked]
        except Exception as exc:
            logger.debug("FullPipeline recommend failed for user %d: %s", uid, exc)
            return []

    def score(self, user_idx: int, item_idx: int) -> float:
        uid = self._idx_to_uid.get(user_idx)
        if uid is None:
            return 0.0
        try:
            recs = self._bridge.recommend(user_id=uid, top_k=1, candidate_item_ids=[item_idx])
            if recs:
                return recs[0][1]
        except Exception:
            pass
        return 0.0

    def observe(self, user_idx: int, item_idx: int, outcome: float) -> None:
        uid = self._idx_to_uid.get(user_idx)
        if uid is not None:
            # Observe in bridge
            self._bridge.observe(uid, item_idx, outcome=outcome)

        # Observe in taste ranker (uses index space for user_id)
        cat = self._item_categories.get(item_idx, "__general__")
        positive = outcome > 0.5
        rating = 4.5 if positive else 2.0
        self._taste_ranker.observe(
            user_id=user_idx,
            item_id=item_idx,
            purchased=True,
            returned=not positive,
            rating=rating,
            category=cat,
        )


# ---------------------------------------------------------------------------
# Replay with observation and per-phase tracking
# ---------------------------------------------------------------------------

def _replay_with_phases(
    simulator: ClickSimulator,
    recommender: Any,
    num_users: int,
    max_steps: int = 30,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Run replay sessions with per-phase kept-rate tracking.

    Phases:
    - cold: interactions 0-5
    - transition: interactions 5-15
    - warm: interactions 15+

    On each step the recommender produces a top-1 recommendation.
    The simulator decides if the user clicks.  On click, observe()
    is called on the recommender.  On non-click, the session ends.
    """
    rng = np.random.default_rng(seed)
    session_lengths = np.zeros(num_users, dtype=np.int64)

    # Per-phase tracking: count clicks and attempts
    phase_clicks: Dict[str, int] = {"cold": 0, "transition": 0, "warm": 0}
    phase_attempts: Dict[str, int] = {"cold": 0, "transition": 0, "warm": 0}

    for user_idx in range(num_users):
        seen: set[int] = set()
        length = 0

        for step in range(max_steps):
            # Determine phase
            if length < 5:
                phase = "cold"
            elif length < 15:
                phase = "transition"
            else:
                phase = "warm"

            recs = recommender.recommend(user_idx, k=1, exclude=seen)
            if not recs:
                break

            item_idx = recs[0]
            p = simulator.click_prob(user_idx, item_idx)
            clicked = bool(rng.random() < p)

            phase_attempts[phase] += 1

            if clicked:
                seen.add(item_idx)
                length += 1
                phase_clicks[phase] += 1
                # Notify the recommender about the click
                if hasattr(recommender, "observe"):
                    recommender.observe(user_idx, item_idx, outcome=1.0)
            else:
                break

        session_lengths[user_idx] = length

        if (user_idx + 1) % 100 == 0:
            logger.info(
                "  replayed %d / %d users (mean length so far: %.2f)",
                user_idx + 1,
                num_users,
                float(session_lengths[: user_idx + 1].mean()),
            )

    # Compute metrics
    mean_len = float(session_lengths.mean())
    survival_5 = float((session_lengths >= 5).mean())
    survival_10 = float((session_lengths >= 10).mean())
    survival_20 = float((session_lengths >= 20).mean())

    # Per-phase kept-rate
    kept_rate = {}
    for phase in ["cold", "transition", "warm"]:
        attempts = phase_attempts[phase]
        clicks = phase_clicks[phase]
        kept_rate[phase] = round(clicks / max(attempts, 1), 4)

    return {
        "session_lengths": session_lengths,
        "survival_5": survival_5,
        "survival_10": survival_10,
        "survival_20": survival_20,
        "mean_session_length": mean_len,
        "phase_kept_rate": kept_rate,
        "phase_attempts": {k: int(v) for k, v in phase_attempts.items()},
        "phase_clicks": {k: int(v) for k, v in phase_clicks.items()},
    }


# ---------------------------------------------------------------------------
# Survival benchmark
# ---------------------------------------------------------------------------

def bench_end_to_end_survival(
    data: MovieLensData,
    simulator: ClickSimulator,
    pop_prior: PopularityPrior,
    item_index: ItemFeatureIndex,
    orchid_rec: Any,
    sophistication: Dict[int, float],
    item_categories: Dict[int, str],
    *,
    max_steps: int = 30,
    num_users: int = 200,
    smoke: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run replay sessions for all four systems.

    Returns per-system survival, session length, and phase kept-rate.
    """
    if smoke:
        num_users = min(num_users, 50)
        max_steps = min(max_steps, 10)

    results: Dict[str, Dict[str, Any]] = {}

    # --- System 1: Popularity baseline ---
    pop_rec = _PopularityOnlyRecommender(pop_prior, data.num_items)
    logger.info("Replaying: popularity baseline (%d users, %d steps)", num_users, max_steps)
    pop_replay = _replay_with_phases(
        simulator=simulator,
        recommender=pop_rec,
        num_users=num_users,
        max_steps=max_steps,
        seed=SEED,
    )
    results["popularity"] = _extract_results(pop_replay)
    _print_system_results("popularity", results["popularity"])

    # --- System 2: Orchid direct (no cold-start handling) ---
    orchid_adapter = _OrchidDirectRecommender(
        rec=orchid_rec,
        idx_to_user_id=data.idx_to_user_id,
        idx_to_item_id=data.idx_to_item_id,
        item_id_to_idx=data.item_id_to_idx,
    )
    logger.info("Replaying: Orchid direct (%d users, %d steps)", num_users, max_steps)
    orchid_replay = _replay_with_phases(
        simulator=simulator,
        recommender=orchid_adapter,
        num_users=num_users,
        max_steps=max_steps,
        seed=SEED,
    )
    results["orchid_direct"] = _extract_results(orchid_replay)
    _print_system_results("orchid_direct", results["orchid_direct"])

    # --- System 3: ColdStartBridge only (no taste re-ranking) ---
    orchid_idx = _IndexSpaceOrchidWrapper(
        orchid_rec,
        data.idx_to_user_id,
        data.idx_to_item_id,
        data.item_id_to_idx,
        data.user_id_to_idx,
    )
    bridge_only = ColdStartBridge(
        recommender=orchid_idx,
        item_features=data.item_features,
        popularity_prior=pop_prior,
        config=ColdStartConfig(min_interactions=3, blend_until=20),
    )
    bridge_adapter = _ColdStartBridgeRecommender(
        bridge=bridge_only,
        idx_to_user_id=data.idx_to_user_id,
        idx_to_item_id=data.idx_to_item_id,
        item_id_to_idx=data.item_id_to_idx,
        num_items=data.num_items,
    )
    logger.info("Replaying: ColdStartBridge only (%d users, %d steps)", num_users, max_steps)
    bridge_replay = _replay_with_phases(
        simulator=simulator,
        recommender=bridge_adapter,
        num_users=num_users,
        max_steps=max_steps,
        seed=SEED,
    )
    results["bridge_only"] = _extract_results(bridge_replay)
    _print_system_results("bridge_only", results["bridge_only"])

    # --- System 4: Full pipeline (ColdStartBridge + TasteProgressionRanker) ---
    orchid_idx_full = _IndexSpaceOrchidWrapper(
        orchid_rec,
        data.idx_to_user_id,
        data.idx_to_item_id,
        data.item_id_to_idx,
        data.user_id_to_idx,
    )
    bridge_full = ColdStartBridge(
        recommender=orchid_idx_full,
        item_features=data.item_features,
        popularity_prior=pop_prior,
        config=ColdStartConfig(min_interactions=3, blend_until=20),
    )

    # Set up taste progression ranker
    soph_mapper = SophisticationMapper(sophistication)
    taste_config = TasteConfig(
        stretch_width=0.15,
        keep_threshold=4.0,
        bkt_p_init=0.2,
        bkt_p_transit=0.08,
    )
    taste_ranker = TasteProgressionRanker(
        sophistication_scores=soph_mapper,
        config=taste_config,
    )
    taste_ranker.set_item_categories(item_categories)

    full_pipeline = _FullPipelineRecommender(
        bridge=bridge_full,
        taste_ranker=taste_ranker,
        idx_to_user_id=data.idx_to_user_id,
        idx_to_item_id=data.idx_to_item_id,
        item_id_to_idx=data.item_id_to_idx,
        item_categories=item_categories,
        sophistication=sophistication,
        num_items=data.num_items,
    )
    logger.info("Replaying: Full pipeline (%d users, %d steps)", num_users, max_steps)
    full_replay = _replay_with_phases(
        simulator=simulator,
        recommender=full_pipeline,
        num_users=num_users,
        max_steps=max_steps,
        seed=SEED,
    )
    results["full_pipeline"] = _extract_results(full_replay)
    _print_system_results("full_pipeline", results["full_pipeline"])

    return results


def _extract_results(replay: Dict[str, Any]) -> Dict[str, Any]:
    """Extract serializable results from a replay dict."""
    return {
        "survival_5": round(replay["survival_5"], 4),
        "survival_10": round(replay["survival_10"], 4),
        "survival_20": round(replay["survival_20"], 4),
        "mean_session_length": round(replay["mean_session_length"], 2),
        "phase_kept_rate": replay["phase_kept_rate"],
        "phase_attempts": replay["phase_attempts"],
        "phase_clicks": replay["phase_clicks"],
    }


def _print_system_results(name: str, res: Dict[str, Any]) -> None:
    """Print results for one system."""
    pkr = res["phase_kept_rate"]
    print(
        f"  {name:20s}: surv@5={res['survival_5']:.3f} "
        f"surv@10={res['survival_10']:.3f} "
        f"surv@20={res['survival_20']:.3f} "
        f"mean={res['mean_session_length']:.2f}  "
        f"kept[cold={pkr['cold']:.3f} trans={pkr['transition']:.3f} "
        f"warm={pkr['warm']:.3f}]"
    )


# ---------------------------------------------------------------------------
# Warmth transition analysis
# ---------------------------------------------------------------------------

def bench_warmth_curve(
    data: MovieLensData,
    orchid_rec: Any,
    pop_prior: PopularityPrior,
    *,
    num_users: int = 200,
    max_interactions: int = 25,
    seed: int = SEED,
) -> Dict[int, float]:
    """Track mean warmth as users accumulate interactions.

    Returns {interaction_count: mean_warmth} for the cold-to-warm
    transition curve.
    """
    rng = np.random.default_rng(seed)

    orchid_idx = _IndexSpaceOrchidWrapper(
        orchid_rec,
        data.idx_to_user_id,
        data.idx_to_item_id,
        data.item_id_to_idx,
        data.user_id_to_idx,
    )
    bridge = ColdStartBridge(
        recommender=orchid_idx,
        item_features=data.item_features,
        popularity_prior=pop_prior,
        config=ColdStartConfig(min_interactions=3, blend_until=20),
    )

    user_counts = data.train.groupby("user_id").size()
    eligible = user_counts[user_counts >= max_interactions].index.tolist()
    eval_users = rng.choice(eligible, size=min(num_users, len(eligible)), replace=False)

    warmth_by_step: Dict[int, List[float]] = {i: [] for i in range(max_interactions + 1)}

    for uid in eval_users:
        user_df = data.train[data.train["user_id"] == uid].sort_values("timestamp")
        orig_items = user_df["item_id"].tolist()[:max_interactions]
        labels = user_df["label"].tolist()[:max_interactions]
        items_idx = [data.item_id_to_idx.get(int(iid)) for iid in orig_items]

        warmth_by_step[0].append(bridge.warmth(int(uid)))

        for step, (item_idx, lbl) in enumerate(zip(items_idx, labels), start=1):
            if item_idx is not None:
                bridge.observe(int(uid), item_idx, outcome=float(lbl))
            warmth_by_step[step].append(bridge.warmth(int(uid)))

    return {
        step: round(float(np.mean(vals)), 4) if vals else 0.0
        for step, vals in warmth_by_step.items()
    }


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def _print_markdown(results: Dict[str, Any]) -> None:
    """Print Markdown tables for docs."""
    survival = results.get("survival", {})

    # Survival table
    print("\n### End-to-end session survival\n")
    print("| System | Surv@5 | Surv@10 | Surv@20 | Mean session |")
    print("|--------|--------|---------|---------|-------------|")
    for sys_name in ["popularity", "orchid_direct", "bridge_only", "full_pipeline"]:
        row = survival.get(sys_name, {})
        print(
            f"| {sys_name:<20s} "
            f"| {row.get('survival_5', 0):.4f} "
            f"| {row.get('survival_10', 0):.4f} "
            f"| {row.get('survival_20', 0):.4f} "
            f"| {row.get('mean_session_length', 0):.2f} |"
        )

    # Phase kept-rate table
    print("\n### Kept-rate by phase\n")
    print("| System | Cold (0-5) | Transition (5-15) | Warm (15+) |")
    print("|--------|-----------|-------------------|-----------|")
    for sys_name in ["popularity", "orchid_direct", "bridge_only", "full_pipeline"]:
        row = survival.get(sys_name, {})
        pkr = row.get("phase_kept_rate", {})
        print(
            f"| {sys_name:<20s} "
            f"| {pkr.get('cold', 0):.4f} "
            f"| {pkr.get('transition', 0):.4f} "
            f"| {pkr.get('warm', 0):.4f} |"
        )

    # Warmth curve
    warmth = results.get("warmth_curve", {})
    if warmth:
        print("\n### Warmth transition\n")
        print("| Interactions | Mean warmth |")
        print("|-------------|------------|")
        for step, w in sorted(warmth.items(), key=lambda x: int(x[0])):
            print(f"| {int(step):>11d} | {w:.4f} |")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(*, smoke: bool = False) -> Dict[str, Any]:
    """Execute the full end-to-end pipeline benchmark."""
    t_start = time.perf_counter()
    mode = "SMOKE" if smoke else "FULL"
    logger.info("=" * 70)
    logger.info("End-to-End Pipeline Benchmark (MovieLens-1M) -- %s mode", mode)
    logger.info("=" * 70)

    # Step 1: Download and preprocess
    logger.info("[1/6] Downloading MovieLens-1M...")
    ml_dir = download_and_extract()

    logger.info("[2/6] Preprocessing...")
    data = preprocess(data_dir=ml_dir, seed=SEED)
    logger.info(
        "  users=%d  items=%d  train=%d",
        data.num_users, data.num_items, len(data.train),
    )

    # Load raw movies for genre / sophistication
    from movielens_1m.preprocess import load_raw
    _, movies_df = load_raw(ml_dir)

    # Step 2: Build sophistication scores and item categories
    logger.info("[3/6] Building sophistication scores and item categories...")
    sophistication = build_sophistication_scores(data, movies_df)
    item_categories = build_item_categories(data, movies_df)

    soph_vals = list(sophistication.values())
    logger.info(
        "  Sophistication: min=%.3f  max=%.3f  mean=%.3f",
        min(soph_vals), max(soph_vals), np.mean(soph_vals),
    )

    # Step 3: Train click simulator
    logger.info("[4/6] Training click simulator...")
    sim_epochs = 3 if smoke else 15
    simulator = ClickSimulator(
        num_users=data.num_users,
        num_items=data.num_items,
        item_features=data.item_features,
        embed_dim=32,
        hidden_dim=64,
        device="cpu",
    )
    sim_metrics = simulator.fit(
        data.train,
        data.user_id_to_idx,
        data.item_id_to_idx,
        epochs=sim_epochs,
        seed=SEED,
    )
    logger.info(
        "  Simulator final loss=%.4f  acc=%.4f",
        sim_metrics["final_loss"], sim_metrics["final_acc"],
    )

    # Step 4: Fit Orchid recommender
    logger.info("[5/6] Fitting OrchidRecommender (ALS)...")
    from orchid_ranker.recommender import OrchidRecommender

    orchid_rec = OrchidRecommender(strategy="als", emb_dim=64, lr=0.01)
    orchid_rec.fit(
        data.train,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
    )
    logger.info("  OrchidRecommender fitted.")

    # Step 5: Build shared components
    pop_prior = PopularityPrior(smoothing=1.0)
    pop_prior.fit(data.train["item_id"].tolist())

    item_index = ItemFeatureIndex(data.item_features)

    # Step 6: Run benchmarks
    logger.info("[6/6] Running end-to-end benchmarks...")
    results: Dict[str, Any] = {}

    # 6a: Session survival and phase kept-rate
    num_users = 50 if smoke else 200
    max_steps = 10 if smoke else 30
    print(f"\n=== End-to-end session survival ({num_users} users, {max_steps} steps) ===")
    survival = bench_end_to_end_survival(
        data, simulator, pop_prior, item_index, orchid_rec,
        sophistication, item_categories,
        max_steps=max_steps,
        num_users=num_users,
        smoke=smoke,
    )
    results["survival"] = survival

    # 6b: Warmth transition curve
    print("\n=== Warmth transition curve ===")
    warmth = bench_warmth_curve(
        data, orchid_rec, pop_prior,
        num_users=50 if smoke else 200,
    )
    results["warmth_curve"] = warmth
    print("  warmth: " + " -> ".join(
        f"{step}:{w:.2f}" for step, w in list(warmth.items())[:11]
    ))

    # Build config
    import datetime
    config = {
        "seed": SEED,
        "smoke": smoke,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "num_users": data.num_users,
        "num_items": data.num_items,
        "num_eval_users": num_users,
        "max_steps": max_steps,
        "simulator_epochs": sim_epochs,
        "cold_start_config": {
            "min_interactions": 3,
            "blend_until": 20,
            "popularity_weight": 0.3,
        },
        "taste_config": {
            "stretch_width": 0.15,
            "keep_threshold": 4.0,
            "bkt_p_init": 0.2,
            "bkt_p_transit": 0.08,
        },
    }
    results["config"] = config

    # Write JSON
    out_path = OUTPUT_DIR / "results_end_to_end.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", out_path)

    # Print markdown tables
    _print_markdown(results)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "End-to-end benchmark complete in %.1fs (%.1f min)",
        elapsed, elapsed / 60,
    )
    print(f"\nCompleted in {elapsed:.1f}s. Results: {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    global SEED  # noqa: PLW0603

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="python benchmarks/end_to_end_bench.py",
        description="End-to-end pipeline benchmark on MovieLens-1M.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test (50 users, 10 steps).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED}).",
    )
    args = parser.parse_args()

    SEED = args.seed

    run(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
