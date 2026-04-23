#!/usr/bin/env python3
"""Benchmark: orchid_ranker.cold_start module on MovieLens-1M.

Evaluates cold-start handling by simulating brand-new users with no
interaction history.  For each user the first 10 ratings are stripped;
three systems are compared on their ability to sustain engagement
during the cold-start ramp-up phase:

1. **Popularity baseline** — always recommends globally popular items.
2. **Content-only baseline** — cosine similarity from seed interactions.
3. **ColdStartBridge** — blends popularity + content → Orchid as
   interactions accumulate.

The headline metric is **session-N survival** (N ∈ {5, 10, 20}):
what fraction of simulated user sessions last at least N steps?

Usage::

    # Full run (requires MovieLens-1M download, ~30 min)
    PYTHONPATH=src python benchmarks/cold_start_bench.py

    # Quick smoke test (~2 min)
    PYTHONPATH=src python benchmarks/cold_start_bench.py --smoke

Outputs ``benchmarks/results_cold_start.json`` and a Markdown table
suitable for ``docs/benchmarks/cold-start.md``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
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
from movielens_1m.simulator import ClickSimulator, replay_sessions

from orchid_ranker.cold_start import (
    ColdStartBridge,
    ColdStartConfig,
    ItemFeatureIndex,
    PopularityPrior,
)

logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Cold-start recommender adapters (for replay_sessions interface)
# ---------------------------------------------------------------------------

class _PopularityOnlyRecommender:
    """Baseline: always returns globally popular items."""

    def __init__(
        self,
        pop_prior: PopularityPrior,
        num_items: int,
    ) -> None:
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


class _ContentOnlyRecommender:
    """Baseline: recommends items similar to seed items (user's recent history)."""

    def __init__(
        self,
        item_index: ItemFeatureIndex,
        seed_count: int = 5,
    ) -> None:
        self._index = item_index
        self._seed_count = seed_count
        # Track per-user history for building profiles
        self._user_history: Dict[int, List[int]] = defaultdict(list)

    def observe(self, user_idx: int, item_idx: int) -> None:
        self._user_history[user_idx].append(item_idx)

    def recommend(
        self, user_idx: int, k: int = 10, exclude: Optional[Set[int]] = None
    ) -> List[int]:
        exclude = exclude or set()
        seeds = self._user_history.get(user_idx, [])[-self._seed_count:]

        if not seeds:
            # No history at all — return arbitrary items
            available = [i for i in range(self._index.num_items) if i not in exclude]
            return available[:k]

        scores = self._index.user_profile_scores(seeds, exclude=exclude)
        # Mask excluded
        mask = np.ones(len(scores), dtype=bool)
        for eid in exclude:
            if 0 <= eid < len(scores):
                mask[eid] = False
        scores[~mask] = -np.inf

        top_idx = np.argsort(scores)[::-1][:k]
        return [int(i) for i in top_idx if scores[i] > -np.inf]

    def score(self, user_idx: int, item_idx: int) -> float:
        seeds = self._user_history.get(user_idx, [])[-self._seed_count:]
        if not seeds:
            return 0.0
        all_scores = self._index.user_profile_scores(seeds)
        return float(all_scores[item_idx]) if 0 <= item_idx < len(all_scores) else 0.0


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
        """Recommend in index space.

        Accepts user_id as either a 0-based index or an original ID.
        Returns objects with ``.item_id`` and ``.score`` in index space.
        Respects ``candidate_item_ids`` from kwargs when passed by
        the ColdStartBridge.
        """
        # user_id from the bridge is an original user ID
        uid = user_id
        try:
            recs = self._rec.recommend(uid, top_k=top_k, filter_seen=True)
        except (KeyError, RuntimeError):
            return []

        # Build allowed set from candidate_item_ids (index space) if provided
        candidate_idx = kwargs.get("candidate_item_ids")
        allowed: Optional[Set[int]] = set(candidate_idx) if candidate_idx is not None else None

        # Convert original item IDs to 0-based indices
        result = []
        for r in recs:
            orig_iid = r.item_id
            idx = self._iid_to_idx.get(orig_iid)
            if idx is not None:
                if allowed is not None and idx not in allowed:
                    continue
                result.append(type("Rec", (), {"item_id": idx, "score": r.score})())
        return result[:top_k]


class _ColdStartBridgeRecommender:
    """Wraps ColdStartBridge for the replay_sessions interface.

    Translates between 0-based indices (used by the simulator) and
    original IDs (used by the ColdStartBridge internally).
    """

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

        # Bridge operates in index space (0-based) since that's how
        # the item features and IndexSpaceOrchidWrapper are set up.
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
            # recs are (item_idx, score) — already in index space
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


class _OrchidDirectRecommender:
    """Wraps a raw OrchidRecommender (no cold-start handling) for replay.

    Translates original item IDs from OrchidRecommender back to 0-based
    indices that the click simulator expects.
    """

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
            # Convert original item IDs to 0-based indices
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


# ---------------------------------------------------------------------------
# Incremental evaluation: NDCG@10 and Hit@10 at each interaction count
# ---------------------------------------------------------------------------

def _hit_at_k(
    recs: List[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    """1.0 if any of the top-k recommendations are in the relevant set."""
    for item in recs[:k]:
        if item in relevant:
            return 1.0
    return 0.0


def _ndcg_at_k(recs: List[int], relevant: Set[int], k: int = 10) -> float:
    """NDCG@k for a single user."""
    relevances = np.array(
        [1.0 if item in relevant else 0.0 for item in recs[:k]],
        dtype=np.float64,
    )
    if relevances.sum() == 0:
        return 0.0
    dcg = np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2, dtype=np.float64)))
    ideal = np.sort(relevances)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2, dtype=np.float64)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def bench_incremental_quality(
    data: MovieLensData,
    pop_prior: PopularityPrior,
    item_index: ItemFeatureIndex,
    orchid_rec: Any,  # raw OrchidRecommender (original-ID space)
    *,
    max_cold_interactions: int = 10,
    num_eval_users: int = 500,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Measure recommendation quality as users accumulate interactions.

    For each evaluation user, we:
    1. Start with 0 interactions (fully cold)
    2. Feed their actual interactions one-by-one
    3. After each interaction, measure Hit@10 and NDCG@10

    Returns quality curves for each system.
    """
    rng = np.random.default_rng(seed)

    # Select evaluation users: those with at least max_cold_interactions + 5 ratings
    # so we have holdout items to evaluate against
    user_counts = data.train.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= max_cold_interactions + 5].index.tolist()
    if len(eligible_users) > num_eval_users:
        eval_users = rng.choice(eligible_users, size=num_eval_users, replace=False).tolist()
    else:
        eval_users = eligible_users
    logger.info("Incremental quality: %d evaluation users", len(eval_users))

    # For each user, sort their train interactions by time and split
    # first max_cold_interactions as "cold-start phase", rest as "holdout"
    user_interactions: Dict[int, pd.DataFrame] = {}
    for uid in eval_users:
        user_df = data.train[data.train["user_id"] == uid].sort_values("timestamp")
        user_interactions[uid] = user_df

    # Systems to evaluate
    # Wrap Orchid in index-space adapter so the bridge operates in the
    # same 0-based ID space as the item feature matrix.
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
        config=ColdStartConfig(min_interactions=3, blend_until=max_cold_interactions),
    )

    # Everything operates in 0-based index space.
    # Convert relevant items to indices; feed observations as indices.
    iid_to_idx = data.item_id_to_idx

    systems = ["popularity", "content", "cold_start_bridge", "orchid_direct"]
    metrics_by_step: Dict[str, Dict[int, List[float]]] = {
        sys_name: {step: [] for step in range(max_cold_interactions + 1)}
        for sys_name in systems
    }

    content_rec = _ContentOnlyRecommender(item_index, seed_count=5)

    # Pre-compute popularity scores in index space
    all_indices = list(range(data.num_items))
    pop_scores_arr = pop_prior.scores(
        [data.idx_to_item_id.get(i, i) for i in all_indices]
    )
    pop_ranked = np.argsort(pop_scores_arr)[::-1][:10].tolist()

    for uid in eval_users:
        user_df = user_interactions[uid]
        cold_items = user_df.head(max_cold_interactions)
        holdout_items = user_df.iloc[max_cold_interactions:]

        # Relevant set in INDEX space
        relevant_idx = set()
        for orig_iid in holdout_items[holdout_items["label"] == 1]["item_id"]:
            idx = iid_to_idx.get(int(orig_iid))
            if idx is not None:
                relevant_idx.add(idx)
        if not relevant_idx:
            continue

        # Step 0: no interactions yet
        for sys_name in systems:
            if sys_name == "popularity":
                top_items = pop_ranked
            elif sys_name == "content":
                top_items = all_indices[:10]  # no history, arbitrary
            elif sys_name == "cold_start_bridge":
                # Bridge uses original user IDs, returns indices
                recs = bridge.recommend(user_id=uid, top_k=10)
                top_items = [iid for iid, _ in recs]
            else:  # orchid_direct
                try:
                    recs = orchid_rec.recommend(uid, top_k=10, filter_seen=True)
                    # Convert original item IDs to indices
                    top_items = [iid_to_idx[r.item_id] for r in recs if r.item_id in iid_to_idx]
                except (KeyError, RuntimeError):
                    top_items = []

            ndcg = _ndcg_at_k(top_items, relevant_idx)
            metrics_by_step[sys_name][0].append(ndcg)

        # Steps 1..max_cold_interactions: feed interactions one-by-one
        for step_idx, (_, row) in enumerate(cold_items.iterrows(), start=1):
            orig_item_id = int(row["item_id"])
            item_idx = iid_to_idx.get(orig_item_id)
            if item_idx is None:
                continue
            outcome = float(row["label"])

            # Record observation using INDEX space for bridge and content
            bridge.observe(uid, item_idx, outcome=outcome)
            content_rec.observe(uid, item_idx)

            for sys_name in systems:
                if sys_name == "popularity":
                    top_items = pop_ranked
                elif sys_name == "content":
                    recs = content_rec.recommend(uid, k=10)
                    top_items = recs
                elif sys_name == "cold_start_bridge":
                    recs = bridge.recommend(user_id=uid, top_k=10)
                    top_items = [iid for iid, _ in recs]
                else:  # orchid_direct
                    try:
                        recs = orchid_rec.recommend(uid, top_k=10, filter_seen=True)
                        top_items = [iid_to_idx[r.item_id] for r in recs if r.item_id in iid_to_idx]
                    except (KeyError, RuntimeError):
                        top_items = []

                ndcg = _ndcg_at_k(top_items, relevant_idx)
                metrics_by_step[sys_name][step_idx].append(ndcg)

    # Aggregate: mean NDCG@10 at each step
    results = {}
    for sys_name in systems:
        curve = {}
        for step in range(max_cold_interactions + 1):
            vals = metrics_by_step[sys_name][step]
            curve[step] = round(float(np.mean(vals)) if vals else 0.0, 4)
        results[sys_name] = curve

    return results


# ---------------------------------------------------------------------------
# Replay-based survival evaluation
# ---------------------------------------------------------------------------

def bench_cold_start_survival(
    data: MovieLensData,
    simulator: ClickSimulator,
    pop_prior: PopularityPrior,
    item_index: ItemFeatureIndex,
    orchid_rec: Any,
    *,
    max_steps: int = 30,
    num_users: int = 500,
    smoke: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run replay sessions for cold-start systems.

    Each system gets the same click simulator as the oracle.
    The ColdStartBridge observes successful clicks to build up warmth.
    """
    if smoke:
        num_users = min(num_users, 200)
        max_steps = min(max_steps, 10)

    results = {}

    # 1. Popularity-only baseline
    pop_rec = _PopularityOnlyRecommender(pop_prior, data.num_items)
    logger.info("Replaying: popularity baseline (%d users, %d steps)", num_users, max_steps)
    replay = replay_sessions(
        simulator=simulator,
        recommender=pop_rec,
        num_users=num_users,
        max_steps=max_steps,
        seed=SEED,
    )
    results["popularity"] = {
        "survival_5": round(replay["survival_5"], 4),
        "survival_10": round(replay["survival_10"], 4),
        "survival_20": round(replay["survival_20"], 4),
        "mean_session_length": round(replay["mean_session_length"], 2),
    }
    print(
        f"  popularity: surv@5={replay['survival_5']:.3f} "
        f"surv@10={replay['survival_10']:.3f} "
        f"surv@20={replay['survival_20']:.3f}"
    )

    # 2. Content-only baseline
    content_rec = _ContentOnlyRecommender(item_index, seed_count=5)
    # Content recommender needs per-step observation — wrap in custom replay
    logger.info("Replaying: content-only baseline")
    content_results = _replay_with_observation(
        simulator=simulator,
        recommender=content_rec,
        observe_fn=lambda uid, iid: content_rec.observe(uid, iid),
        num_users=num_users,
        max_steps=max_steps,
        seed=SEED,
    )
    results["content"] = {
        "survival_5": round(content_results["survival_5"], 4),
        "survival_10": round(content_results["survival_10"], 4),
        "survival_20": round(content_results["survival_20"], 4),
        "mean_session_length": round(content_results["mean_session_length"], 2),
    }
    print(
        f"  content:    surv@5={content_results['survival_5']:.3f} "
        f"surv@10={content_results['survival_10']:.3f} "
        f"surv@20={content_results['survival_20']:.3f}"
    )

    # 3. ColdStartBridge (with index-space Orchid wrapper)
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
    bridge_adapter = _ColdStartBridgeRecommender(
        bridge=bridge,
        idx_to_user_id=data.idx_to_user_id,
        idx_to_item_id=data.idx_to_item_id,
        item_id_to_idx=data.item_id_to_idx,
        num_items=data.num_items,
    )
    logger.info("Replaying: ColdStartBridge")
    bridge_results = _replay_with_observation(
        simulator=simulator,
        recommender=bridge_adapter,
        observe_fn=lambda uid, iid: bridge.observe(
            data.idx_to_user_id.get(uid, uid),
            iid,
            outcome=1.0,
        ),
        num_users=num_users,
        max_steps=max_steps,
        seed=SEED,
    )
    results["cold_start_bridge"] = {
        "survival_5": round(bridge_results["survival_5"], 4),
        "survival_10": round(bridge_results["survival_10"], 4),
        "survival_20": round(bridge_results["survival_20"], 4),
        "mean_session_length": round(bridge_results["mean_session_length"], 2),
    }
    print(
        f"  bridge:     surv@5={bridge_results['survival_5']:.3f} "
        f"surv@10={bridge_results['survival_10']:.3f} "
        f"surv@20={bridge_results['survival_20']:.3f}"
    )

    # 4. Orchid direct (no cold-start handling) — upper bound for warm users
    orchid_adapter = _OrchidDirectRecommender(
        rec=orchid_rec,
        idx_to_user_id=data.idx_to_user_id,
        idx_to_item_id=data.idx_to_item_id,
        item_id_to_idx=data.item_id_to_idx,
    )
    logger.info("Replaying: Orchid direct (no cold-start)")
    orchid_replay = replay_sessions(
        simulator=simulator,
        recommender=orchid_adapter,
        num_users=num_users,
        max_steps=max_steps,
        seed=SEED,
    )
    results["orchid_direct"] = {
        "survival_5": round(orchid_replay["survival_5"], 4),
        "survival_10": round(orchid_replay["survival_10"], 4),
        "survival_20": round(orchid_replay["survival_20"], 4),
        "mean_session_length": round(orchid_replay["mean_session_length"], 2),
    }
    print(
        f"  orchid:     surv@5={orchid_replay['survival_5']:.3f} "
        f"surv@10={orchid_replay['survival_10']:.3f} "
        f"surv@20={orchid_replay['survival_20']:.3f}"
    )

    return results


def _replay_with_observation(
    simulator: ClickSimulator,
    recommender: Any,
    observe_fn: Any,
    num_users: int,
    max_steps: int = 30,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Modified replay that calls observe_fn on successful clicks.

    This allows content-based and bridge recommenders to update their
    internal state as the session progresses.
    """
    rng = np.random.default_rng(seed)
    session_lengths = np.zeros(num_users, dtype=np.int64)

    for user_idx in range(num_users):
        seen: set[int] = set()
        length = 0

        for _step in range(max_steps):
            recs = recommender.recommend(user_idx, k=1, exclude=seen)
            if not recs:
                break

            item_idx = recs[0]
            p = simulator.click_prob(user_idx, item_idx)
            clicked = bool(rng.random() < p)

            if clicked:
                seen.add(item_idx)
                length += 1
                # Notify the recommender about the click
                observe_fn(user_idx, item_idx)
            else:
                break

        session_lengths[user_idx] = length

    return {
        "session_lengths": session_lengths,
        "survival_5": float((session_lengths >= 5).mean()),
        "survival_10": float((session_lengths >= 10).mean()),
        "survival_20": float((session_lengths >= 20).mean()),
        "mean_session_length": float(session_lengths.mean()),
    }


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

    Returns {interaction_count: mean_warmth} for charting the
    cold-to-warm transition curve.
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

    # Select users with enough interactions
    user_counts = data.train.groupby("user_id").size()
    eligible = user_counts[user_counts >= max_interactions].index.tolist()
    eval_users = rng.choice(eligible, size=min(num_users, len(eligible)), replace=False)

    warmth_by_step: Dict[int, List[float]] = {i: [] for i in range(max_interactions + 1)}

    for uid in eval_users:
        user_df = data.train[data.train["user_id"] == uid].sort_values("timestamp")
        orig_items = user_df["item_id"].tolist()[:max_interactions]
        labels = user_df["label"].tolist()[:max_interactions]

        # Convert original item IDs to indices for the bridge
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
# Main pipeline
# ---------------------------------------------------------------------------

def run(*, smoke: bool = False) -> Dict[str, Any]:
    """Execute the full cold-start benchmark."""
    t_start = time.perf_counter()
    mode = "SMOKE" if smoke else "FULL"
    logger.info("=" * 70)
    logger.info("Cold-Start Benchmark (MovieLens-1M) — %s mode", mode)
    logger.info("=" * 70)

    # Step 1: Download and preprocess
    logger.info("[1/5] Downloading MovieLens-1M...")
    ml_dir = download_and_extract()

    logger.info("[2/5] Preprocessing...")
    data = preprocess(data_dir=ml_dir, seed=SEED)
    logger.info(
        "  users=%d  items=%d  train=%d",
        data.num_users, data.num_items, len(data.train),
    )

    # Step 2: Train click simulator
    logger.info("[3/5] Training click simulator...")
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

    # Step 3: Fit Orchid recommender (ALS — fast for this benchmark)
    logger.info("[4/5] Fitting OrchidRecommender (ALS)...")
    from orchid_ranker.recommender import OrchidRecommender

    orchid_rec = OrchidRecommender(strategy="als", emb_dim=64, lr=0.01)
    orchid_rec.fit(
        data.train,
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
    )
    logger.info("  OrchidRecommender fitted.")

    # Step 4: Build shared components
    pop_prior = PopularityPrior(smoothing=1.0)
    pop_prior.fit(data.train["item_id"].tolist())

    item_index = ItemFeatureIndex(data.item_features)

    # Step 5: Run benchmarks
    logger.info("[5/5] Running cold-start benchmarks...")
    results: Dict[str, Any] = {}

    # 5a: Incremental quality curve
    print("\n=== Incremental NDCG@10 by interaction count ===")
    num_eval = 200 if smoke else 1000
    quality = bench_incremental_quality(
        data, pop_prior, item_index, orchid_rec,
        max_cold_interactions=10,
        num_eval_users=num_eval,
    )
    results["incremental_quality"] = quality

    for sys_name, curve in quality.items():
        print(f"  {sys_name:20s}: " + " → ".join(f"{v:.3f}" for v in curve.values()))

    # 5b: Session survival (headline metric)
    print("\n=== Session survival (cold-start users) ===")
    num_replay = 200 if smoke else 500
    survival = bench_cold_start_survival(
        data, simulator, pop_prior, item_index, orchid_rec,
        max_steps=10 if smoke else 30,
        num_users=num_replay,
        smoke=smoke,
    )
    results["survival"] = survival

    # 5c: Warmth transition curve
    print("\n=== Warmth transition curve ===")
    warmth = bench_warmth_curve(
        data, orchid_rec, pop_prior,
        num_users=100 if smoke else 200,
    )
    results["warmth_curve"] = warmth
    print("  warmth: " + " → ".join(
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
        "simulator_epochs": sim_epochs,
        "cold_start_config": {
            "min_interactions": 3,
            "blend_until": 20,
            "popularity_weight": 0.3,
        },
    }
    results["config"] = config

    # Write JSON
    out_path = OUTPUT_DIR / "results_cold_start.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", out_path)

    # Print markdown tables
    _print_markdown(results)

    elapsed = time.perf_counter() - t_start
    logger.info("Cold-start benchmark complete in %.1fs (%.1f min)", elapsed, elapsed / 60)
    print(f"\nCompleted in {elapsed:.1f}s. Results: {out_path}")

    return results


def _print_markdown(results: Dict[str, Any]) -> None:
    """Print Markdown tables for docs."""
    survival = results.get("survival", {})
    quality = results.get("incremental_quality", {})

    # Survival table
    print("\n### Session survival (cold-start users)\n")
    print("| System | Surv@5 | Surv@10 | Surv@20 | Mean session |")
    print("|--------|--------|---------|---------|-------------|")
    for sys_name in ["popularity", "content", "cold_start_bridge", "orchid_direct"]:
        row = survival.get(sys_name, {})
        print(
            f"| {sys_name:<20s} "
            f"| {row.get('survival_5', 0):.4f} "
            f"| {row.get('survival_10', 0):.4f} "
            f"| {row.get('survival_20', 0):.4f} "
            f"| {row.get('mean_session_length', 0):.2f} |"
        )

    # Quality curve table
    print("\n### NDCG@10 by interaction count\n")
    steps = list(range(11))
    header = "| System | " + " | ".join(f"N={s}" for s in steps) + " |"
    sep = "|--------|" + "|".join(["------"] * len(steps)) + "|"
    print(header)
    print(sep)
    for sys_name in ["popularity", "content", "cold_start_bridge", "orchid_direct"]:
        curve = quality.get(sys_name, {})
        vals = " | ".join(f"{curve.get(s, curve.get(str(s), 0)):.4f}" for s in steps)
        print(f"| {sys_name:<20s} | {vals} |")

    # Warmth curve
    warmth = results.get("warmth_curve", {})
    if warmth:
        print("\n### Warmth transition\n")
        print("| Interactions | Mean warmth |")
        print("|-------------|------------|")
        for step, w in sorted(warmth.items(), key=lambda x: int(x[0])):
            print(f"| {int(step):>11d} | {w:.4f} |")


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
        prog="python benchmarks/cold_start_bench.py",
        description="Cold-start benchmark on MovieLens-1M.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test (~2 min).",
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
