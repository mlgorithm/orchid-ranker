#!/usr/bin/env python3
"""Benchmark: orchid_ranker.curated_feed module.

Evaluates the curated feed ranker's ability to surface progression-aware
content for editorially curated publications.  Uses synthetic data that
models a newsletter / technical publication with topic hierarchy,
freshness decay, and reader learning arcs.

The headline metrics are **retention uplift** (do users stay longer?) and
**diversity** (does the feed avoid topic monoculture?).

Usage::

    # Full run (~5 min)
    PYTHONPATH=src python benchmarks/curated_feed_bench.py

    # Smoke test (~30 sec)
    PYTHONPATH=src python benchmarks/curated_feed_bench.py --smoke

    # When MIND dataset is available:
    PYTHONPATH=src python benchmarks/curated_feed_bench.py \\
        --data-path /path/to/mind/

Outputs ``benchmarks/results_curated_feed.json`` and a Markdown table.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orchid_ranker.curated_feed import (
    FeedItem,
    FeedRanker,
    FreshnessScorer,
)

logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

@dataclass
class SyntheticFeedData:
    """Container for synthetic curated-feed evaluation data."""
    topics: List[str]
    items: List[FeedItem]
    users: List[int]
    # Per-user interaction history: user_id -> [(item, engaged, timestamp)]
    interactions: Dict[int, List[Tuple[FeedItem, bool, float]]]
    train_items: List[FeedItem]
    test_items: List[FeedItem]


def _generate_synthetic_feed_data(
    num_users: int = 100,
    num_items: int = 500,
    num_topics: int = 8,
    interactions_per_user: int = 40,
    seed: int = SEED,
) -> SyntheticFeedData:
    """Generate synthetic curated-feed data.

    Simulates a technical publication with:
    - Topics at varying difficulty levels
    - Users who progressively engage with harder content
    - Freshness decay (older items are less engaging)
    - Topic fatigue (too many same-topic items in a row reduces engagement)
    """
    rng = np.random.default_rng(seed)

    topics = [f"topic_{i}" for i in range(num_topics)]
    users = list(range(num_users))

    # Create items with varying difficulty and timestamps
    now = time.time()
    items = []
    for iid in range(num_items):
        topic = topics[iid % num_topics]
        # Difficulty increases within each topic as items accumulate
        within_topic_idx = iid // num_topics
        difficulty = min(1.0, within_topic_idx / (num_items / num_topics) + rng.normal(0, 0.05))
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        # Timestamps spread over 30 days, newer items are later
        age_hours = rng.uniform(0, 720)  # 0 to 30 days
        timestamp = now - age_hours * 3600
        items.append(FeedItem(
            item_id=iid,
            topic=topic,
            difficulty=difficulty,
            timestamp=timestamp,
        ))

    # Generate user interaction histories
    interactions: Dict[int, List[Tuple[FeedItem, bool, float]]] = defaultdict(list)

    for uid in users:
        # User has a primary topic interest and a reading level
        primary_topic_idx = uid % num_topics
        reading_level = 0.2  # start as novice

        recent_topics: list[str] = []

        for step in range(interactions_per_user):
            # Pick candidates: user sees some items from different topics
            candidate_pool = rng.choice(items, size=min(20, len(items)), replace=False)

            # Simulate engagement:
            # Higher if: topic matches interest, difficulty matches level,
            #            item is fresh, not repetitive topic
            best_item = None
            best_score = -1.0
            for item in candidate_pool:
                # Topic match
                topic_match = 1.0 if item.topic == topics[primary_topic_idx] else 0.3

                # Difficulty match (Gaussian around reading level)
                diff_fit = math.exp(-2.0 * (item.difficulty - reading_level) ** 2)

                # Freshness
                age_h = (now - item.timestamp) / 3600
                freshness = math.exp(-age_h / 24)

                # Topic fatigue
                recent_same = sum(1 for t in recent_topics[-3:] if t == item.topic)
                fatigue = 1.0 - 0.3 * recent_same

                score = topic_match * diff_fit * freshness * fatigue
                if score > best_score:
                    best_score = score
                    best_item = item

            if best_item is None:
                continue

            # Engagement probability
            engaged = rng.random() < min(best_score * 0.8, 0.95)

            interactions[uid].append((best_item, bool(engaged), now - (interactions_per_user - step) * 3600))

            if engaged:
                reading_level = min(reading_level + 0.02, 1.0)
            recent_topics.append(best_item.topic)
            if len(recent_topics) > 5:
                recent_topics.pop(0)

    # Split items: 80% train, 20% test (by timestamp)
    sorted_items = sorted(items, key=lambda x: x.timestamp)
    split_idx = int(len(sorted_items) * 0.8)
    train_items = sorted_items[:split_idx]
    test_items = sorted_items[split_idx:]

    return SyntheticFeedData(
        topics=topics,
        items=items,
        users=users,
        interactions=interactions,
        train_items=train_items,
        test_items=test_items,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class _FreshnessOnlyRanker:
    """Baseline: ranks purely by recency."""

    def rank(
        self, user_id: int, candidates: List[FeedItem], top_k: int = 20,
    ) -> List[Tuple[int, float]]:
        scored = [(item.item_id, item.timestamp) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class _PopularityFeedBaseline:
    """Baseline: ranks by global engagement rate."""

    def __init__(self) -> None:
        self._scores: Dict[int, float] = {}

    def fit(self, interactions: Dict[int, List[Tuple[FeedItem, bool, float]]]) -> None:
        counts: Dict[int, int] = defaultdict(int)
        engagements: Dict[int, int] = defaultdict(int)
        for uid, ixs in interactions.items():
            for item, engaged, _ts in ixs:
                counts[item.item_id] += 1
                if engaged:
                    engagements[item.item_id] += 1
        self._scores = {
            iid: engagements.get(iid, 0) / max(counts.get(iid, 1), 1)
            for iid in counts
        }

    def rank(
        self, user_id: int, candidates: List[FeedItem], top_k: int = 20,
    ) -> List[Tuple[int, float]]:
        scored = [(item.item_id, self._scores.get(item.item_id, 0.0)) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _topic_diversity(ranked_items: List[FeedItem], k: int = 10) -> float:
    """Fraction of unique topics in the top-k recommendations."""
    topics = set()
    for item in ranked_items[:k]:
        topics.add(item.topic)
    return len(topics) / max(k, 1)


def _engagement_rate(
    ranker: Any,
    data: SyntheticFeedData,
    *,
    top_k: int = 10,
    is_feed_ranker: bool = False,
) -> Dict[str, float]:
    """Simulate engagement: for each user, rank test items and check engagement.

    Uses the user's actual interaction patterns to estimate engagement.
    """
    engagement_counts = []
    diversity_scores = []
    session_lengths = []

    rng = np.random.default_rng(SEED + 1)

    for uid in data.users:
        # Rank test items
        candidates = data.test_items
        if not candidates:
            continue

        if is_feed_ranker:
            scored = ranker.rank(user_id=uid, candidates=candidates, top_k=top_k)
            ranked = [s.item for s in scored]
        else:
            scored_pairs = ranker.rank(uid, candidates, top_k=top_k)
            item_map = {item.item_id: item for item in candidates}
            ranked = [item_map[iid] for iid, _ in scored_pairs if iid in item_map]

        if not ranked:
            continue

        # Simulate engagement based on user's reading level (from history)
        user_ixs = data.interactions.get(uid, [])
        n_engaged = sum(1 for _, e, _ in user_ixs if e)
        reading_level = min(0.2 + 0.02 * n_engaged, 1.0)

        engaged = 0
        session_len = 0
        for item in ranked[:top_k]:
            # Engagement probability based on difficulty match + freshness
            diff_fit = math.exp(-2.0 * (item.difficulty - reading_level) ** 2)
            age_h = max(0, (time.time() - item.timestamp) / 3600)
            freshness = math.exp(-age_h / 24)
            p_engage = min(diff_fit * freshness * 0.8, 0.95)

            if rng.random() < p_engage:
                engaged += 1
                session_len += 1
            else:
                break  # session ends on first non-engagement

        engagement_counts.append(engaged / max(len(ranked[:top_k]), 1))
        diversity_scores.append(_topic_diversity(ranked, k=top_k))
        session_lengths.append(session_len)

    return {
        "mean_engagement_rate": round(float(np.mean(engagement_counts)) if engagement_counts else 0.0, 4),
        "mean_diversity": round(float(np.mean(diversity_scores)) if diversity_scores else 0.0, 4),
        "mean_session_length": round(float(np.mean(session_lengths)) if session_lengths else 0.0, 2),
        "survival_5": round(float(np.mean([1 if s >= 5 else 0 for s in session_lengths])) if session_lengths else 0.0, 4),
        "survival_10": round(float(np.mean([1 if s >= 10 else 0 for s in session_lengths])) if session_lengths else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_feed_ranking(
    data: SyntheticFeedData,
    *,
    smoke: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compare feed ranking systems on engagement and diversity."""
    results = {}
    top_k = 10

    # 1. Freshness-only baseline
    fresh_ranker = _FreshnessOnlyRanker()
    fresh_metrics = _engagement_rate(fresh_ranker, data, top_k=top_k)
    results["freshness_only"] = fresh_metrics
    print(
        f"  freshness_only: engagement={fresh_metrics['mean_engagement_rate']:.4f}  "
        f"diversity={fresh_metrics['mean_diversity']:.4f}  "
        f"session={fresh_metrics['mean_session_length']:.2f}"
    )

    # 2. Popularity baseline
    pop_ranker = _PopularityFeedBaseline()
    pop_ranker.fit(data.interactions)
    pop_metrics = _engagement_rate(pop_ranker, data, top_k=top_k)
    results["popularity"] = pop_metrics
    print(
        f"  popularity:     engagement={pop_metrics['mean_engagement_rate']:.4f}  "
        f"diversity={pop_metrics['mean_diversity']:.4f}  "
        f"session={pop_metrics['mean_session_length']:.2f}"
    )

    # 3. FeedRanker (progression-aware)
    feed_ranker = FeedRanker(
        freshness=FreshnessScorer(halflife_hours=12),
        w_relevance=0.3,
        w_freshness=0.25,
        w_stretch=0.2,
        w_diversity=0.15,
        w_competence=0.1,
    )

    # Train: feed the ranker user interaction histories
    for uid, ixs in data.interactions.items():
        for item, engaged, _ts in ixs:
            feed_ranker.observe(uid, item, engaged)

    feed_metrics = _engagement_rate(feed_ranker, data, top_k=top_k, is_feed_ranker=True)
    results["curated_feed_ranker"] = feed_metrics
    print(
        f"  curated_feed:   engagement={feed_metrics['mean_engagement_rate']:.4f}  "
        f"diversity={feed_metrics['mean_diversity']:.4f}  "
        f"session={feed_metrics['mean_session_length']:.2f}"
    )

    # Compute uplifts
    pop_eng = pop_metrics["mean_engagement_rate"]
    feed_eng = feed_metrics["mean_engagement_rate"]
    engagement_uplift = ((feed_eng - pop_eng) / max(pop_eng, 0.001)) * 100

    pop_div = pop_metrics["mean_diversity"]
    feed_div = feed_metrics["mean_diversity"]
    diversity_uplift = ((feed_div - pop_div) / max(pop_div, 0.001)) * 100

    results["uplift_vs_popularity"] = {
        "engagement_uplift_pct": round(engagement_uplift, 1),
        "diversity_uplift_pct": round(diversity_uplift, 1),
    }
    print(f"\n  Engagement uplift vs popularity: {engagement_uplift:+.1f}%")
    print(f"  Diversity uplift vs popularity:  {diversity_uplift:+.1f}%")

    return results


def bench_freshness_sensitivity(
    data: SyntheticFeedData,
) -> Dict[str, Dict[str, float]]:
    """Test different freshness half-lives to find the sweet spot."""
    results = {}

    for halflife in [6, 12, 24, 48, 168]:
        ranker = FeedRanker(
            freshness=FreshnessScorer(halflife_hours=halflife),
            w_freshness=0.25,
        )
        # Train
        for uid, ixs in data.interactions.items():
            for item, engaged, _ts in ixs:
                ranker.observe(uid, item, engaged)

        metrics = _engagement_rate(ranker, data, top_k=10, is_feed_ranker=True)
        results[f"halflife_{halflife}h"] = metrics
        print(f"  halflife={halflife:>3d}h: engagement={metrics['mean_engagement_rate']:.4f}")

    return results


def bench_weight_sensitivity(
    data: SyntheticFeedData,
) -> Dict[str, Dict[str, float]]:
    """Test different weight configurations."""
    configs = {
        "relevance_heavy": {"w_relevance": 0.5, "w_freshness": 0.2, "w_stretch": 0.1, "w_diversity": 0.1, "w_competence": 0.1},
        "freshness_heavy": {"w_relevance": 0.2, "w_freshness": 0.4, "w_stretch": 0.15, "w_diversity": 0.15, "w_competence": 0.1},
        "balanced": {"w_relevance": 0.3, "w_freshness": 0.25, "w_stretch": 0.2, "w_diversity": 0.15, "w_competence": 0.1},
        "diversity_heavy": {"w_relevance": 0.2, "w_freshness": 0.2, "w_stretch": 0.15, "w_diversity": 0.35, "w_competence": 0.1},
        "stretch_heavy": {"w_relevance": 0.2, "w_freshness": 0.2, "w_stretch": 0.35, "w_diversity": 0.15, "w_competence": 0.1},
    }

    results = {}
    for name, weights in configs.items():
        ranker = FeedRanker(
            freshness=FreshnessScorer(halflife_hours=12),
            **weights,
        )
        for uid, ixs in data.interactions.items():
            for item, engaged, _ts in ixs:
                ranker.observe(uid, item, engaged)

        metrics = _engagement_rate(ranker, data, top_k=10, is_feed_ranker=True)
        results[name] = metrics
        print(
            f"  {name:<18s}: engagement={metrics['mean_engagement_rate']:.4f}  "
            f"diversity={metrics['mean_diversity']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# MIND loader
# ---------------------------------------------------------------------------

def _try_load_mind(
    data_path: str,
    *,
    smoke: bool = False,
) -> Optional[SyntheticFeedData]:
    """Try loading MIND data, returning None on failure.

    If *data_path* is ``"auto"``, attempts to download via the pipeline.
    Otherwise treats it as a path to an extracted MIND split directory.
    """
    try:
        from mind.preprocess import preprocess as mind_preprocess
    except ImportError:
        logger.warning("mind.preprocess not importable; using synthetic data.")
        return None

    mind_dir = Path(data_path)

    if data_path == "auto":
        try:
            from mind.download import download_mind
            mind_dir = download_mind(split="train")
        except Exception as exc:
            logger.warning("MIND download failed: %s. Using synthetic data.", exc)
            return None

    if not mind_dir.is_dir():
        logger.warning("MIND directory not found: %s. Using synthetic data.", mind_dir)
        return None

    try:
        max_users = 500 if smoke else None
        max_items = 5000 if smoke else None
        mind_data = mind_preprocess(mind_dir, max_users=max_users, max_items=max_items)

        logger.info(
            "MIND loaded: %d users, %d items, %d topics, %d interactions",
            mind_data.num_users, mind_data.num_items,
            len(mind_data.topics),
            sum(len(v) for v in mind_data.interactions.values()),
        )

        # Convert to SyntheticFeedData schema
        return SyntheticFeedData(
            topics=mind_data.topics,
            items=mind_data.items,
            users=mind_data.users,
            interactions=mind_data.interactions,
            train_items=mind_data.train_items,
            test_items=mind_data.test_items,
        )
    except Exception as exc:
        logger.warning("MIND preprocessing failed: %s. Using synthetic data.", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(*, smoke: bool = False, data_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute the curated-feed benchmark."""
    t_start = time.perf_counter()
    mode = "SMOKE" if smoke else "FULL"
    logger.info("=" * 70)
    logger.info("Curated Feed Benchmark — %s mode", mode)
    logger.info("=" * 70)

    data: Optional[SyntheticFeedData] = None
    data_source = "synthetic"

    if data_path:
        data = _try_load_mind(data_path, smoke=smoke)
        if data is not None:
            data_source = f"mind:{data_path}"

    if data is None:
        num_users = 50 if smoke else 200
        num_items = 200 if smoke else 1000
        interactions_per_user = 20 if smoke else 50

        logger.info(
            "Generating synthetic data: %d users, %d items, %d interactions/user",
            num_users, num_items, interactions_per_user,
        )
        data = _generate_synthetic_feed_data(
            num_users=num_users,
            num_items=num_items,
            interactions_per_user=interactions_per_user,
        )

    results: Dict[str, Any] = {}

    # Main comparison
    print("\n=== Feed ranking comparison ===")
    results["ranking"] = bench_feed_ranking(data, smoke=smoke)

    # Freshness sensitivity
    print("\n=== Freshness half-life sensitivity ===")
    results["freshness_sensitivity"] = bench_freshness_sensitivity(data)

    # Weight sensitivity
    if not smoke:
        print("\n=== Weight sensitivity ===")
        results["weight_sensitivity"] = bench_weight_sensitivity(data)

    # Config
    import datetime
    results["config"] = {
        "seed": SEED,
        "smoke": smoke,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_source": data_source,
        "num_users": len(data.users),
        "num_items": len(data.items),
    }

    # Write JSON
    out_path = OUTPUT_DIR / "results_curated_feed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", out_path)

    _print_markdown(results)

    elapsed = time.perf_counter() - t_start
    print(f"\nCompleted in {elapsed:.1f}s. Results: {out_path}")
    return results


def _print_markdown(results: Dict[str, Any]) -> None:
    """Print Markdown tables for docs."""
    ranking = results.get("ranking", {})

    print("\n### Feed ranking comparison\n")
    print("| System | Engagement | Diversity | Session length | Surv@5 | Surv@10 |")
    print("|--------|-----------|----------|---------------|--------|---------|")
    for sys_name in ["freshness_only", "popularity", "curated_feed_ranker"]:
        row = ranking.get(sys_name, {})
        print(
            f"| {sys_name:<20s} "
            f"| {row.get('mean_engagement_rate', 0):.4f} "
            f"| {row.get('mean_diversity', 0):.4f} "
            f"| {row.get('mean_session_length', 0):>13.2f} "
            f"| {row.get('survival_5', 0):.4f} "
            f"| {row.get('survival_10', 0):.4f} |"
        )

    uplift = ranking.get("uplift_vs_popularity", {})
    print(f"\n**Engagement uplift vs popularity:** {uplift.get('engagement_uplift_pct', 0):+.1f}%")
    print(f"**Diversity uplift vs popularity:** {uplift.get('diversity_uplift_pct', 0):+.1f}%")


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
        prog="python benchmarks/curated_feed_bench.py",
        description="Curated feed benchmark (synthetic / MIND).",
    )
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to MIND dataset.")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED}).")
    args = parser.parse_args()

    SEED = args.seed

    run(smoke=args.smoke, data_path=args.data_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
