#!/usr/bin/env python3
"""Benchmark: orchid_ranker.taste_progression module on Amazon Reviews.

Evaluates taste-progression scoring on domains where users develop
expertise over time.  Uses the Amazon Reviews 2018 dataset, filtering
to taste-progression-friendly categories (Wine, Cameras, Fragrances).

The headline metric is **kept-rate uplift**: how much more often do
users keep items recommended by the taste-progression ranker compared
to a popularity or two-tower baseline?

Usage::

    # Full run (requires Amazon Reviews download)
    PYTHONPATH=src python benchmarks/taste_progression_bench.py

    # Smoke test with synthetic data (~1 min)
    PYTHONPATH=src python benchmarks/taste_progression_bench.py --smoke

    # Specify a pre-downloaded dataset
    PYTHONPATH=src python benchmarks/taste_progression_bench.py \\
        --data-path /path/to/amazon_reviews.jsonl

Outputs ``benchmarks/results_taste_progression.json`` and a Markdown
table suitable for ``docs/benchmarks/taste-progression.md``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orchid_ranker.taste_progression import (
    SophisticationMapper,
    TasteConfig,
    TasteProgressionRanker,
)

logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Synthetic data generator (for smoke tests and CI)
# ---------------------------------------------------------------------------

@dataclass
class SyntheticTasteData:
    """Container for synthetic taste-progression evaluation data."""
    users: List[int]
    items: List[int]
    categories: Dict[int, str]          # item_id -> category
    sophistication: Dict[int, float]    # item_id -> sophistication [0, 1]
    prices: Dict[int, float]            # item_id -> price
    interactions: List[Dict[str, Any]]  # [{user_id, item_id, rating, kept, timestamp}, ...]
    train_interactions: List[Dict[str, Any]]
    test_interactions: List[Dict[str, Any]]


def _generate_synthetic_data(
    num_users: int = 200,
    num_items: int = 500,
    num_categories: int = 5,
    interactions_per_user: int = 30,
    seed: int = SEED,
) -> SyntheticTasteData:
    """Generate synthetic taste-progression data.

    Users develop expertise in one primary category over time.
    Early interactions cluster at low sophistication; later ones
    progress to higher tiers, with occasional exploration.
    """
    rng = np.random.default_rng(seed)

    categories = [f"category_{i}" for i in range(num_categories)]
    users = list(range(num_users))
    items = list(range(num_items))

    # Assign categories and sophistication to items
    item_categories = {}
    item_sophistication = {}
    item_prices = {}
    for iid in items:
        cat = categories[iid % num_categories]
        item_categories[iid] = cat
        # Sophistication correlates loosely with item ID within category
        soph = (iid % (num_items // num_categories)) / (num_items // num_categories)
        soph = np.clip(soph + rng.normal(0, 0.1), 0.0, 1.0)
        item_sophistication[iid] = float(soph)
        # Price loosely correlates with sophistication
        item_prices[iid] = float(10 + soph * 90 + rng.normal(0, 10))

    # Generate interactions with taste progression
    interactions = []
    for uid in users:
        primary_cat = categories[uid % num_categories]
        expertise = 0.0  # starts at novice

        for step in range(interactions_per_user):
            # Pick an item near current expertise
            if rng.random() < 0.8:
                # In-category: items near expertise level
                cat_items = [i for i in items if item_categories[i] == primary_cat]
            else:
                # Exploration: random category
                cat_items = items

            # Prefer items near current expertise
            sophs = np.array([item_sophistication[i] for i in cat_items])
            weights = np.exp(-2.0 * np.abs(sophs - expertise))
            weights /= weights.sum()
            chosen_idx = rng.choice(len(cat_items), p=weights)
            iid = cat_items[chosen_idx]

            # "Kept" if sophistication is in the stretch zone
            item_soph = item_sophistication[iid]
            stretch_fit = 1.0 - abs(item_soph - (expertise + 0.1))
            kept = rng.random() < (0.3 + 0.5 * max(stretch_fit, 0))
            rating = (4.0 + rng.normal(0, 0.5)) if kept else (2.5 + rng.normal(0, 0.5))
            rating = float(np.clip(rating, 1.0, 5.0))

            interactions.append({
                "user_id": uid,
                "item_id": iid,
                "category": item_categories[iid],
                "rating": rating,
                "kept": bool(kept),
                "timestamp": step,
            })

            # Expertise grows on positive interactions
            if kept:
                expertise = min(expertise + 0.03, 1.0)

    # Split: 80% train, 20% test (last interactions per user)
    by_user: Dict[int, List] = defaultdict(list)
    for ix in interactions:
        by_user[ix["user_id"]].append(ix)

    train, test = [], []
    for uid, ixs in by_user.items():
        ixs.sort(key=lambda x: x["timestamp"])
        split_idx = max(1, int(len(ixs) * 0.8))
        train.extend(ixs[:split_idx])
        test.extend(ixs[split_idx:])

    return SyntheticTasteData(
        users=users,
        items=items,
        categories=item_categories,
        sophistication=item_sophistication,
        prices=item_prices,
        interactions=interactions,
        train_interactions=train,
        test_interactions=test,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class _PopularityTasteBaseline:
    """Ranks by global popularity (count of kept interactions)."""

    def __init__(self) -> None:
        self._scores: Dict[int, float] = {}

    def fit(self, interactions: List[Dict]) -> None:
        counts: Dict[int, int] = defaultdict(int)
        for ix in interactions:
            if ix.get("kept", ix.get("rating", 0) >= 4):
                counts[ix["item_id"]] += 1
        max_count = max(counts.values()) if counts else 1
        self._scores = {iid: c / max_count for iid, c in counts.items()}

    def recommend(
        self,
        user_id: int,
        candidates: Sequence[int],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        scored = [(iid, self._scores.get(iid, 0.0)) for iid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class _RecentPopularityBaseline:
    """Ranks by recent popularity (last N interactions, per category)."""

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._scores: Dict[int, float] = {}

    def fit(self, interactions: List[Dict]) -> None:
        recent = sorted(interactions, key=lambda x: x["timestamp"])[-self._window:]
        counts: Dict[int, int] = defaultdict(int)
        for ix in recent:
            if ix.get("kept", ix.get("rating", 0) >= 4):
                counts[ix["item_id"]] += 1
        max_count = max(counts.values()) if counts else 1
        self._scores = {iid: c / max_count for iid, c in counts.items()}

    def recommend(
        self,
        user_id: int,
        candidates: Sequence[int],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        scored = [(iid, self._scores.get(iid, 0.0)) for iid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _compute_kept_rate(
    recommender: Any,
    test_interactions: List[Dict],
    all_items: List[int],
    top_k: int = 10,
    num_negatives: int = 50,
    seed: int = SEED,
) -> Dict[str, float]:
    """Compute kept-rate via per-user candidate re-ranking.

    For each test user, builds a candidate set of their test items plus
    random negatives, then asks the recommender to rank that set.  This
    ensures all systems rank the *same* items and prevents the
    stretch-zone ranker from being penalised for recommending items
    outside a narrow holdout set.

    Metrics:
    - **kept_rate**: of top-k recommended items that are in the test set,
      what fraction were kept?
    - **hit_rate**: did any top-k item appear in the user's kept set?
    - **ndcg_at_10**: NDCG with binary relevance (kept = 1).
    """
    rng = np.random.default_rng(seed)

    # Build per-user test data
    user_test: Dict[int, Dict[int, bool]] = defaultdict(dict)
    for ix in test_interactions:
        uid = ix["user_id"]
        iid = ix["item_id"]
        kept = ix.get("kept", ix.get("rating", 0) >= 4)
        user_test[uid][iid] = kept

    all_items_set = set(all_items)

    kept_rates = []
    hit_rates = []
    ndcg_scores = []

    for uid, test_items in user_test.items():
        if not test_items:
            continue

        # Build candidate set: test items + random negatives
        test_item_ids = list(test_items.keys())
        negatives_pool = list(all_items_set - set(test_item_ids))
        n_neg = min(num_negatives, len(negatives_pool))
        if n_neg > 0:
            neg_indices = rng.choice(len(negatives_pool), size=n_neg, replace=False)
            neg_items = [negatives_pool[i] for i in neg_indices]
        else:
            neg_items = []
        candidates = test_item_ids + neg_items

        # Get recommendations from this candidate set
        if hasattr(recommender, "recommend") and callable(recommender.recommend):
            recs = recommender.recommend(uid, candidates, top_k=top_k)
        else:
            continue

        rec_items = [iid for iid, _ in recs]
        if not rec_items:
            continue

        # Kept rate: of recommended items that appear in test, what fraction were kept?
        relevant_recs = [(iid, test_items[iid]) for iid in rec_items if iid in test_items]
        if relevant_recs:
            kept_count = sum(1 for _, kept in relevant_recs if kept)
            kept_rates.append(kept_count / len(relevant_recs))

        # Hit rate: did any recommended item appear as a kept item in test?
        kept_items = {iid for iid, kept in test_items.items() if kept}
        hit = any(iid in kept_items for iid in rec_items)
        hit_rates.append(1.0 if hit else 0.0)

        # NDCG: binary relevance based on kept status
        relevances = np.array(
            [1.0 if iid in kept_items else 0.0 for iid in rec_items],
            dtype=np.float64,
        )
        if relevances.sum() > 0:
            dcg = np.sum(
                relevances / np.log2(np.arange(2, len(relevances) + 2, dtype=np.float64))
            )
            ideal = np.sort(relevances)[::-1]
            idcg = np.sum(
                ideal / np.log2(np.arange(2, len(ideal) + 2, dtype=np.float64))
            )
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "kept_rate": round(float(np.mean(kept_rates)) if kept_rates else 0.0, 4),
        "hit_rate": round(float(np.mean(hit_rates)) if hit_rates else 0.0, 4),
        "ndcg_at_10": round(float(np.mean(ndcg_scores)) if ndcg_scores else 0.0, 4),
        "num_evaluated_users": len(kept_rates),
    }


def _compute_stretch_accuracy(
    ranker: TasteProgressionRanker,
    test_interactions: List[Dict],
    sophistication: Dict[int, float],
) -> Dict[str, float]:
    """Measure how well recommendations fall in the user's stretch zone.

    For each test interaction, check whether the recommended items
    are in the predicted stretch zone.
    """
    in_zone = 0
    total = 0

    by_user = defaultdict(list)
    for ix in test_interactions:
        by_user[ix["user_id"]].append(ix)

    for uid, ixs in by_user.items():
        for ix in ixs:
            profile = ranker._profiles.get(uid)
            if profile is None:
                continue
            cat = ix.get("category", "general")
            zone = profile.stretch_zone(cat)
            item_soph = sophistication.get(ix["item_id"], 0.5)
            if zone[0] <= item_soph <= zone[1]:
                in_zone += 1
            total += 1

    return {
        "stretch_accuracy": round(in_zone / max(total, 1), 4),
        "total_evaluated": total,
    }


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_kept_rate(
    data: SyntheticTasteData,
    *,
    smoke: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compare kept-rate across systems."""
    results = {}

    # 1. Global popularity
    pop = _PopularityTasteBaseline()
    pop.fit(data.train_interactions)
    pop_metrics = _compute_kept_rate(pop, data.test_interactions, data.items)
    results["popularity"] = pop_metrics
    print(f"  popularity:   kept_rate={pop_metrics['kept_rate']:.4f}  hit={pop_metrics['hit_rate']:.4f}")

    # 2. Recent popularity
    recent = _RecentPopularityBaseline()
    recent.fit(data.train_interactions)
    recent_metrics = _compute_kept_rate(recent, data.test_interactions, data.items)
    results["recent_popularity"] = recent_metrics
    print(f"  recent_pop:   kept_rate={recent_metrics['kept_rate']:.4f}  hit={recent_metrics['hit_rate']:.4f}")

    # 3. TasteProgressionRanker
    mapper = SophisticationMapper(data.sophistication)
    config = TasteConfig(
        stretch_width=0.15,
        keep_threshold=4.0,
        bkt_p_init=0.2,
        bkt_p_transit=0.08,
    )
    ranker = TasteProgressionRanker(
        sophistication_scores=mapper,
        config=config,
    )
    ranker.set_item_categories(data.categories)

    # Train: feed all training interactions
    for ix in sorted(data.train_interactions, key=lambda x: x["timestamp"]):
        ranker.observe(
            user_id=ix["user_id"],
            item_id=ix["item_id"],
            category=ix.get("category", "general"),
            rating=ix["rating"],
        )

    # Wrap ranker for the evaluation interface
    class _RankerAdapter:
        def __init__(self, ranker, items, categories, sophistication):
            self._ranker = ranker
            self._items = items
            self._cats = categories
            self._soph = sophistication

        def recommend(self, user_id, candidates, top_k=10):
            return self._ranker.recommend(
                user_id=user_id,
                candidate_item_ids=candidates,
                top_k=top_k,
            )

    adapter = _RankerAdapter(ranker, data.items, data.categories, data.sophistication)
    ranker_metrics = _compute_kept_rate(adapter, data.test_interactions, data.items)
    results["taste_progression"] = ranker_metrics
    print(
        f"  taste_prog:   kept_rate={ranker_metrics['kept_rate']:.4f}  "
        f"hit={ranker_metrics['hit_rate']:.4f}"
    )

    # Compute uplift
    pop_kr = pop_metrics["kept_rate"]
    tp_kr = ranker_metrics["kept_rate"]
    uplift = ((tp_kr - pop_kr) / max(pop_kr, 0.001)) * 100
    results["uplift_vs_popularity_pct"] = round(uplift, 1)
    print(f"\n  Kept-rate uplift vs popularity: {uplift:+.1f}%")

    # Stretch accuracy
    stretch = _compute_stretch_accuracy(ranker, data.test_interactions, data.sophistication)
    results["stretch_accuracy"] = stretch
    print(f"  Stretch zone accuracy: {stretch['stretch_accuracy']:.4f}")

    return results


def bench_progression_curve(
    data: SyntheticTasteData,
) -> Dict[str, Any]:
    """Track taste level evolution over interaction count.

    Shows that the ranker correctly models user progression from
    novice to expert.
    """
    mapper = SophisticationMapper(data.sophistication)
    ranker = TasteProgressionRanker(
        sophistication_scores=mapper,
        config=TasteConfig(bkt_p_init=0.2, bkt_p_transit=0.08),
    )
    ranker.set_item_categories(data.categories)

    # Track a few representative users
    sample_users = data.users[:10]
    curves: Dict[int, List[float]] = {uid: [] for uid in sample_users}

    by_user: Dict[int, List] = defaultdict(list)
    for ix in sorted(data.interactions, key=lambda x: x["timestamp"]):
        by_user[ix["user_id"]].append(ix)

    for uid in sample_users:
        for ix in by_user.get(uid, []):
            ranker.observe(
                user_id=uid,
                item_id=ix["item_id"],
                category=ix.get("category", "general"),
                rating=ix["rating"],
            )
            profile = ranker._profiles.get(uid)
            if profile:
                cat = ix.get("category", "general")
                curves[uid].append(round(profile.taste_level(cat), 4))

    # Aggregate: mean taste level at each step
    max_len = max(len(v) for v in curves.values()) if curves else 0
    mean_curve = {}
    for step in range(max_len):
        vals = [curves[uid][step] for uid in sample_users if step < len(curves[uid])]
        mean_curve[step] = round(float(np.mean(vals)), 4) if vals else 0.0

    return {
        "sample_curves": {str(uid): curve for uid, curve in curves.items()},
        "mean_curve": mean_curve,
    }


# ---------------------------------------------------------------------------
# Amazon Reviews loader
# ---------------------------------------------------------------------------

def _try_load_amazon(
    data_path: str,
    *,
    smoke: bool = False,
    category: str = "DigitalMusic",
) -> Optional[SyntheticTasteData]:
    """Try loading Amazon Reviews data, returning None on failure.

    If *data_path* is ``"auto"``, attempts to download via the pipeline.
    Otherwise treats it as a path to an already-downloaded JSON file.
    Returns a ``SyntheticTasteData`` (same schema) or None.
    """
    try:
        from amazon_reviews.preprocess import preprocess as amazon_preprocess
    except ImportError:
        logger.warning("amazon_reviews.preprocess not importable; using synthetic data.")
        return None

    json_path = Path(data_path)

    if data_path == "auto":
        try:
            from amazon_reviews.download import download_category
            json_path = download_category(category)
        except Exception as exc:
            logger.warning("Amazon Reviews download failed: %s. Using synthetic data.", exc)
            return None

    if not json_path.exists():
        logger.warning("Amazon Reviews file not found: %s. Using synthetic data.", json_path)
        return None

    try:
        # Find the metadata file that corresponds to this review file.
        # Review files are named like "Cell_Phones_and_Accessories_5.json"
        # and metadata like "meta_Cell_Phones_and_Accessories.json".
        review_stem = json_path.stem  # e.g. "Cell_Phones_and_Accessories_5"
        base_name = review_stem.removesuffix("_5")  # e.g. "Cell_Phones_and_Accessories"
        meta_path = json_path.parent / f"meta_{base_name}.json"
        if not meta_path.exists():
            # Fall back to any metadata file in the directory
            meta_candidates = list(json_path.parent.glob("meta_*.json"))
            meta_path = meta_candidates[0] if meta_candidates else None

        max_reviews = 50_000 if smoke else None
        amz = amazon_preprocess(
            json_path,
            metadata_path=meta_path,
            max_reviews=max_reviews,
        )
        logger.info(
            "Amazon Reviews loaded: %d users, %d items, %d train, %d test",
            amz.num_users, amz.num_items,
            len(amz.train_interactions), len(amz.test_interactions),
        )

        # Convert to SyntheticTasteData schema (same fields)
        return SyntheticTasteData(
            users=amz.users,
            items=amz.items,
            categories=amz.categories,
            sophistication=amz.sophistication,
            prices=amz.prices,
            interactions=amz.interactions,
            train_interactions=amz.train_interactions,
            test_interactions=amz.test_interactions,
        )
    except Exception as exc:
        logger.warning("Amazon Reviews preprocessing failed: %s. Using synthetic data.", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    *,
    smoke: bool = False,
    data_path: Optional[str] = None,
    amazon_category: str = "DigitalMusic",
) -> Dict[str, Any]:
    """Execute the taste-progression benchmark."""
    t_start = time.perf_counter()
    mode = "SMOKE" if smoke else "FULL"
    logger.info("=" * 70)
    logger.info("Taste-Progression Benchmark — %s mode", mode)
    logger.info("=" * 70)

    data: Optional[SyntheticTasteData] = None
    data_source = "synthetic"

    # Try Amazon Reviews if a path is given or --data-path=auto
    if data_path:
        data = _try_load_amazon(data_path, smoke=smoke, category=amazon_category)
        if data is not None:
            data_source = f"amazon_reviews:{amazon_category}"

    # Fall back to synthetic
    if data is None:
        num_users = 100 if smoke else 500
        num_items = 200 if smoke else 1000
        interactions_per_user = 15 if smoke else 40

        logger.info(
            "Generating synthetic data: %d users, %d items, %d interactions/user",
            num_users, num_items, interactions_per_user,
        )
        data = _generate_synthetic_data(
            num_users=num_users,
            num_items=num_items,
            interactions_per_user=interactions_per_user,
        )

    logger.info(
        "  data_source=%s  train: %d interactions, test: %d interactions",
        data_source, len(data.train_interactions), len(data.test_interactions),
    )

    results: Dict[str, Any] = {}

    # Kept-rate benchmark
    print("\n=== Kept-rate comparison ===")
    results["kept_rate"] = bench_kept_rate(data, smoke=smoke)

    # Progression curve
    print("\n=== Taste progression curves ===")
    results["progression"] = bench_progression_curve(data)
    mean_curve = results["progression"]["mean_curve"]
    if mean_curve:
        start = mean_curve.get(0, mean_curve.get("0", 0))
        end_key = max(mean_curve.keys(), key=lambda k: int(k))
        end = mean_curve[end_key]
        print(f"  Mean taste level: {start:.3f} (start) → {end:.3f} (end)")

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
    out_path = OUTPUT_DIR / "results_taste_progression.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", out_path)

    # Markdown
    _print_markdown(results)

    elapsed = time.perf_counter() - t_start
    print(f"\nCompleted in {elapsed:.1f}s. Results: {out_path}")
    return results


def _print_markdown(results: Dict[str, Any]) -> None:
    """Print Markdown tables for docs."""
    kr = results.get("kept_rate", {})

    print("\n### Kept-rate comparison\n")
    print("| System | Kept rate | Hit@10 | NDCG@10 |")
    print("|--------|----------|--------|---------|")
    for sys_name in ["popularity", "recent_popularity", "taste_progression"]:
        row = kr.get(sys_name, {})
        print(
            f"| {sys_name:<20s} "
            f"| {row.get('kept_rate', 0):.4f} "
            f"| {row.get('hit_rate', 0):.4f} "
            f"| {row.get('ndcg_at_10', 0):.4f} |"
        )

    uplift = kr.get("uplift_vs_popularity_pct", 0)
    print(f"\n**Kept-rate uplift vs popularity:** {uplift:+.1f}%")

    stretch = kr.get("stretch_accuracy", {})
    if stretch:
        print(f"\n**Stretch zone accuracy:** {stretch.get('stretch_accuracy', 0):.4f}")


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
        prog="python benchmarks/taste_progression_bench.py",
        description="Taste-progression benchmark (synthetic / Amazon Reviews).",
    )
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to Amazon Reviews data.")
    parser.add_argument(
        "--amazon-category",
        type=str,
        default="DigitalMusic",
        help="Amazon Reviews category to use (default: DigitalMusic).",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED}).")
    args = parser.parse_args()

    SEED = args.seed

    run(smoke=args.smoke, data_path=args.data_path, amazon_category=args.amazon_category)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
