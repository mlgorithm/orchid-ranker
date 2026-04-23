"""Preprocess Amazon Reviews data for taste-progression benchmarking.

Loads the JSON review file, extracts sophistication signals from price
and metadata, and prepares train/test splits suitable for
:class:`orchid_ranker.taste_progression.TasteProgressionRanker`.

Usage::

    from benchmarks.amazon_reviews.preprocess import preprocess
    data = preprocess(json_path)
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class AmazonReviewsData:
    """Preprocessed Amazon Reviews dataset for taste-progression eval."""

    users: List[int]
    items: List[int]
    num_users: int
    num_items: int

    # Mappings
    user_id_to_idx: Dict[str, int]  # reviewer_id -> 0-based index
    item_id_to_idx: Dict[str, int]  # asin -> 0-based index
    idx_to_user_id: Dict[int, str]
    idx_to_item_id: Dict[int, str]

    # Item metadata
    categories: Dict[int, str]        # item_idx -> category
    sophistication: Dict[int, float]  # item_idx -> sophistication [0, 1]
    prices: Dict[int, float]          # item_idx -> price

    # Interactions
    interactions: List[Dict[str, Any]]
    train_interactions: List[Dict[str, Any]]
    test_interactions: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Sophistication scoring
# ---------------------------------------------------------------------------

def _price_to_sophistication(
    prices: Dict[int, float],
    categories: Dict[int, str],
) -> Dict[int, float]:
    """Convert prices to 0-1 sophistication scores per category.

    Within each category, sophistication is the quantile rank of price.
    This captures the relative positioning: a $50 wine is sophisticated
    for table wine but entry-level for collectible spirits.
    """
    # Group items by category
    cat_items: Dict[str, List[int]] = defaultdict(list)
    for idx, cat in categories.items():
        if idx in prices:
            cat_items[cat].append(idx)

    sophistication: Dict[int, float] = {}
    for cat, idxs in cat_items.items():
        cat_prices = np.array([prices[i] for i in idxs])
        if len(cat_prices) <= 1:
            for i in idxs:
                sophistication[i] = 0.5
            continue

        # Quantile rank within category
        ranks = np.argsort(np.argsort(cat_prices)).astype(float)
        ranks /= max(len(ranks) - 1, 1)
        for i, idx in enumerate(idxs):
            sophistication[idx] = float(ranks[i])

    # Items without price get default 0.5
    for idx in categories:
        if idx not in sophistication:
            sophistication[idx] = 0.5

    return sophistication


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_reviews(path: Path, max_reviews: Optional[int] = None) -> List[Dict]:
    """Load Amazon Reviews JSONL file."""
    reviews = []
    logger.info("Loading reviews from %s", path)
    with open(path) as f:
        for line_num, line in enumerate(f):
            if max_reviews and line_num >= max_reviews:
                break
            line = line.strip()
            if not line:
                continue
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d reviews", len(reviews))
    return reviews


def _load_metadata(path: Path) -> Dict[str, Dict]:
    """Load Amazon product metadata JSONL file."""
    meta = {}
    logger.info("Loading metadata from %s", path)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                asin = item.get("asin")
                if asin:
                    meta[asin] = item
            except json.JSONDecodeError:
                continue
    logger.info("Loaded metadata for %d items", len(meta))
    return meta


def _extract_price(item_meta: Dict) -> Optional[float]:
    """Extract numeric price from metadata."""
    price = item_meta.get("price")
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        # Remove $, commas, whitespace
        cleaned = price.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_category(item_meta: Dict) -> str:
    """Extract primary category from metadata."""
    cats = item_meta.get("category", [])
    if isinstance(cats, list) and len(cats) >= 2:
        return str(cats[1])  # first sub-category
    if isinstance(cats, list) and len(cats) >= 1:
        return str(cats[0])
    main_cat = item_meta.get("main_cat", "")
    if main_cat:
        return str(main_cat)
    return "general"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    review_path: Path,
    metadata_path: Optional[Path] = None,
    *,
    min_user_interactions: int = 5,
    min_item_interactions: int = 3,
    max_reviews: Optional[int] = None,
) -> AmazonReviewsData:
    """Preprocess Amazon Reviews data for taste-progression benchmarking.

    Parameters
    ----------
    review_path : Path
        Path to the decompressed reviews JSON file.
    metadata_path : Path, optional
        Path to item metadata JSON file.  If None, prices and
        categories are inferred from review text.
    min_user_interactions : int
        Minimum reviews per user to include.
    min_item_interactions : int
        Minimum reviews per item to include.
    max_reviews : int, optional
        Cap the number of reviews loaded (for development).

    Returns
    -------
    AmazonReviewsData
    """
    reviews = _load_reviews(review_path, max_reviews=max_reviews)

    # Load metadata if available
    meta: Dict[str, Dict] = {}
    if metadata_path and metadata_path.exists():
        meta = _load_metadata(metadata_path)

    # Filter by minimum interactions
    user_counts: Dict[str, int] = defaultdict(int)
    item_counts: Dict[str, int] = defaultdict(int)
    for r in reviews:
        user_counts[r.get("reviewerID", "")] += 1
        item_counts[r.get("asin", "")] += 1

    valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    valid_items = {i for i, c in item_counts.items() if c >= min_item_interactions}

    filtered = [
        r for r in reviews
        if r.get("reviewerID") in valid_users
        and r.get("asin") in valid_items
    ]
    logger.info(
        "After filtering: %d reviews, %d users, %d items",
        len(filtered),
        len(valid_users),
        len(valid_items),
    )

    # Build ID mappings
    user_ids = sorted(set(r["reviewerID"] for r in filtered))
    item_ids = sorted(set(r["asin"] for r in filtered))
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}
    idx_to_item_id = {i: iid for iid, i in item_id_to_idx.items()}

    # Extract item metadata
    categories: Dict[int, str] = {}
    prices: Dict[int, float] = {}
    for asin, idx in item_id_to_idx.items():
        if asin in meta:
            item_meta = meta[asin]
            categories[idx] = _extract_category(item_meta)
            price = _extract_price(item_meta)
            if price is not None and price > 0:
                prices[idx] = price
        else:
            categories[idx] = "general"

    # Compute sophistication from prices
    sophistication = _price_to_sophistication(prices, categories)

    # Build interaction list
    interactions = []
    for r in filtered:
        uid = user_id_to_idx[r["reviewerID"]]
        iid = item_id_to_idx[r["asin"]]
        rating = float(r.get("overall", 3.0))
        timestamp = float(r.get("unixReviewTime", 0))

        # "kept" = rated >= 4.0 (satisfied with purchase)
        kept = rating >= 4.0

        interactions.append({
            "user_id": uid,
            "item_id": iid,
            "rating": rating,
            "kept": kept,
            "category": categories.get(iid, "general"),
            "timestamp": timestamp,
        })

    # Sort by timestamp, then split per-user: 80% train, 20% test
    interactions.sort(key=lambda x: x["timestamp"])

    by_user: Dict[int, List[Dict]] = defaultdict(list)
    for ix in interactions:
        by_user[ix["user_id"]].append(ix)

    train, test = [], []
    for uid, ixs in by_user.items():
        ixs.sort(key=lambda x: x["timestamp"])
        split_idx = max(1, int(len(ixs) * 0.8))
        train.extend(ixs[:split_idx])
        test.extend(ixs[split_idx:])

    logger.info("Train: %d interactions, Test: %d interactions", len(train), len(test))

    return AmazonReviewsData(
        users=list(range(len(user_ids))),
        items=list(range(len(item_ids))),
        num_users=len(user_ids),
        num_items=len(item_ids),
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        idx_to_user_id=idx_to_user_id,
        idx_to_item_id=idx_to_item_id,
        categories=categories,
        sophistication=sophistication,
        prices=prices,
        interactions=interactions,
        train_interactions=train,
        test_interactions=test,
    )
