"""Preprocess MIND dataset for curated-feed benchmarking.

Loads the MIND news dataset and converts it to the format expected by
the curated-feed benchmark: :class:`FeedItem` objects with topics,
difficulty levels, and timestamps.

Usage::

    from benchmarks.mind.preprocess import preprocess
    data = preprocess(mind_path)
"""
from __future__ import annotations

import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# We import FeedItem lazily to avoid circular imports when the module
# is imported from non-benchmark code.


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MINDData:
    """Preprocessed MIND dataset for curated-feed evaluation."""

    topics: List[str]
    items: List[Any]  # List of FeedItem
    users: List[int]
    num_users: int
    num_items: int

    # Per-user click history: user_id -> [(item, engaged, timestamp)]
    interactions: Dict[int, List[Tuple[Any, bool, float]]]

    # Train/test split items
    train_items: List[Any]
    test_items: List[Any]

    # Metadata
    news_id_to_idx: Dict[str, int]
    user_id_to_idx: Dict[str, int]


# ---------------------------------------------------------------------------
# Loading MIND files
# ---------------------------------------------------------------------------

def _load_news(news_tsv: Path) -> Dict[str, Dict]:
    """Load news.tsv and return {news_id: metadata}."""
    news = {}
    logger.info("Loading news from %s", news_tsv)
    with open(news_tsv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            news_id = row[0]
            category = row[1] if len(row) > 1 else "general"
            subcategory = row[2] if len(row) > 2 else ""
            title = row[3] if len(row) > 3 else ""
            abstract = row[4] if len(row) > 4 else ""
            news[news_id] = {
                "news_id": news_id,
                "category": category,
                "subcategory": subcategory,
                "title": title,
                "abstract": abstract,
            }
    logger.info("Loaded %d news articles", len(news))
    return news


def _load_behaviors(behaviors_tsv: Path) -> List[Dict]:
    """Load behaviors.tsv and return click logs."""
    behaviors = []
    logger.info("Loading behaviors from %s", behaviors_tsv)
    with open(behaviors_tsv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            impression_id = row[0]
            user_id = row[1]
            timestamp_str = row[2]  # e.g. "11/15/2019 7:28:42 AM"
            history_str = row[3]    # space-separated news IDs
            impressions_str = row[4]  # space-separated "newsID-{0,1}"

            # Parse timestamp
            try:
                import datetime
                ts = datetime.datetime.strptime(
                    timestamp_str, "%m/%d/%Y %I:%M:%S %p"
                ).timestamp()
            except (ValueError, TypeError):
                ts = time.time()

            # Parse history
            history = history_str.split() if history_str else []

            # Parse impressions: "N12345-1" means clicked, "N12345-0" means not
            impressions = []
            for imp in impressions_str.split():
                parts = imp.rsplit("-", 1)
                if len(parts) == 2:
                    nid, label = parts[0], int(parts[1])
                    impressions.append((nid, label == 1))

            behaviors.append({
                "impression_id": impression_id,
                "user_id": user_id,
                "timestamp": ts,
                "history": history,
                "impressions": impressions,
            })
    logger.info("Loaded %d behavior records", len(behaviors))
    return behaviors


# ---------------------------------------------------------------------------
# Difficulty estimation
# ---------------------------------------------------------------------------

def _estimate_difficulty(
    news: Dict[str, Dict],
) -> Dict[str, float]:
    """Estimate reading difficulty from title + abstract length.

    Longer articles with more complex titles are harder.  This is a
    rough proxy; a production system would use readability metrics.
    """
    difficulties: Dict[str, float] = {}
    lengths = []
    for nid, meta in news.items():
        title_len = len(meta.get("title", "").split())
        abstract_len = len(meta.get("abstract", "").split())
        total = title_len + abstract_len
        lengths.append(total)

    if not lengths:
        return difficulties

    lengths_arr = np.array(lengths, dtype=float)
    pmin, pmax = lengths_arr.min(), lengths_arr.max()
    if pmax > pmin:
        normed = (lengths_arr - pmin) / (pmax - pmin)
    else:
        normed = np.full_like(lengths_arr, 0.5)

    for i, nid in enumerate(news):
        difficulties[nid] = float(normed[i])

    return difficulties


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    mind_dir: Path,
    *,
    max_users: Optional[int] = None,
    max_items: Optional[int] = None,
) -> MINDData:
    """Preprocess MIND dataset for curated-feed benchmarking.

    Parameters
    ----------
    mind_dir : Path
        Path to the extracted MIND split directory (containing
        ``news.tsv`` and ``behaviors.tsv``).
    max_users : int, optional
        Limit number of users for faster testing.
    max_items : int, optional
        Limit number of news items.

    Returns
    -------
    MINDData
    """
    from orchid_ranker.curated_feed import FeedItem

    news_tsv = mind_dir / "news.tsv"
    behaviors_tsv = mind_dir / "behaviors.tsv"

    if not news_tsv.exists():
        raise FileNotFoundError(f"news.tsv not found in {mind_dir}")
    if not behaviors_tsv.exists():
        raise FileNotFoundError(f"behaviors.tsv not found in {mind_dir}")

    # Load data
    news = _load_news(news_tsv)
    behaviors = _load_behaviors(behaviors_tsv)

    # Estimate difficulty
    difficulties = _estimate_difficulty(news)

    # Build news-to-index mapping
    news_ids = sorted(news.keys())
    if max_items:
        news_ids = news_ids[:max_items]
    news_id_to_idx = {nid: i for i, nid in enumerate(news_ids)}
    valid_news = set(news_ids)

    # Build user-to-index mapping
    user_ids_raw = sorted(set(b["user_id"] for b in behaviors))
    if max_users:
        user_ids_raw = user_ids_raw[:max_users]
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids_raw)}
    valid_users = set(user_ids_raw)

    # Create FeedItem objects
    items: List[FeedItem] = []
    for nid in news_ids:
        meta = news[nid]
        item = FeedItem(
            item_id=news_id_to_idx[nid],
            topic=meta.get("category", "general"),
            difficulty=difficulties.get(nid, 0.5),
            timestamp=time.time() - np.random.uniform(0, 720 * 3600),  # spread over 30 days
        )
        items.append(item)

    topics = sorted(set(item.topic for item in items))

    # Build user interaction histories from behaviors
    interactions: Dict[int, List[Tuple[FeedItem, bool, float]]] = defaultdict(list)
    item_by_idx = {item.item_id: item for item in items}

    for beh in behaviors:
        uid_str = beh["user_id"]
        if uid_str not in valid_users:
            continue
        uid = user_id_to_idx[uid_str]
        ts = beh["timestamp"]

        for nid, clicked in beh["impressions"]:
            if nid not in valid_news:
                continue
            idx = news_id_to_idx[nid]
            if idx in item_by_idx:
                interactions[uid].append((item_by_idx[idx], clicked, ts))

    logger.info(
        "Preprocessed: %d users, %d items, %d topics, %d interaction records",
        len(user_id_to_idx),
        len(items),
        len(topics),
        sum(len(v) for v in interactions.values()),
    )

    # Split items: 80% train, 20% test by timestamp
    sorted_items = sorted(items, key=lambda x: x.timestamp)
    split_idx = int(len(sorted_items) * 0.8)
    train_items = sorted_items[:split_idx]
    test_items = sorted_items[split_idx:]

    return MINDData(
        topics=topics,
        items=items,
        users=list(range(len(user_id_to_idx))),
        num_users=len(user_id_to_idx),
        num_items=len(items),
        interactions=interactions,
        train_items=train_items,
        test_items=test_items,
        news_id_to_idx=news_id_to_idx,
        user_id_to_idx=user_id_to_idx,
    )
