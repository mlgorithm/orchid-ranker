"""Preprocess MovieLens-1M into train/val/test splits and item features.

Follows Appendix D of the implementation plan:

* Rating >= 4 is implicit positive (label=1), else negative (label=0).
* Global leave-one-out split per user (last-in-time rating -> test).
* Validation: random 10% of remaining train, stratified by user.
* Item features: genre multi-hot (18) + year bucket one-hot (5) + avg rating (1) = 24 dims.

CLI usage::

    python benchmarks/movielens_1m/preprocess.py
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .download import DATA_DIR
except ImportError:
    # Running as a standalone script (python benchmarks/movielens_1m/preprocess.py)
    from download import DATA_DIR  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

ALL_GENRES: list[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
"""Canonical genre order for the multi-hot vector (18 genres)."""

YEAR_BUCKETS: list[str] = ["pre-1970", "1970s", "1980s", "1990s", "2000s"]
"""Decade buckets for the year one-hot vector (5 dims)."""


@dataclass
class MovieLensData:
    """Container for preprocessed MovieLens-1M data."""

    train: pd.DataFrame  # columns: user_id, item_id, rating, label, timestamp
    val: pd.DataFrame  # same columns
    test: pd.DataFrame  # same columns
    item_features: np.ndarray  # shape (num_items, 24)
    item_id_to_idx: dict  # original item_id -> 0-based index
    user_id_to_idx: dict  # original user_id -> 0-based index
    idx_to_item_id: dict  # 0-based index -> original item_id
    idx_to_user_id: dict  # 0-based index -> original user_id
    genre_names: list  # list of genre names for the multi-hot columns
    num_users: int
    num_items: int


# ---------------------------------------------------------------------------
# Raw I/O
# ---------------------------------------------------------------------------

def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``ratings.dat`` and ``movies.dat`` from *data_dir*.

    Parameters
    ----------
    data_dir:
        Path to the extracted ``ml-1m/`` directory (containing the ``.dat`` files).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(ratings, movies)`` DataFrames.
    """
    ratings_path = data_dir / "ratings.dat"
    movies_path = data_dir / "movies.dat"

    logger.info("Loading ratings from %s", ratings_path)
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    logger.info("Loading movies from %s", movies_path)
    movies = pd.read_csv(
        movies_path,
        sep="::",
        header=None,
        names=["item_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    logger.info(
        "Loaded %d ratings, %d movies, %d users",
        len(ratings),
        movies["item_id"].nunique(),
        ratings["user_id"].nunique(),
    )
    return ratings, movies


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def build_splits(
    ratings: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global leave-one-out split with 10%% stratified validation.

    For each user the last-in-time rating becomes the **test** example.
    Of the remaining ratings, 10%% (at least one per user where possible)
    are held out as **validation**; the rest form **train**.

    Parameters
    ----------
    ratings:
        DataFrame with columns ``user_id, item_id, rating, timestamp``.
    seed:
        Random seed for the validation sampling.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` DataFrames, each with an added ``label`` column.
    """
    rng = np.random.RandomState(seed)

    # Add implicit label
    df = ratings.copy()
    df["label"] = (df["rating"] >= 4).astype(int)

    # Sort by timestamp within each user (ascending)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # --- Leave-one-out: last interaction per user -> test ---
    # Use the index of the last row per user
    last_idx = df.groupby("user_id").tail(1).index
    test = df.loc[last_idx].copy()
    remaining = df.drop(last_idx).copy()

    logger.info(
        "Leave-one-out: %d test rows, %d remaining rows",
        len(test),
        len(remaining),
    )

    # --- Stratified 10% validation from remaining ---
    val_indices: list[int] = []
    for _uid, group in remaining.groupby("user_id"):
        n_val = max(1, int(round(len(group) * 0.1)))
        chosen = rng.choice(group.index, size=min(n_val, len(group)), replace=False)
        val_indices.extend(chosen.tolist())

    val_index_set = set(val_indices)
    val = remaining.loc[remaining.index.isin(val_index_set)].copy()
    train = remaining.loc[~remaining.index.isin(val_index_set)].copy()

    logger.info(
        "Splits — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Item features
# ---------------------------------------------------------------------------

def _extract_year(title: str) -> int | None:
    """Extract the 4-digit year from a MovieLens title string.

    E.g. ``"Toy Story (1995)"`` -> ``1995``.
    """
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


def _year_to_bucket(year: int | None) -> int:
    """Map a year to a bucket index (0-4)."""
    if year is None or year < 1970:
        return 0  # pre-1970
    if year < 1980:
        return 1  # 1970s
    if year < 1990:
        return 2  # 1980s
    if year < 2000:
        return 3  # 1990s
    return 4  # 2000s


def build_item_features(
    movies: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Build a 24-dim feature vector for every item.

    Columns (in order):
        0-17  : genre multi-hot (18 genres)
        18-22 : year bucket one-hot (5 dims)
        23    : average rating from training data

    Parameters
    ----------
    movies:
        DataFrame with columns ``item_id, title, genres``.
    train:
        Training split DataFrame (used for average rating — no leakage).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(feature_matrix, genre_names)`` where *feature_matrix* has shape
        ``(num_items, 24)`` and rows are ordered by the 0-based item index
        derived from sorted unique ``item_id`` values.
    """
    # Build sorted item id list
    all_item_ids = sorted(movies["item_id"].unique())
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}
    num_items = len(all_item_ids)

    genre_to_col = {g: i for i, g in enumerate(ALL_GENRES)}
    num_genres = len(ALL_GENRES)

    # Pre-compute average rating from training data
    avg_rating_map = train.groupby("item_id")["rating"].mean().to_dict()
    global_avg = train["rating"].mean()

    features = np.zeros((num_items, 24), dtype=np.float32)

    for _, row in movies.iterrows():
        idx = item_id_to_idx.get(row["item_id"])
        if idx is None:
            continue

        # Genre multi-hot (cols 0-17)
        for genre in str(row["genres"]).split("|"):
            col = genre_to_col.get(genre)
            if col is not None:
                features[idx, col] = 1.0

        # Year bucket one-hot (cols 18-22)
        year = _extract_year(str(row["title"]))
        bucket = _year_to_bucket(year)
        features[idx, 18 + bucket] = 1.0

        # Average rating (col 23) — from training data only
        features[idx, 23] = avg_rating_map.get(row["item_id"], global_avg)

    return features, list(ALL_GENRES)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def preprocess(
    data_dir: Path | None = None,
    seed: int = 42,
) -> MovieLensData:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    data_dir:
        Path to the extracted ``ml-1m/`` directory.  If *None*, uses the
        default ``DATA_DIR / "ml-1m"`` from :mod:`download`.
    seed:
        Random seed for the validation split.

    Returns
    -------
    MovieLensData
        Preprocessed data ready for training and evaluation.
    """
    if data_dir is None:
        data_dir = DATA_DIR / "ml-1m"

    ratings, movies = load_raw(data_dir)
    train, val, test = build_splits(ratings, seed=seed)
    item_features, genre_names = build_item_features(movies, train)

    # Build ID mappings
    all_user_ids = sorted(ratings["user_id"].unique())
    all_item_ids = sorted(movies["item_id"].unique())

    user_id_to_idx = {uid: idx for idx, uid in enumerate(all_user_ids)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}
    idx_to_user_id = {idx: uid for uid, idx in user_id_to_idx.items()}
    idx_to_item_id = {idx: iid for iid, idx in item_id_to_idx.items()}

    return MovieLensData(
        train=train,
        val=val,
        test=test,
        item_features=item_features,
        item_id_to_idx=item_id_to_idx,
        user_id_to_idx=user_id_to_idx,
        idx_to_item_id=idx_to_item_id,
        idx_to_user_id=idx_to_user_id,
        genre_names=genre_names,
        num_users=len(all_user_ids),
        num_items=len(all_item_ids),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    data = preprocess()

    print("\n=== MovieLens-1M Preprocessing Summary ===")
    print(f"  Users:        {data.num_users:,}")
    print(f"  Items:        {data.num_items:,}")
    print(f"  Train rows:   {len(data.train):,}")
    print(f"  Val rows:     {len(data.val):,}")
    print(f"  Test rows:    {len(data.test):,}")
    print(f"  Item features shape: {data.item_features.shape}")
    print(f"  Genres: {data.genre_names}")
    print()

    # Label distribution
    for name, split in [("train", data.train), ("val", data.val), ("test", data.test)]:
        pos = (split["label"] == 1).sum()
        neg = (split["label"] == 0).sum()
        print(f"  {name:>5s} — positive: {pos:,}  negative: {neg:,}  ratio: {pos / len(split):.3f}")

    # Feature stats
    print(f"\n  Item features — min: {data.item_features.min():.3f}, "
          f"max: {data.item_features.max():.3f}, "
          f"mean: {data.item_features.mean():.3f}")


if __name__ == "__main__":
    main()
