"""Preprocess Last.fm 1K into train/val/test splits and item features.

Follows the same Appendix D protocol as the MovieLens-1M benchmark:

* Each unique (user, track) listen event is an implicit positive interaction.
* Global leave-one-out split per user (last-in-time interaction -> test).
* Validation: random 10% of remaining train, stratified by user.
* Item features: top-N artist one-hot + log-scaled play-count = variable dims.

The dataset TSV has columns::

    userid \\t timestamp \\t musicbrainz-artist-id \\t artist-name \\t
    musicbrainz-track-id \\t track-name

Some MusicBrainz IDs may be empty; we use artist-name + track-name as a
fallback composite key.

CLI usage::

    python benchmarks/music/preprocess.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .download import DATA_DIR
except ImportError:
    # Running as a standalone script
    from download import DATA_DIR  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_N_ARTISTS = 200
"""Number of top artists to keep as individual one-hot features.
Artists outside the top-N are grouped into an 'other' bucket."""

MIN_USER_INTERACTIONS = 20
"""Minimum number of unique track interactions to keep a user."""

MIN_ITEM_INTERACTIONS = 10
"""Minimum number of unique user interactions to keep a track."""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MusicData:
    """Container for preprocessed Last.fm 1K data."""

    train: pd.DataFrame  # columns: user_id, item_id, label, timestamp
    val: pd.DataFrame  # same columns
    test: pd.DataFrame  # same columns
    item_features: np.ndarray  # shape (num_items, feature_dim)
    item_id_to_idx: dict  # composite track key -> 0-based index
    user_id_to_idx: dict  # original userid -> 0-based index
    idx_to_item_id: dict  # 0-based index -> composite track key
    idx_to_user_id: dict  # 0-based index -> original userid
    artist_names: list  # list of artist names for one-hot columns
    num_users: int
    num_items: int
    feature_dim: int


# ---------------------------------------------------------------------------
# Raw I/O
# ---------------------------------------------------------------------------

def _build_track_key(row: pd.Series) -> str:
    """Build a stable composite key for a track.

    Prefers MusicBrainz track ID when available, otherwise falls back to
    ``artist_name::track_name``.
    """
    mb_track = row.get("mb_track_id", "")
    if pd.notna(mb_track) and str(mb_track).strip():
        return str(mb_track).strip()
    artist = str(row.get("artist_name", "")).strip()
    track = str(row.get("track_name", "")).strip()
    return f"{artist}::{track}"


def load_raw(data_dir: Path) -> pd.DataFrame:
    """Read ``userid-timestamp-artid-artname-traid-traname.tsv`` from *data_dir*.

    Parameters
    ----------
    data_dir:
        Path to the extracted ``lastfm-dataset-1K/`` directory.

    Returns
    -------
    pd.DataFrame
        Raw listening events with columns:
        ``userid, timestamp, mb_artist_id, artist_name, mb_track_id, track_name``.
    """
    tsv_path = data_dir / "userid-timestamp-artid-artname-traid-traname.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Expected TSV file not found: {tsv_path}. "
            f"Make sure the Last.fm 1K dataset is properly extracted."
        )

    logger.info("Loading listening events from %s", tsv_path)
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=[
            "userid",
            "timestamp",
            "mb_artist_id",
            "artist_name",
            "mb_track_id",
            "track_name",
        ],
        on_bad_lines="skip",
        quoting=3,  # QUOTE_NONE — prevents issues with unescaped quotes
        encoding="utf-8",
        dtype=str,
    )

    logger.info(
        "Loaded %d listening events from %d users",
        len(df),
        df["userid"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Build implicit interactions
# ---------------------------------------------------------------------------

def _build_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw listening events into unique (user, track) interactions.

    Each unique (user, track) pair becomes one positive interaction.  The
    timestamp is the *last* listen time for that pair (used for the temporal
    leave-one-out split).  Play count (number of listens) is preserved as
    an auxiliary feature.

    Returns
    -------
    pd.DataFrame
        Columns: ``user_id, item_id, play_count, timestamp, artist_name, label``.
    """
    raw = raw.copy()

    # Build composite track keys
    raw["item_id"] = raw.apply(_build_track_key, axis=1)

    # Parse timestamps (coerce errors to NaT, then drop)
    raw["ts_parsed"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["ts_parsed"])

    # Aggregate: one row per (user, track) pair
    grouped = raw.groupby(["userid", "item_id"]).agg(
        play_count=("ts_parsed", "size"),
        timestamp=("ts_parsed", "max"),
        artist_name=("artist_name", "first"),
    ).reset_index()

    grouped = grouped.rename(columns={"userid": "user_id"})

    # Convert timestamp to Unix epoch (float) for consistent sorting
    grouped["timestamp"] = grouped["timestamp"].astype(np.int64) // 10**9

    # All interactions are positive (implicit feedback)
    grouped["label"] = 1

    logger.info(
        "Built %d unique interactions from %d users and %d tracks",
        len(grouped),
        grouped["user_id"].nunique(),
        grouped["item_id"].nunique(),
    )
    return grouped


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_interactions(
    df: pd.DataFrame,
    min_user: int = MIN_USER_INTERACTIONS,
    min_item: int = MIN_ITEM_INTERACTIONS,
) -> pd.DataFrame:
    """Iteratively filter users and items below minimum interaction thresholds.

    Applies alternating user and item filters until convergence (since
    removing items can drop users below threshold and vice versa).

    Parameters
    ----------
    df:
        Interaction DataFrame with ``user_id`` and ``item_id`` columns.
    min_user:
        Minimum number of unique track interactions per user.
    min_item:
        Minimum number of unique user interactions per track.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    prev_len = -1
    iteration = 0

    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1

        # Filter items
        item_counts = df.groupby("item_id").size()
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df["item_id"].isin(valid_items)]

        # Filter users
        user_counts = df.groupby("user_id").size()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["user_id"].isin(valid_users)]

        logger.info(
            "  Filter iteration %d: %d interactions, %d users, %d items",
            iteration,
            len(df),
            df["user_id"].nunique(),
            df["item_id"].nunique(),
        )

    logger.info(
        "Filtering complete: %d interactions, %d users, %d items",
        len(df),
        df["user_id"].nunique(),
        df["item_id"].nunique(),
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def build_splits(
    interactions: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global leave-one-out split with 10% stratified validation.

    For each user the last-in-time interaction becomes the **test** example.
    Of the remaining interactions, 10% (at least one per user where possible)
    are held out as **validation**; the rest form **train**.

    Parameters
    ----------
    interactions:
        DataFrame with columns ``user_id, item_id, play_count, timestamp, label``.
    seed:
        Random seed for the validation sampling.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` DataFrames.
    """
    rng = np.random.RandomState(seed)

    # Sort by timestamp within each user (ascending)
    df = interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # --- Leave-one-out: last interaction per user -> test ---
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
        "Splits -- train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Item features
# ---------------------------------------------------------------------------

def build_item_features(
    interactions: pd.DataFrame,
    train: pd.DataFrame,
    top_n_artists: int = TOP_N_ARTISTS,
) -> tuple[np.ndarray, list[str]]:
    """Build item feature vectors for every track.

    Feature layout (in order):
        0 .. top_n_artists  : artist one-hot (top_n_artists + 1 dims, last = "other")
        top_n_artists + 1   : log-scaled play count from training data

    Total dimension = top_n_artists + 2.

    Parameters
    ----------
    interactions:
        Full interaction DataFrame (for artist metadata).
    train:
        Training split DataFrame (for play-count features -- no leakage).
    top_n_artists:
        Number of top artists to encode individually.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(feature_matrix, artist_names)`` where *feature_matrix* has shape
        ``(num_items, feature_dim)`` and rows are ordered by the 0-based item
        index derived from sorted unique ``item_id`` values.
    """
    # Build sorted item id list
    all_item_ids = sorted(interactions["item_id"].unique())
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}
    num_items = len(all_item_ids)

    # Identify top-N artists by frequency in the full dataset
    artist_counts = interactions.groupby("artist_name").size().sort_values(ascending=False)
    top_artists = list(artist_counts.head(top_n_artists).index)
    artist_to_col = {a: i for i, a in enumerate(top_artists)}
    n_artist_cols = top_n_artists + 1  # +1 for "other" bucket

    feature_dim = n_artist_cols + 1  # artist one-hot + log play count
    features = np.zeros((num_items, feature_dim), dtype=np.float32)

    # Build item -> artist mapping (take the most common artist per track)
    item_artist = (
        interactions.groupby("item_id")["artist_name"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "")
        .to_dict()
    )

    # Pre-compute play count from training data (sum across users)
    train_play_counts = train.groupby("item_id")["play_count"].sum().to_dict()
    global_mean_play = train["play_count"].mean() if len(train) > 0 else 1.0

    for item_id, idx in item_id_to_idx.items():
        # Artist one-hot (cols 0 .. n_artist_cols-1)
        artist = item_artist.get(item_id, "")
        col = artist_to_col.get(artist)
        if col is not None:
            features[idx, col] = 1.0
        else:
            features[idx, top_n_artists] = 1.0  # "other" bucket

        # Log-scaled play count (last column)
        pc = train_play_counts.get(item_id, global_mean_play)
        features[idx, n_artist_cols] = np.log1p(float(pc))

    return features, top_artists


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def preprocess(
    data_dir: Path | None = None,
    seed: int = 42,
) -> MusicData:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    data_dir:
        Path to the extracted ``lastfm-dataset-1K/`` directory.  If *None*,
        uses the default ``DATA_DIR / "lastfm-dataset-1K"`` from :mod:`download`.
    seed:
        Random seed for the validation split.

    Returns
    -------
    MusicData
        Preprocessed data ready for training and evaluation.
    """
    if data_dir is None:
        data_dir = DATA_DIR / "lastfm-dataset-1K"

    raw = load_raw(data_dir)
    interactions = _build_interactions(raw)
    interactions = _filter_interactions(interactions)

    train, val, test = build_splits(interactions, seed=seed)
    item_features, artist_names = build_item_features(interactions, train)

    # Build ID mappings
    all_user_ids = sorted(interactions["user_id"].unique())
    all_item_ids = sorted(interactions["item_id"].unique())

    user_id_to_idx = {uid: idx for idx, uid in enumerate(all_user_ids)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}
    idx_to_user_id = {idx: uid for uid, idx in user_id_to_idx.items()}
    idx_to_item_id = {idx: iid for iid, idx in item_id_to_idx.items()}

    feature_dim = item_features.shape[1]

    return MusicData(
        train=train,
        val=val,
        test=test,
        item_features=item_features,
        item_id_to_idx=item_id_to_idx,
        user_id_to_idx=user_id_to_idx,
        idx_to_item_id=idx_to_item_id,
        idx_to_user_id=idx_to_user_id,
        artist_names=artist_names,
        num_users=len(all_user_ids),
        num_items=len(all_item_ids),
        feature_dim=feature_dim,
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

    print("\n=== Last.fm 1K Preprocessing Summary ===")
    print(f"  Users:        {data.num_users:,}")
    print(f"  Items:        {data.num_items:,}")
    print(f"  Train rows:   {len(data.train):,}")
    print(f"  Val rows:     {len(data.val):,}")
    print(f"  Test rows:    {len(data.test):,}")
    print(f"  Item features shape: {data.item_features.shape}")
    print(f"  Feature dim:  {data.feature_dim}")
    print(f"  Top artists:  {data.artist_names[:10]} ...")
    print()

    # Label distribution (all labels are 1 for implicit data)
    for name, split in [("train", data.train), ("val", data.val), ("test", data.test)]:
        pos = (split["label"] == 1).sum()
        print(f"  {name:>5s} -- positive: {pos:,}  (implicit, all positive)")

    # Feature stats
    print(
        f"\n  Item features -- min: {data.item_features.min():.3f}, "
        f"max: {data.item_features.max():.3f}, "
        f"mean: {data.item_features.mean():.3f}"
    )


if __name__ == "__main__":
    main()
