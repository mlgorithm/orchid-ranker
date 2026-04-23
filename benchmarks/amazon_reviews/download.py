"""Idempotent downloader for the Amazon Reviews 2018 dataset.

Downloads review data for taste-progression-friendly categories
(Wines, Cameras, Fragrances).  Uses the 5-core subset (users and items
with at least 5 reviews each).

The data source is the McAuley Lab Amazon Reviews '18 dataset:
https://nijianmo.github.io/amazon/index.html

Usage::

    python benchmarks/amazon_reviews/download.py
    python benchmarks/amazon_reviews/download.py --category CellPhones

The main entry point is :func:`download_category`.
"""
from __future__ import annotations

import gzip
import logging
import shutil
import ssl
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path(__file__).resolve().parent / "data"

# Amazon Reviews 2018 (5-core) download URLs.
# These are the compressed JSONL files from McAuley Lab.
CATEGORY_URLS = {
    "CellPhones": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Cell_Phones_and_Accessories_5.json.gz",
    "Automotive": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Automotive_5.json.gz",
    "MusicalInstruments": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz",
    "DigitalMusic": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz",
    "VideoGames": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz",
}

# Metadata URLs (item descriptions, categories, price)
METADATA_URLS = {
    "CellPhones": "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Cell_Phones_and_Accessories.json.gz",
    "Automotive": "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Automotive.json.gz",
    "MusicalInstruments": "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz",
    "DigitalMusic": "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Digital_Music.json.gz",
    "VideoGames": "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz",
}

DEFAULT_CATEGORY = "DigitalMusic"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ssl_context() -> ssl.SSLContext:
    """Permissive SSL context for academic dataset servers."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest* with progress reporting."""
    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    req = urllib.request.Request(url, headers={"User-Agent": "orchid-ranker/0.1"})
    resp = urllib.request.urlopen(req, context=_ssl_context())  # noqa: S310
    total = int(resp.headers.get("Content-Length", 0))

    downloaded = 0
    with open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(
                    f"\r  {dest.name}: {downloaded}/{total} bytes ({pct}%)",
                    end="",
                    flush=True,
                )
    if total:
        print()  # newline
    shutil.move(str(tmp), str(dest))
    logger.info("Download complete: %s (%d bytes)", dest, downloaded)


def _decompress_gz(gz_path: Path, out_path: Path) -> None:
    """Decompress a .gz file to *out_path*."""
    logger.info("Decompressing %s -> %s", gz_path, out_path)
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_category(
    category: str = DEFAULT_CATEGORY,
    data_dir: Optional[Path] = None,
    *,
    include_metadata: bool = True,
) -> Path:
    """Download Amazon Reviews data for a category.

    Parameters
    ----------
    category : str
        Category name (key in ``CATEGORY_URLS``).
    data_dir : Path, optional
        Where to store files.  Defaults to ``benchmarks/amazon_reviews/data/``.
    include_metadata : bool
        Whether to also download item metadata (descriptions, prices).

    Returns
    -------
    Path
        Path to the decompressed reviews JSON file.
    """
    if category not in CATEGORY_URLS:
        raise ValueError(
            f"Unknown category '{category}'. Available: {list(CATEGORY_URLS.keys())}"
        )

    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download reviews
    review_url = CATEGORY_URLS[category]
    gz_name = review_url.rsplit("/", 1)[-1]
    gz_path = data_dir / gz_name
    json_path = data_dir / gz_name.replace(".json.gz", ".json")

    if json_path.exists():
        logger.info("Reviews already exist: %s", json_path)
    elif gz_path.exists():
        logger.info("Compressed reviews exist, decompressing: %s", gz_path)
        _decompress_gz(gz_path, json_path)
    else:
        _download_with_progress(review_url, gz_path)
        _decompress_gz(gz_path, json_path)

    # Download metadata
    if include_metadata and category in METADATA_URLS:
        meta_url = METADATA_URLS[category]
        meta_gz_name = meta_url.rsplit("/", 1)[-1]
        meta_gz_path = data_dir / meta_gz_name
        meta_json_path = data_dir / meta_gz_name.replace(".json.gz", ".json")

        if meta_json_path.exists():
            logger.info("Metadata already exists: %s", meta_json_path)
        elif meta_gz_path.exists():
            logger.info("Compressed metadata exists, decompressing: %s", meta_gz_path)
            _decompress_gz(meta_gz_path, meta_json_path)
        else:
            _download_with_progress(meta_url, meta_gz_path)
            _decompress_gz(meta_gz_path, meta_json_path)

    return json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Download Amazon Reviews data")
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        choices=list(CATEGORY_URLS.keys()),
        help="Product category to download",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    path = download_category(args.category, args.data_dir)
    logger.info("Amazon Reviews (%s) ready at %s", args.category, path)


if __name__ == "__main__":
    main()
