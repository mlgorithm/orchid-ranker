"""Idempotent downloader for the Microsoft News Dataset (MIND).

Downloads the MIND-small dataset for curated-feed benchmarking.
MIND contains 160K+ news articles with topics, titles, and click logs
from Microsoft News.

Data source: https://msnews.github.io/

Usage::

    python benchmarks/mind/download.py

The main entry point is :func:`download_mind`.
"""
from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path(__file__).resolve().parent / "data"

# MIND-small dataset URLs (train + dev splits)
MIND_SMALL_TRAIN_URL = "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
MIND_SMALL_DEV_URL = "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest* with progress reporting."""
    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    req = urllib.request.Request(url, headers={"User-Agent": "orchid-ranker/0.1"})
    resp = urllib.request.urlopen(req)  # noqa: S310
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
        print()
    shutil.move(str(tmp), str(dest))
    logger.info("Download complete: %s (%d bytes)", dest, downloaded)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_mind(
    data_dir: Optional[Path] = None,
    *,
    split: str = "train",
) -> Path:
    """Download and extract the MIND-small dataset.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to store files.  Defaults to ``benchmarks/mind/data/``.
    split : str
        Which split to download: ``"train"`` or ``"dev"``.

    Returns
    -------
    Path
        Path to the extracted split directory.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        url = MIND_SMALL_TRAIN_URL
        zip_name = "MINDsmall_train.zip"
        extracted_name = "MINDsmall_train"
    elif split == "dev":
        url = MIND_SMALL_DEV_URL
        zip_name = "MINDsmall_dev.zip"
        extracted_name = "MINDsmall_dev"
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'train' or 'dev'.")

    zip_path = data_dir / zip_name
    extracted_path = data_dir / extracted_name

    # Download (idempotent)
    if not zip_path.exists():
        _download_with_progress(url, zip_path)
    else:
        logger.info("Zip already exists: %s", zip_path)

    # Extract (idempotent)
    if not extracted_path.is_dir():
        logger.info("Extracting %s -> %s", zip_path, data_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        if not extracted_path.is_dir():
            # Some MIND zips extract directly without a subdirectory
            # Try extracting to the expected path
            extracted_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extracted_path)
        logger.info("Extraction complete: %s", extracted_path)
    else:
        logger.info("Already extracted: %s", extracted_path)

    return extracted_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Download MIND dataset")
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    path = download_mind(args.data_dir, split=args.split)
    logger.info("MIND (%s) ready at %s", args.split, path)


if __name__ == "__main__":
    main()
