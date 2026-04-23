"""Idempotent downloader for the MovieLens-1M dataset.

Downloads the dataset zip, verifies its MD5 checksum, and extracts
the contents.  Designed to be both importable and runnable as a CLI::

    python benchmarks/movielens_1m/download.py

The main entry point is :func:`download_and_extract`.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_MD5 = "c4d9eecfca2ab87c1945afe126590906"

DATA_DIR: Path = Path(__file__).resolve().parent / "data"
"""Default directory where the zip is downloaded and extracted."""

_ZIP_NAME = "ml-1m.zip"
_EXTRACTED_DIR_NAME = "ml-1m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex MD5 digest of *path*."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress bar (tqdm or fallback)."""
    logger.info("Downloading %s -> %s", url, dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    req = urllib.request.Request(url, headers={"User-Agent": "orchid-ranker/0.1"})
    resp = urllib.request.urlopen(req)  # noqa: S310
    total = int(resp.headers.get("Content-Length", 0))

    try:
        from tqdm import tqdm

        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        )
        _use_tqdm = True
    except ImportError:
        _use_tqdm = False
        _downloaded = 0

    try:
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                if _use_tqdm:
                    progress.update(len(chunk))  # type: ignore[possibly-undefined]
                else:
                    _downloaded += len(chunk)  # type: ignore[possibly-undefined]
                    if total:
                        pct = _downloaded * 100 // total
                        print(
                            f"\r  {dest.name}: {_downloaded}/{total} bytes ({pct}%)",
                            end="",
                            flush=True,
                        )
    finally:
        if _use_tqdm:
            progress.close()  # type: ignore[possibly-undefined]
        elif total:
            print()  # newline after progress line

    shutil.move(str(tmp), str(dest))
    logger.info("Download complete: %s", dest)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_and_extract(data_dir: Path | None = None) -> Path:
    """Download and extract the MovieLens-1M dataset.

    Parameters
    ----------
    data_dir:
        Directory to store the zip and extracted files.
        Defaults to ``benchmarks/movielens_1m/data/``.

    Returns
    -------
    Path
        The path to the extracted ``ml-1m/`` directory.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / _ZIP_NAME
    extracted_path = data_dir / _EXTRACTED_DIR_NAME

    # ------------------------------------------------------------------
    # Step 1: download (idempotent)
    # ------------------------------------------------------------------
    if zip_path.exists():
        digest = _md5(zip_path)
        if digest == EXPECTED_MD5:
            logger.info("Zip already exists and checksum matches: %s", zip_path)
        else:
            logger.warning(
                "Zip exists but checksum mismatch (got %s, expected %s). "
                "Re-downloading.",
                digest,
                EXPECTED_MD5,
            )
            zip_path.unlink()
            _download_with_progress(URL, zip_path)
    else:
        _download_with_progress(URL, zip_path)

    # Verify checksum after download
    digest = _md5(zip_path)
    if digest != EXPECTED_MD5:
        raise RuntimeError(
            f"MD5 checksum mismatch after download: got {digest}, "
            f"expected {EXPECTED_MD5}"
        )
    logger.info("MD5 checksum verified: %s", digest)

    # ------------------------------------------------------------------
    # Step 2: extract (idempotent)
    # ------------------------------------------------------------------
    if extracted_path.is_dir():
        logger.info("Extracted directory already exists: %s", extracted_path)
    else:
        logger.info("Extracting %s -> %s", zip_path, data_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        if not extracted_path.is_dir():
            raise RuntimeError(
                f"Expected extracted directory {extracted_path} not found after unzip"
            )
        logger.info("Extraction complete: %s", extracted_path)

    return extracted_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    path = download_and_extract()
    logger.info("MovieLens-1M ready at %s", path)


if __name__ == "__main__":
    main()
