"""Idempotent downloader for the Last.fm 1K dataset.

Downloads the dataset tar.gz, verifies its SHA-256 checksum, and extracts
the contents.  Designed to be both importable and runnable as a CLI::

    python benchmarks/music/download.py

The main entry point is :func:`download_and_extract`.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
EXPECTED_SHA256 = ""  # populated on first verified download; empty = skip check

DATA_DIR: Path = Path(__file__).resolve().parent / "data"
"""Default directory where the archive is downloaded and extracted."""

_ARCHIVE_NAME = "lastfm-dataset-1K.tar.gz"
_EXTRACTED_DIR_NAME = "lastfm-dataset-1K"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
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
    """Download and extract the Last.fm 1K dataset.

    Parameters
    ----------
    data_dir:
        Directory to store the archive and extracted files.
        Defaults to ``benchmarks/music/data/``.

    Returns
    -------
    Path
        The path to the extracted ``lastfm-dataset-1K/`` directory.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / _ARCHIVE_NAME
    extracted_path = data_dir / _EXTRACTED_DIR_NAME

    # ------------------------------------------------------------------
    # Step 1: download (idempotent)
    # ------------------------------------------------------------------
    if archive_path.exists():
        if EXPECTED_SHA256:
            digest = _sha256(archive_path)
            if digest == EXPECTED_SHA256:
                logger.info(
                    "Archive already exists and checksum matches: %s",
                    archive_path,
                )
            else:
                logger.warning(
                    "Archive exists but checksum mismatch (got %s, expected %s). "
                    "Re-downloading.",
                    digest,
                    EXPECTED_SHA256,
                )
                archive_path.unlink()
                _download_with_progress(URL, archive_path)
        else:
            logger.info(
                "Archive already exists (checksum verification skipped): %s",
                archive_path,
            )
    else:
        _download_with_progress(URL, archive_path)

    # Verify checksum after download (if we have an expected value)
    if EXPECTED_SHA256:
        digest = _sha256(archive_path)
        if digest != EXPECTED_SHA256:
            raise RuntimeError(
                f"SHA-256 checksum mismatch after download: got {digest}, "
                f"expected {EXPECTED_SHA256}"
            )
        logger.info("SHA-256 checksum verified: %s", digest)

    # ------------------------------------------------------------------
    # Step 2: extract (idempotent)
    # ------------------------------------------------------------------
    if extracted_path.is_dir():
        logger.info("Extracted directory already exists: %s", extracted_path)
    else:
        logger.info("Extracting %s -> %s", archive_path, data_dir)
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(data_dir)
        if not extracted_path.is_dir():
            raise RuntimeError(
                f"Expected extracted directory {extracted_path} not found "
                f"after extraction"
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
    logger.info("Last.fm 1K ready at %s", path)


if __name__ == "__main__":
    main()
