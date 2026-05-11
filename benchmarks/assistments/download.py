#!/usr/bin/env python3
"""Download helper for ASSISTments-style KT benchmark data.

Most ASSISTments datasets require accepting terms, filling an access form, or
using a mirror. This helper therefore supports two modes:

1. `--url`: download a user-provided CSV/ZIP URL.
2. `--source assistments-2009`: convenience mirror URL for the classic 2009
   skill-builder data, useful when the mirror is reachable from your network.

For gated sources such as ASSISTments 2017 or FoundationalASSIST, download the
files manually after accepting the terms, then run `preprocess.py`.
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

SOURCE_URLS = {
    "assistments-2009": "http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ASSISTments-style benchmark data.")
    parser.add_argument("--source", choices=sorted(SOURCE_URLS), help="Known public mirror to download")
    parser.add_argument("--url", help="Direct CSV or ZIP URL. Overrides --source.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/assistments_raw"))
    parser.add_argument("--filename", help="Optional output filename")
    parser.add_argument("--extract", action="store_true", help="Extract ZIP files after download")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    url = args.url or (SOURCE_URLS.get(args.source) if args.source else None)
    if not url:
        raise SystemExit("Provide --url or --source. Gated datasets must be downloaded manually.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = args.filename or Path(url.split("?")[0]).name or "assistments_download"
    target = args.output_dir / filename

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    print(f"Wrote {target}")

    if args.extract and zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as zf:
            zf.extractall(args.output_dir)
        print(f"Extracted ZIP into {args.output_dir}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
