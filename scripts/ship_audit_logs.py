"""Utility to forward audit logs to an HTTP endpoint (e.g., SIEM)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
from urllib import request
from urllib.parse import urlsplit


def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _validate_destination(url: str, *, allow_insecure_http: bool = False) -> None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" and not allow_insecure_http:
        raise ValueError("audit log shipping requires an https:// destination")


def ship(log_path: Path, url: str, api_key: str | None = None, *, allow_insecure_http: bool = False) -> None:
    _validate_destination(url, allow_insecure_http=allow_insecure_http)
    for record in iter_records(log_path):
        data = json.dumps(record).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"  # adjust per SIEM
        req = request.Request(url, data=data, headers=headers, method="POST")
        with request.urlopen(req, timeout=10) as resp:  # noqa: S310
            if resp.status >= 400:
                raise RuntimeError(f"SIEM endpoint returned status {resp.status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ship audit logs to an HTTP endpoint")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("url", help="Destination webhook / SIEM endpoint")
    parser.add_argument("--api-key", help="Optional bearer token")
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="Permit plaintext HTTP destinations for local testing only.",
    )
    args = parser.parse_args(argv)
    ship(args.log_path, args.url, args.api_key, allow_insecure_http=args.allow_insecure_http)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
