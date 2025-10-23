"""General entry point for running registered preprocessors."""
from __future__ import annotations

import argparse
from pathlib import Path

from .base import PreprocessorConfig, get_preprocessor, list_preprocessors

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Orchid Ranker dataset preprocessors")
    parser.add_argument("dataset", help="Registered dataset name")
    parser.add_argument("base_path", help="Path to raw data")
    parser.add_argument("output_path", help="Where to write processed CSVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--extra", type=str, default=None, help="Optional key=value pairs, comma separated")
    args = parser.parse_args()

    extra = {}
    if args.extra:
        for pair in args.extra.split(","):
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Invalid extra parameter '{pair}' (expected key=value)")
            k, v = pair.split("=", 1)
            extra[k.strip()] = v.strip()

    try:
        preprocessor = get_preprocessor(args.dataset)
    except KeyError as exc:
        available = ", ".join(list_preprocessors().keys())
        raise SystemExit(f"Unknown dataset {args.dataset!r}. Available: {available}") from exc

    cfg = PreprocessorConfig(
        base_path=args.base_path,
        output_path=args.output_path,
        seed=args.seed,
        n_users=args.n_users,
        extra=extra or None,
    )
    preprocessor.run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
