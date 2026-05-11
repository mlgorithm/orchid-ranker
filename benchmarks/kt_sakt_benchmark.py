#!/usr/bin/env python3
"""Run leakage-safe KT benchmarks on EdNet/ASSISTments-style CSV data.

Example:
    PYTHONPATH=src python benchmarks/kt_sakt_benchmark.py \
        --data data/ednet_interactions.csv \
        --user-col user_id \
        --item-col question_id \
        --correct-col correct \
        --timestamp-col timestamp \
        --model akt \
        --item-difficulty-col difficulty \
        --output benchmarks/results_kt_sakt.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orchid_ranker.kt_benchmark import run_kt_benchmark  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe KT benchmark for adaptive learning CSVs.")
    parser.add_argument("--data", type=Path, required=True, help="Interaction CSV path")
    parser.add_argument("--model", choices=["sakt", "akt"], default="sakt", help="KT model to train")
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--correct-col", default="correct")
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--item-difficulty-col", default=None, help="Optional difficulty column for --model akt")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row cap")
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.data)
    if args.max_rows is not None:
        frame = frame.head(max(1, int(args.max_rows))).copy()

    metrics = run_kt_benchmark(
        frame,
        model=args.model,
        user_col=args.user_col,
        item_col=args.item_col,
        correct_col=args.correct_col,
        timestamp_col=args.timestamp_col,
        item_difficulty_col=args.item_difficulty_col,
        test_fraction=args.test_fraction,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
