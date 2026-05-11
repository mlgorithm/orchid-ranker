#!/usr/bin/env python3
"""Run logged-policy OPE benchmarks for KT-guided adaptive recommendation.

Example:
    PYTHONPATH=src python benchmarks/kt_policy_ope_benchmark.py \
        --data data/assistments_kt/interactions.csv \
        --model akt \
        --timestamp-col timestamp \
        --item-difficulty-col difficulty \
        --candidate-size 20 \
        --max-events 10000 \
        --output benchmarks/results_kt_policy_ope_assistments_akt.json
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

from orchid_ranker.policy_benchmark import run_kt_policy_ope_benchmark, run_kt_policy_ope_seed_sweep  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KT-guided policy OPE benchmark for adaptive learning CSVs.")
    parser.add_argument("--data", type=Path, required=True, help="Interaction CSV path")
    parser.add_argument("--model", choices=["sakt", "akt"], default="akt", help="KT model backing the policy")
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--correct-col", default="correct")
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--item-difficulty-col", default=None, help="Optional difficulty column for --model akt")
    parser.add_argument("--concept-col", default=None, help="Optional concept/skill column for progression policy")
    parser.add_argument(
        "--policy",
        choices=["kt_value", "progression", "delayed_gain", "support_delayed_gain"],
        default="kt_value",
    )
    parser.add_argument("--reward-mode", choices=["correctness", "progression", "delayed_gain"], default="correctness")
    parser.add_argument(
        "--delayed-gain-window",
        type=int,
        default=5,
        help="Future same-concept events used for delayed_gain reward.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap on held-out OPE events")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional input row cap for smoke tests")
    parser.add_argument("--max-weight", type=float, default=None, help="Optional importance-weight clipping")
    parser.add_argument("--logging-propensity-col", default=None, help="Use a real logged-action propensity column")
    parser.add_argument("--seed", type=int, default=42, help="Single-run random seed")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional multi-seed sweep")
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-correct", type=float, default=0.70)
    parser.add_argument(
        "--reward-model-max-examples",
        type=int,
        default=50000,
        help="Training example cap for --policy support_delayed_gain.",
    )
    parser.add_argument(
        "--reward-model-example-weighting",
        choices=["uniform", "support_inverse"],
        default="uniform",
        help="Example weighting for the support-delayed-gain direct reward model.",
    )
    parser.add_argument(
        "--reward-model-cross-fit-folds",
        type=int,
        default=1,
        help="OOF folds for reward-model diagnostics; 1 disables cross-fit diagnostics.",
    )
    parser.add_argument(
        "--reward-model-max-sample-weight",
        type=float,
        default=20.0,
        help="Clip reward-model example weights before normalization.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.data)
    if args.max_rows is not None:
        frame = frame.head(max(1, int(args.max_rows))).copy()

    kwargs = dict(
        model=args.model,
        user_col=args.user_col,
        item_col=args.item_col,
        correct_col=args.correct_col,
        timestamp_col=args.timestamp_col,
        item_difficulty_col=args.item_difficulty_col,
        concept_col=args.concept_col,
        policy=args.policy,
        reward_mode=args.reward_mode,
        delayed_gain_window=args.delayed_gain_window,
        test_fraction=args.test_fraction,
        candidate_size=args.candidate_size,
        max_events=args.max_events,
        max_weight=args.max_weight,
        logging_propensity_col=args.logging_propensity_col,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_correct=args.target_correct,
        reward_model_max_examples=args.reward_model_max_examples,
        reward_model_example_weighting=args.reward_model_example_weighting,
        reward_model_cross_fit_folds=args.reward_model_cross_fit_folds,
        reward_model_max_sample_weight=args.reward_model_max_sample_weight,
        device=args.device,
    )
    if args.seeds:
        metrics = run_kt_policy_ope_seed_sweep(frame, seeds=args.seeds, **kwargs)
    else:
        metrics = run_kt_policy_ope_benchmark(frame, random_state=args.seed, **kwargs)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
