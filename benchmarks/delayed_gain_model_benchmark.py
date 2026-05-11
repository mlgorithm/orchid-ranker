#!/usr/bin/env python3
"""Diagnose delayed-gain reward models under target-policy replay.

The policy OPE benchmark says whether a candidate policy wins. This benchmark
explains whether the direct delayed-gain reward model is trustworthy enough for
doubly robust OPE and value-driven ranking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orchid_ranker.delayed_gain import (  # noqa: E402
    FEATURE_NAMES,
    build_delayed_gain_training_frame,
    diagnose_delayed_gain_predictions,
    fit_delayed_gain_reward_model,
)
from orchid_ranker.kt_benchmark import time_ordered_user_split  # noqa: E402
from orchid_ranker.policy_benchmark import (  # noqa: E402
    _fit_tracer,
    _mode_or_first,
    _support_tables,
    attach_delayed_gain_rewards,
    build_kt_policy_ope_events,
    estimate_delayed_gain_priors,
)
from orchid_ranker.progression_reward import ProgressionRewardConfig  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delayed-gain reward-model diagnostic benchmark.")
    parser.add_argument("--data", type=Path, required=True, help="Interaction CSV path")
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--correct-col", default="correct")
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--item-difficulty-col", default=None)
    parser.add_argument("--concept-col", required=True)
    parser.add_argument("--model", choices=["sakt", "akt"], default="akt")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional input row cap for smoke tests")
    parser.add_argument("--delayed-gain-window", type=int, default=5)
    parser.add_argument("--target-correct", type=float, default=0.95)
    parser.add_argument("--reward-model-max-examples", type=int, default=50000)
    parser.add_argument(
        "--reward-model-weightings",
        nargs="+",
        choices=["uniform", "support_inverse"],
        default=["uniform", "support_inverse"],
    )
    parser.add_argument("--reward-model-cross-fit-folds", type=int, default=3)
    parser.add_argument("--reward-model-max-sample-weight", type=float, default=20.0)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.data)
    if args.max_rows is not None:
        frame = frame.head(max(1, int(args.max_rows))).copy()

    split = time_ordered_user_split(
        frame,
        user_col=args.user_col,
        item_col=args.item_col,
        correct_col=args.correct_col,
        timestamp_col=args.timestamp_col,
        test_fraction=args.test_fraction,
    )
    priors = estimate_delayed_gain_priors(
        split,
        concept_col=args.concept_col,
        future_window=args.delayed_gain_window,
    )
    progression_config = ProgressionRewardConfig(target_correct=args.target_correct)
    training_frame = build_delayed_gain_training_frame(
        split,
        concept_col=args.concept_col,
        item_difficulty_col=args.item_difficulty_col,
        item_gain_prior=priors["item_gain_prior"],
        concept_gain_prior=priors["concept_gain_prior"],
        global_gain_prior=priors["global_gain_prior"],
        future_window=args.delayed_gain_window,
        config=progression_config,
    )

    runs = []
    for weighting in args.reward_model_weightings:
        model = fit_delayed_gain_reward_model(
            split,
            concept_col=args.concept_col,
            item_difficulty_col=args.item_difficulty_col,
            item_gain_prior=priors["item_gain_prior"],
            concept_gain_prior=priors["concept_gain_prior"],
            global_gain_prior=priors["global_gain_prior"],
            future_window=args.delayed_gain_window,
            max_examples=args.reward_model_max_examples,
            example_weighting=weighting,
            max_sample_weight=args.reward_model_max_sample_weight,
            cross_fit_folds=args.reward_model_cross_fit_folds,
            random_state=args.seed,
            config=progression_config,
        )
        events = _target_policy_events(args, split, priors, model, progression_config)
        runs.append(
            {
                "reward_model_example_weighting": weighting,
                "model_report": model.to_dict(),
                "training_frame_in_sample": _training_frame_diagnostics(training_frame, model),
                "target_policy": _event_diagnostics(events),
            }
        )

    payload = {
        "config": _config(args, frame, training_frame),
        "delayed_gain_policy": {
            "global_gain_prior": priors["global_gain_prior"],
            "item_priors": float(len(priors["item_gain_prior"])),
            "concept_priors": float(len(priors["concept_gain_prior"])),
            "shrinkage": priors["shrinkage"],
        },
        "runs": runs,
        "summary": _summary(runs),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _target_policy_events(
    args: argparse.Namespace,
    split: Any,
    priors: dict[str, Any],
    reward_model: Any,
    progression_config: ProgressionRewardConfig,
) -> pd.DataFrame:
    tracer = _fit_tracer(
        split,
        model=args.model,
        user_col=args.user_col,
        item_col=args.item_col,
        correct_col=args.correct_col,
        timestamp_col=args.timestamp_col,
        item_difficulty_col=args.item_difficulty_col,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.seed,
        device=args.device,
    )
    difficulty_by_item = None
    if args.item_difficulty_col is not None and args.item_difficulty_col in split.train.columns:
        difficulty_by_item = {
            item_id: float(value)
            for item_id, value in split.train.groupby(args.item_col)[args.item_difficulty_col].mean().items()
        }
    concept_by_item = {
        item_id: value
        for item_id, value in split.train.groupby(args.item_col)[args.concept_col].agg(_mode_or_first).items()
    }
    support_by_item, support_by_concept = _support_tables(split, concept_col=args.concept_col)
    events = build_kt_policy_ope_events(
        tracer,
        split,
        candidate_size=args.candidate_size,
        max_events=args.max_events,
        random_state=args.seed,
        target_correct=args.target_correct,
        policy="support_delayed_gain",
        reward_mode="progression",
        difficulty_by_item=difficulty_by_item,
        concept_by_item=concept_by_item,
        progression_config=progression_config,
        delayed_gain_priors=priors,
        delayed_gain_reward_model=reward_model,
        support_by_item=support_by_item,
        support_by_concept=support_by_concept,
    )
    events = attach_delayed_gain_rewards(
        events,
        split,
        concept_col=args.concept_col,
        future_window=args.delayed_gain_window,
    )
    return events.dropna(subset=["delayed_gain_reward"]).copy()


def _training_frame_diagnostics(frame: pd.DataFrame, model: Any) -> dict[str, Any]:
    if frame.empty:
        return diagnose_delayed_gain_predictions([], [])
    rows = frame[FEATURE_NAMES].to_dict("records")
    predictions = model.predict_many(rows)
    return diagnose_delayed_gain_predictions(frame["delayed_gain_reward"].tolist(), predictions)


def _event_diagnostics(events: pd.DataFrame) -> dict[str, Any]:
    target_matches = events[events["target_probability"] > 0.0]
    return {
        "n_events": int(len(events)),
        "target_match_rate": float(events["target_probability"].mean()) if len(events) else 0.0,
        "target_effective_matches": float(events["target_probability"].sum()) if len(events) else 0.0,
        "logging_reward_mean": float(events["delayed_gain_reward"].mean()) if len(events) else None,
        "direct_target_value_mean": float(events["target_value"].mean()) if len(events) else None,
        "direct_random_value_mean": float(events["random_value"].mean()) if len(events) else None,
        "logged_action_diagnostics": diagnose_delayed_gain_predictions(
            events["delayed_gain_reward"].tolist(),
            events["logged_action_value"].tolist(),
        ),
        "target_match_diagnostics": diagnose_delayed_gain_predictions(
            target_matches["delayed_gain_reward"].tolist(),
            target_matches["logged_action_value"].tolist(),
        ),
    }


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    table = []
    for run in runs:
        policy = run["target_policy"]
        target_diag = policy["target_match_diagnostics"]
        logged_diag = policy["logged_action_diagnostics"]
        table.append(
            {
                "reward_model_example_weighting": run["reward_model_example_weighting"],
                "validation_rmse": run["model_report"]["validation_rmse"],
                "cross_fit_rmse": run["model_report"]["cross_fit_rmse"],
                "target_match_n": target_diag["n"],
                "target_match_bias": target_diag["bias"],
                "target_match_rmse": target_diag["rmse"],
                "logged_action_bias": logged_diag["bias"],
                "logged_action_rmse": logged_diag["rmse"],
                "direct_target_value_mean": policy["direct_target_value_mean"],
                "logging_reward_mean": policy["logging_reward_mean"],
            }
        )
    return {"table": table}


def _config(args: argparse.Namespace, frame: pd.DataFrame, training_frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "input_rows": float(len(frame)),
        "training_examples": float(len(training_frame)),
        "model": args.model,
        "test_fraction": args.test_fraction,
        "candidate_size": float(args.candidate_size),
        "max_events": None if args.max_events is None else float(args.max_events),
        "delayed_gain_window": float(args.delayed_gain_window),
        "target_correct": args.target_correct,
        "reward_model_max_examples": float(args.reward_model_max_examples),
        "reward_model_cross_fit_folds": float(args.reward_model_cross_fit_folds),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
