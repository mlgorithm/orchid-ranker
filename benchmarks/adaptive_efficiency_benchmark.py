#!/usr/bin/env python3
"""Consolidated adaptive-learning efficiency benchmark.

This benchmark answers three questions in one artifact:

1. How well do KT models predict held-out correctness?
2. Which next-item policy/reward settings improve offline value?
3. How expensive are the runs in wall-clock time?
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orchid_ranker.kt_benchmark import run_kt_benchmark  # noqa: E402
from orchid_ranker.policy_benchmark import run_kt_policy_ope_seed_sweep  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive KT + policy + runtime benchmark.")
    parser.add_argument("--data", type=Path, required=True, help="Interaction CSV path")
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--correct-col", default="correct")
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--item-difficulty-col", default=None)
    parser.add_argument("--concept-col", default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional input row cap for smoke tests")
    parser.add_argument("--models", nargs="+", choices=["sakt", "akt"], default=["akt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--policy-targets", nargs="+", type=float, default=[0.70, 0.90])
    parser.add_argument("--policy-rewards", nargs="+", choices=["correctness", "progression", "delayed_gain"], default=["progression", "delayed_gain"])
    parser.add_argument("--include-kt-value-policy", action="store_true", help="Also evaluate KTValuePolicy on correctness reward")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--logging-propensity-col", default=None)
    parser.add_argument("--delayed-gain-window", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.data)
    if args.max_rows is not None:
        frame = frame.head(max(1, int(args.max_rows))).copy()

    started = time.perf_counter()
    quality = _run_quality(args, frame)
    policies = _run_policies(args, frame)
    payload = {
        "config": _config(args, frame),
        "quality": quality,
        "policy": policies,
        "summary": _summary(quality, policies, total_seconds=time.perf_counter() - started),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _run_quality(args: argparse.Namespace, frame: pd.DataFrame) -> dict[str, Any]:
    runs = []
    for model in args.models:
        for seed in args.seeds:
            started = time.perf_counter()
            metrics = run_kt_benchmark(
                frame,
                model=model,
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
                random_state=seed,
                device=args.device,
            )
            elapsed = time.perf_counter() - started
            model_metrics = metrics[model]
            test_events = float(metrics["split"]["test_events"])
            runs.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "seconds": elapsed,
                    "events_per_second": test_events / elapsed if elapsed > 0.0 else None,
                    "metrics": model_metrics,
                    "item_mean": metrics["item_mean"],
                    "split": metrics["split"],
                }
            )
    return {
        "runs": runs,
        "summary": _quality_summary(runs),
    }


def _run_policies(args: argparse.Namespace, frame: pd.DataFrame) -> dict[str, Any]:
    runs = []
    for spec in _policy_specs(args):
        started = time.perf_counter()
        result = run_kt_policy_ope_seed_sweep(
            frame,
            seeds=args.seeds,
            model="akt",
            user_col=args.user_col,
            item_col=args.item_col,
            correct_col=args.correct_col,
            timestamp_col=args.timestamp_col,
            item_difficulty_col=args.item_difficulty_col,
            concept_col=args.concept_col,
            policy=spec["policy"],
            reward_mode=spec["reward_mode"],
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
            target_correct=spec["target_correct"],
            device=args.device,
        )
        elapsed = time.perf_counter() - started
        n_events = float(result["summary"].get("n_events_mean", args.max_events) or args.max_events)
        n_runs = float(result["summary"]["n_runs"])
        runs.append(
            {
                **spec,
                "seconds": elapsed,
                "events_per_second": (n_events * n_runs) / elapsed if elapsed > 0.0 else None,
                "result": result,
            }
        )
    return {
        "runs": runs,
        "summary": _policy_summary(runs),
    }


def _policy_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for target in args.policy_targets:
        if args.include_kt_value_policy and "correctness" in args.policy_rewards:
            specs.append({"policy": "kt_value", "reward_mode": "correctness", "target_correct": float(target)})
        for reward in args.policy_rewards:
            if reward in {"progression", "delayed_gain"} and args.concept_col is None:
                continue
            specs.append({"policy": "progression", "reward_mode": reward, "target_correct": float(target)})
            if reward == "delayed_gain":
                specs.append({"policy": "delayed_gain", "reward_mode": reward, "target_correct": float(target)})
                specs.append({"policy": "support_delayed_gain", "reward_mode": reward, "target_correct": float(target)})

    deduped = []
    seen = set()
    for spec in specs:
        key = (spec["policy"], spec["reward_mode"], spec["target_correct"])
        if key not in seen:
            seen.add(key)
            deduped.append(spec)
    return deduped


def _quality_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_model.setdefault(run["model"], []).append(run)
    summary = {}
    for model, model_runs in by_model.items():
        summary[model] = {
            metric: _mean_std([run["metrics"][metric] for run in model_runs])
            for metric in ["accuracy", "auc", "brier", "log_loss", "ece"]
        }
        summary[model]["seconds"] = _mean_std([run["seconds"] for run in model_runs])
        summary[model]["events_per_second"] = _mean_std([run["events_per_second"] for run in model_runs])
    if runs:
        summary["item_mean"] = {
            metric: _mean_std([run["item_mean"][metric] for run in runs])
            for metric in ["accuracy", "auc", "brier", "log_loss", "ece"]
        }
    return summary


def _policy_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    table = []
    for run in runs:
        summary = run["result"]["summary"]
        table.append(
            {
                "policy": run["policy"],
                "reward_mode": run["reward_mode"],
                "target_correct": run["target_correct"],
                "uplift_mean": summary["uplift_mean"],
                "uplift_ci_low": summary["uplift_ci_low"],
                "uplift_ci_high": summary["uplift_ci_high"],
                "target_value_mean": summary["target_value_mean"],
                "baseline_value_mean": summary["baseline_value_mean"],
                "target_match_rate_mean": summary["target_match_rate_mean"],
                "target_ess_mean": summary["target_ess_mean"],
                "seconds": run["seconds"],
                "events_per_second": run["events_per_second"],
            }
        )
    positive = [row for row in table if row["uplift_mean"] > 0.0]
    best = max(table, key=lambda row: row["uplift_mean"], default=None)
    return {
        "best_by_uplift": best,
        "positive_count": len(positive),
        "table": table,
    }


def _summary(quality: dict[str, Any], policy: dict[str, Any], *, total_seconds: float) -> dict[str, Any]:
    quality_models = {k: v for k, v in quality["summary"].items() if k != "item_mean"}
    best_quality = None
    if quality_models:
        best_name, best_data = max(quality_models.items(), key=lambda item: item[1]["auc"]["mean"])
        best_quality = {"model": best_name, "auc": best_data["auc"]["mean"], "accuracy": best_data["accuracy"]["mean"]}
    return {
        "total_seconds": total_seconds,
        "best_quality_model": best_quality,
        "best_policy": policy["summary"]["best_by_uplift"],
    }


def _mean_std(values: list[Any]) -> dict[str, float]:
    array = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def _config(args: argparse.Namespace, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "data": str(args.data),
        "rows": int(len(frame)),
        "users": int(frame[args.user_col].nunique()),
        "items": int(frame[args.item_col].nunique()),
        "models": list(args.models),
        "seeds": [int(seed) for seed in args.seeds],
        "policy_targets": [float(value) for value in args.policy_targets],
        "policy_rewards": list(args.policy_rewards),
        "test_fraction": float(args.test_fraction),
        "candidate_size": int(args.candidate_size),
        "max_events": None if args.max_events is None else int(args.max_events),
        "delayed_gain_window": int(args.delayed_gain_window),
        "max_seq_len": int(args.max_seq_len),
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "device": args.device,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
