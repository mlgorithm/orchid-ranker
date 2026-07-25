#!/usr/bin/env python3
"""Validate a logged adaptive policy against a fixed static baseline.

This is a research/release tool, not a second Orchid API. It consumes completed
decision logs created by ``AdaptiveRanker.recommend_and_log`` and reports
whether the logged evidence supports a controlled rollout.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orchid_ranker.adaptive_schema import parse_candidate_list, validate_logged_decisions  # noqa: E402
from orchid_ranker.ope import LoggedPolicyReport, bootstrap_compare_logged_policies  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for one validation run."""
    parser = argparse.ArgumentParser(
        description="Validate logged adaptive recommendations against a static item-outcome baseline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Completed decision-log CSV, JSON, or JSON Lines path",
    )
    parser.add_argument("--reward-col", default="reward", help="Observed numeric reward column")
    parser.add_argument("--policy-version", default=None, help="Evaluate exactly this logged policy version")
    parser.add_argument("--evaluation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--smoothing",
        type=float,
        default=10.0,
        help="Pseudo-observation count for the static item-outcome baseline.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--min-evaluation-events", type=int, default=100)
    parser.add_argument("--min-evaluation-users", type=int, default=30)
    parser.add_argument("--min-baseline-coverage", type=float, default=0.05)
    parser.add_argument("--min-baseline-ess-fraction", type=float, default=0.05)
    parser.add_argument("--max-clipped-fraction", type=float, default=0.20)
    parser.add_argument("--min-uplift", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--report-md", type=Path, default=None, help="Optional reviewer-friendly Markdown report path")
    parser.add_argument(
        "--benchmark-name",
        default="Logged adaptive-policy validation",
        help="Human-readable report title",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero when the evidence gate is not met after writing reports.",
    )
    return parser.parse_args(argv)


def _validate_arguments(args: argparse.Namespace) -> None:
    if not 0.0 < float(args.evaluation_fraction) < 1.0:
        raise ValueError("evaluation_fraction must be in (0, 1)")
    if float(args.smoothing) < 0.0:
        raise ValueError("smoothing must be non-negative")
    if int(args.n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if args.max_weight is not None and float(args.max_weight) <= 0.0:
        raise ValueError("max_weight must be positive when supplied")
    if int(args.min_evaluation_events) < 1 or int(args.min_evaluation_users) < 1:
        raise ValueError("minimum evaluation counts must be >= 1")
    if not 0.0 <= float(args.min_baseline_coverage) <= 1.0:
        raise ValueError("min_baseline_coverage must be in [0, 1]")
    if not 0.0 <= float(args.min_baseline_ess_fraction) <= 1.0:
        raise ValueError("min_baseline_ess_fraction must be in [0, 1]")
    if not 0.0 <= float(args.max_clipped_fraction) <= 1.0:
        raise ValueError("max_clipped_fraction must be in [0, 1]")


def _select_policy_version(frame: pd.DataFrame, policy_version: Optional[str]) -> pd.DataFrame:
    versions = frame["policy_version"].astype(str)
    if policy_version is not None:
        selected = frame.loc[versions == str(policy_version)].copy()
        if selected.empty:
            raise ValueError(f"policy_version={policy_version!r} does not appear in the decision log")
        return selected
    unique_versions = sorted(versions.unique().tolist())
    if len(unique_versions) != 1:
        raise ValueError(
            "decision log contains multiple policy versions; pass --policy-version to evaluate one version"
        )
    return frame.copy()


def _normalize_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Require numeric, non-negative timestamps before a chronological split."""
    work = frame.copy()
    try:
        timestamps = pd.to_numeric(work["timestamp"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp values must be numeric for chronological validation") from exc
    if timestamps.isna().any() or (timestamps < 0).any():
        raise ValueError("timestamp values must be non-negative")
    work["timestamp"] = timestamps
    return work


def chronological_split(frame: pd.DataFrame, *, evaluation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a decision log at a timestamp boundary without temporal leakage."""
    ordered = frame.sort_values(["timestamp", "decision_id"], kind="mergesort").reset_index(drop=True)
    unique_timestamps = ordered["timestamp"].drop_duplicates().tolist()
    if len(unique_timestamps) < 2:
        raise ValueError("validation requires at least two distinct decision timestamps")
    target_events = int(math.ceil(len(ordered) * float(evaluation_fraction)))
    boundaries = unique_timestamps[1:]
    boundary = min(
        boundaries,
        key=lambda candidate: (abs(int((ordered["timestamp"] >= candidate).sum()) - target_events), candidate),
    )
    training = ordered.loc[ordered["timestamp"] < boundary].copy()
    evaluation = ordered.loc[ordered["timestamp"] >= boundary].copy()
    if training.empty or evaluation.empty:
        raise ValueError("could not construct non-empty chronological training and evaluation windows")
    if training["timestamp"].max() >= evaluation["timestamp"].min():
        raise RuntimeError("chronological split is not strictly ordered")
    return training, evaluation


def _smoothed_item_scores(training: pd.DataFrame, *, reward_col: str, smoothing: float) -> tuple[dict[Any, float], float]:
    """Estimate a static item's value from earlier inverse-propensity-weighted outcomes."""
    weighted = training[["chosen_item_id", reward_col, "propensity"]].copy()
    weighted["__weight"] = 1.0 / weighted["propensity"].astype(float)
    weighted["__weighted_reward"] = weighted[reward_col].astype(float) * weighted["__weight"]
    total_weight = float(weighted["__weight"].sum())
    global_mean = float(weighted["__weighted_reward"].sum() / total_weight)
    grouped = weighted.groupby("chosen_item_id", dropna=False)[["__weighted_reward", "__weight"]].sum()
    scores = {
        item_id: float(
            (float(row["__weighted_reward"]) + smoothing * global_mean)
            / (float(row["__weight"]) + smoothing)
        )
        for item_id, row in grouped.iterrows()
    }
    return scores, global_mean


def _static_choice(candidates: list[Any], *, item_scores: dict[Any, float], fallback_score: float) -> Any:
    if not candidates:
        raise ValueError("candidate_item_ids must not be empty")
    return min(candidates, key=lambda item_id: (-item_scores.get(item_id, fallback_score), str(item_id)))


def build_validation_payload(args: argparse.Namespace, frame: pd.DataFrame) -> dict[str, Any]:
    """Build a leakage-resistant comparison of the logged policy and a static baseline."""
    validated = validate_logged_decisions(frame, reward_col=args.reward_col)
    work = _normalize_timestamps(_select_policy_version(validated, args.policy_version))
    training, evaluation = chronological_split(work, evaluation_fraction=float(args.evaluation_fraction))
    item_scores, fallback_score = _smoothed_item_scores(
        training,
        reward_col=args.reward_col,
        smoothing=float(args.smoothing),
    )

    evaluation = evaluation.copy()
    evaluation["adaptive_probability"] = evaluation["propensity"].astype(float)
    evaluation["static_choice"] = [
        _static_choice(
            parse_candidate_list(candidates),
            item_scores=item_scores,
            fallback_score=fallback_score,
        )
        for candidates in evaluation["candidate_item_ids"]
    ]
    evaluation["static_probability"] = (
        evaluation["chosen_item_id"].eq(evaluation["static_choice"]).astype(float)
    )

    comparison = bootstrap_compare_logged_policies(
        evaluation,
        reward_col=args.reward_col,
        propensity_col="propensity",
        target_probability_col="adaptive_probability",
        baseline_probability_col="static_probability",
        max_weight=args.max_weight,
        n_bootstrap=int(args.n_bootstrap),
        random_state=int(args.random_state),
        cluster_col="user_id",
    )
    base = comparison.base
    baseline = base.baseline
    reasons = _gate_reasons(args, evaluation=evaluation, uplift_ci_low=comparison.bootstrap_ci_low, baseline=baseline)

    return {
        "artifact_schema": "orchid-logged-policy-validation/v1",
        "benchmark_name": str(args.benchmark_name),
        "policy_version": str(work["policy_version"].iloc[0]),
        "method": {
            "target": "The logged adaptive behavior, evaluated from immutable decision records and recorded propensities.",
            "baseline": "A propensity-weighted, smoothed item-outcome policy fit only on the earlier chronological window.",
            "uncertainty": "User-cluster percentile bootstrap over the later chronological window.",
            "smoothing": float(args.smoothing),
            "n_bootstrap": int(args.n_bootstrap),
            "max_weight": args.max_weight,
        },
        "split": {
            "training_events": int(len(training)),
            "evaluation_events": int(len(evaluation)),
            "training_users": int(training["user_id"].nunique()),
            "evaluation_users": int(evaluation["user_id"].nunique()),
            "training_start": _jsonable_scalar(training["timestamp"].min()),
            "training_end": _jsonable_scalar(training["timestamp"].max()),
            "evaluation_start": _jsonable_scalar(evaluation["timestamp"].min()),
            "evaluation_end": _jsonable_scalar(evaluation["timestamp"].max()),
        },
        "adaptive_policy": base.target.to_dict(),
        "static_baseline": base.baseline.to_dict(),
        "uplift": {
            "estimate": float(base.uplift),
            "standard_error": float(base.standard_error),
            "normal_ci_low": float(base.ci_low),
            "normal_ci_high": float(base.ci_high),
            "bootstrap_standard_error": float(comparison.bootstrap_standard_error),
            "bootstrap_ci_low": float(comparison.bootstrap_ci_low),
            "bootstrap_ci_high": float(comparison.bootstrap_ci_high),
        },
        "evidence_gate": {
            "allowed": not reasons,
            "reasons": reasons,
            "minimums": {
                "evaluation_events": int(args.min_evaluation_events),
                "evaluation_users": int(args.min_evaluation_users),
                "baseline_coverage": float(args.min_baseline_coverage),
                "baseline_ess_fraction": float(args.min_baseline_ess_fraction),
                "max_clipped_fraction": float(args.max_clipped_fraction),
                "uplift": float(args.min_uplift),
            },
        },
    }


def _gate_reasons(
    args: argparse.Namespace,
    *,
    evaluation: pd.DataFrame,
    uplift_ci_low: float,
    baseline: LoggedPolicyReport,
) -> list[str]:
    reasons: list[str] = []
    if len(evaluation) < int(args.min_evaluation_events):
        reasons.append("evaluation window has too few decisions")
    if int(evaluation["user_id"].nunique()) < int(args.min_evaluation_users):
        reasons.append("evaluation window has too few users")
    if float(baseline.coverage) < float(args.min_baseline_coverage):
        reasons.append("static baseline has insufficient logged-action coverage")
    ess_fraction = float(baseline.effective_sample_size) / max(1, int(baseline.n_events))
    if ess_fraction < float(args.min_baseline_ess_fraction):
        reasons.append("static baseline has insufficient effective sample size")
    if float(baseline.clipped_fraction) > float(args.max_clipped_fraction):
        reasons.append("static baseline requires too much propensity clipping")
    if float(uplift_ci_low) < float(args.min_uplift):
        reasons.append("uplift confidence interval does not clear the required improvement")
    return reasons


def _jsonable_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def read_decision_log(path: Path) -> pd.DataFrame:
    """Load a completed decision log while preserving structured candidate lists."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    return pd.read_csv(path)


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render the machine-readable result as a compact reviewer report."""
    split = payload["split"]
    uplift = payload["uplift"]
    adaptive = payload["adaptive_policy"]
    baseline = payload["static_baseline"]
    gate = payload["evidence_gate"]
    status = "PASS — eligible for a controlled rollout" if gate["allowed"] else "INCONCLUSIVE — do not claim improvement"
    lines = [
        f"# {payload['benchmark_name']}",
        "",
        f"**Policy version:** `{payload['policy_version']}`",
        "",
        f"## {status}",
        "",
        "| Check | Result |",
        "| --- | ---: |",
        f"| Training decisions | {split['training_events']} |",
        f"| Evaluation decisions | {split['evaluation_events']} |",
        f"| Evaluation users | {split['evaluation_users']} |",
        f"| Adaptive estimated value | {adaptive['value']:.4f} |",
        f"| Static estimated value | {baseline['value']:.4f} |",
        f"| Uplift | {uplift['estimate']:.4f} |",
        f"| Bootstrap uplift interval | [{uplift['bootstrap_ci_low']:.4f}, {uplift['bootstrap_ci_high']:.4f}] |",
        f"| Static baseline coverage | {baseline['coverage']:.1%} |",
        f"| Static baseline effective sample size | {baseline['effective_sample_size']:.1f} |",
        "",
        "## Interpretation",
        "",
        "This is an observational, chronological comparison of the recorded adaptive behavior against "
        "a static baseline learned only from earlier decisions. A passing result supports a guarded live "
        "experiment; it does not establish causal benefit outside this logged population and candidate set.",
        "",
        "## Gate reasons",
        "",
    ]
    if gate["reasons"]:
        lines.extend(f"- {reason}" for reason in gate["reasons"])
    else:
        lines.append("- All configured evidence checks passed.")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Run one logged-policy validation and emit JSON plus optional reports."""
    args = parse_args(argv)
    _validate_arguments(args)
    payload = build_validation_payload(args, read_decision_log(args.data))
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if args.report_md is not None:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(render_markdown_report(payload) + "\n", encoding="utf-8")
    if args.require_pass and not payload["evidence_gate"]["allowed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
