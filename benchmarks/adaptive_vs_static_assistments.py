#!/usr/bin/env python3
"""Compare adaptive-learning policies against static non-adaptive baselines.

This benchmark is intentionally scenario-based rather than a single leaderboard.
It answers three separate questions on chronological ASSISTments-style logs:

1. Which policy maximizes immediate correctness?
2. Which policy maximizes the progression reward proxy?
3. Which policy improves delayed same-concept gain?

Public tutoring logs usually do not include true platform propensities, so this
uses the same documented synthetic-uniform candidate logging assumption as the
KT policy OPE benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orchid_ranker.kt_benchmark import _binary_labels, time_ordered_user_split  # noqa: E402
from orchid_ranker.learning_policy import (  # noqa: E402
    DelayedGainValuePolicy,
    KTValuePolicy,
    ProgressionValuePolicy,
)
from orchid_ranker.ope import compare_logged_policies  # noqa: E402
from orchid_ranker.policy_benchmark import (  # noqa: E402
    _candidate_pool,
    _fit_tracer,
    attach_delayed_gain_rewards,
    estimate_delayed_gain_priors,
)
from orchid_ranker.progression_reward import (  # noqa: E402
    ProgressionRewardConfig,
    observed_progression_reward,
)

STATIC_POLICIES = (
    "static_popularity",
    "static_easiest",
    "static_target_70",
    "static_delayed_gain_prior",
)
ADAPTIVE_POLICIES = (
    "adaptive_kt_value",
    "adaptive_progression",
    "adaptive_delayed_gain",
)
ALL_POLICIES = (*STATIC_POLICIES, *ADAPTIVE_POLICIES)


@dataclass(frozen=True)
class ItemStats:
    support: dict[Any, float]
    correctness: dict[Any, float]
    difficulty: dict[Any, float]
    concept: dict[Any, Any]
    delayed_gain: dict[Any, float]
    global_delayed_gain: float


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scenario benchmark: adaptive policies vs static baselines on ASSISTments-style data."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--correct-col", default="correct")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--item-difficulty-col", default="difficulty")
    parser.add_argument("--concept-col", default="skill_id")
    parser.add_argument("--model", choices=["sakt", "akt"], default="akt")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-correct", type=float, default=0.70)
    parser.add_argument("--delayed-gain-window", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.data)
    runs = [
        _run_seed(frame, args=args, seed=int(seed))
        for seed in args.seeds
    ]
    result = {
        "dataset": str(args.data),
        "assumptions": {
            "split": "chronological_by_user",
            "logging": "synthetic_uniform_over_candidate_set",
            "candidate_size": float(args.candidate_size),
            "max_events": float(args.max_events),
            "baseline_policy": "random_uniform_candidate",
            "target_correct": float(args.target_correct),
        },
        "scenarios": {
            "immediate_correctness": "Observed held-out correctness label.",
            "progression_reward": "Observed progression proxy using a shared AKT-backed reward state.",
            "delayed_same_concept_gain": (
                "Future same-concept correctness improvement proxy; rows without future same-concept outcomes are dropped."
            ),
        },
        "summary": _aggregate_runs(runs),
        "runs": runs,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _run_seed(frame: pd.DataFrame, *, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    split = time_ordered_user_split(
        frame,
        user_col=args.user_col,
        item_col=args.item_col,
        correct_col=args.correct_col,
        timestamp_col=args.timestamp_col,
        test_fraction=args.test_fraction,
    )
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
        random_state=seed,
        device=args.device,
    )
    priors = estimate_delayed_gain_priors(
        split,
        concept_col=args.concept_col,
        future_window=args.delayed_gain_window,
    )
    stats = _item_stats(
        split.train,
        item_col=args.item_col,
        correct_col=args.correct_col,
        difficulty_col=args.item_difficulty_col,
        concept_col=args.concept_col,
        delayed_gain_priors=priors,
    )
    events = _build_events(
        split,
        tracer=tracer,
        stats=stats,
        candidate_size=args.candidate_size,
        max_events=args.max_events,
        random_state=seed,
        target_correct=args.target_correct,
    )
    delayed_events = attach_delayed_gain_rewards(
        events,
        split,
        concept_col=args.concept_col,
        future_window=args.delayed_gain_window,
    ).dropna(subset=["delayed_gain_reward"])

    scenario_frames = {
        "immediate_correctness": (events, "correctness_reward"),
        "progression_reward": (events, "progression_reward"),
        "delayed_same_concept_gain": (delayed_events, "delayed_gain_reward"),
    }
    reports: dict[str, dict[str, Any]] = {}
    for scenario, (scenario_events, reward_col) in scenario_frames.items():
        reports[scenario] = {}
        for policy in ALL_POLICIES:
            reports[scenario][policy] = _compare_policy(scenario_events, policy=policy, reward_col=reward_col)
    return {
        "seed": int(seed),
        "split": {
            "train_events": float(len(split.train)),
            "test_events": float(len(split.test)),
            "train_users": float(split.train[args.user_col].nunique()),
            "test_users": float(split.test[args.user_col].nunique()),
            "train_items": float(split.train[args.item_col].nunique()),
            "test_items": float(split.test[args.item_col].nunique()),
        },
        "scenarios": reports,
    }


def _build_events(
    split: Any,
    *,
    tracer: Any,
    stats: ItemStats,
    candidate_size: int,
    max_events: int,
    random_state: int,
    target_correct: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    known_items = sorted(split.train[split.item_col].drop_duplicates().tolist(), key=lambda value: str(value))
    known_set = set(known_items)
    config = ProgressionRewardConfig(target_correct=target_correct)
    kt_policy = KTValuePolicy(tracer, target_correct=target_correct, difficulty_by_item=stats.difficulty)
    progression_policy = ProgressionValuePolicy(
        tracer,
        difficulty_by_item=stats.difficulty,
        concept_by_item=stats.concept,
        config=config,
    ).seed_history(
        split.train,
        user_col=split.user_col,
        item_col=split.item_col,
        correct_col=split.correct_col,
        timestamp_col=split.timestamp_col,
    )
    delayed_policy = DelayedGainValuePolicy(
        tracer,
        difficulty_by_item=stats.difficulty,
        concept_by_item=stats.concept,
        item_gain_prior=stats.delayed_gain,
        global_gain_prior=stats.global_delayed_gain,
        config=config,
    ).seed_history(
        split.train,
        user_col=split.user_col,
        item_col=split.item_col,
        correct_col=split.correct_col,
        timestamp_col=split.timestamp_col,
    )
    reward_policy = ProgressionValuePolicy(
        tracer,
        difficulty_by_item=stats.difficulty,
        concept_by_item=stats.concept,
        config=config,
    ).seed_history(
        split.train,
        user_col=split.user_col,
        item_col=split.item_col,
        correct_col=split.correct_col,
        timestamp_col=split.timestamp_col,
    )
    rows: list[dict[str, Any]] = []
    test = _ordered(split.test, user_col=split.user_col, timestamp_col=split.timestamp_col)
    if max_events:
        test = test.head(max_events).copy()

    for event_id, row in enumerate(test.itertuples(index=False), start=1):
        data = row._asdict()
        user_id = data[split.user_col]
        logged_item = data[split.item_col]
        if logged_item not in known_set:
            continue
        label = int(_binary_labels([data[split.correct_col]])[0])
        candidates = _candidate_pool(
            logged_item,
            known_items=known_items,
            candidate_size=candidate_size,
            rng=rng,
        )
        reward_ranked = reward_policy.rank(user_id, candidates, top_k=len(candidates))
        logged_reward_rec = {rec.item_id: rec for rec in reward_ranked}[logged_item]
        progression_reward = observed_progression_reward(
            correct=label,
            p_correct=logged_reward_rec.p_correct,
            difficulty=logged_reward_rec.difficulty,
            competence=logged_reward_rec.competence,
            recent_repetition=logged_reward_rec.recent_repetition,
            config=config,
        )
        choices = {
            "static_popularity": _choose_static("static_popularity", candidates, stats, target_correct),
            "static_easiest": _choose_static("static_easiest", candidates, stats, target_correct),
            "static_target_70": _choose_static("static_target_70", candidates, stats, target_correct),
            "static_delayed_gain_prior": _choose_static("static_delayed_gain_prior", candidates, stats, target_correct),
            "adaptive_kt_value": kt_policy.rank(user_id, candidates, top_k=1)[0].item_id,
            "adaptive_progression": progression_policy.rank(user_id, candidates, top_k=1)[0].item_id,
            "adaptive_delayed_gain": delayed_policy.rank(user_id, candidates, top_k=1)[0].item_id,
        }
        out = {
            "event_id": int(event_id),
            "user_id": user_id,
            "logged_item_id": logged_item,
            "correctness_reward": float(label),
            "progression_reward": float(progression_reward),
            "candidate_size": float(len(candidates)),
            "logging_propensity": 1.0 / float(len(candidates)),
            "random_probability": 1.0 / float(len(candidates)),
        }
        for policy, item_id in choices.items():
            out[f"{policy}_probability"] = float(item_id == logged_item)
            out[f"{policy}_item_id"] = item_id
        rows.append(out)

        tracer.observe(user_id, logged_item, label)
        reward_policy.record_outcome(user_id, logged_item, label)
        progression_policy.record_outcome(user_id, logged_item, label)
        delayed_policy.record_outcome(user_id, logged_item, label)

    if not rows:
        raise ValueError("scenario replay produced no events")
    return pd.DataFrame(rows)


def _compare_policy(events: pd.DataFrame, *, policy: str, reward_col: str) -> dict[str, Any]:
    if events.empty:
        return {
            "n_events": 0,
            "estimator": "snips",
            "value": None,
            "baseline_value": None,
            "uplift": None,
            "ci_low": None,
            "ci_high": None,
            "match_rate": 0.0,
            "effective_sample_size": 0.0,
        }
    comparison = compare_logged_policies(
        events,
        reward_col=reward_col,
        propensity_col="logging_propensity",
        target_probability_col=f"{policy}_probability",
        baseline_probability_col="random_probability",
    )
    data = comparison.to_dict()
    return {
        "n_events": int(comparison.target.n_events),
        "estimator": comparison.estimator,
        "value": comparison.target.value,
        "baseline_value": comparison.baseline.value,
        "uplift": comparison.uplift,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "match_rate": float(events[f"{policy}_probability"].mean()),
        "effective_sample_size": comparison.target.effective_sample_size,
        "weight_max": comparison.target.weight_max,
        "raw": data,
    }


def _item_stats(
    train: pd.DataFrame,
    *,
    item_col: str,
    correct_col: str,
    difficulty_col: str,
    concept_col: str,
    delayed_gain_priors: dict[str, Any],
) -> ItemStats:
    work = train.copy()
    work["__label__"] = _binary_labels(work[correct_col].tolist())
    support = {item: float(value) for item, value in work.groupby(item_col).size().items()}
    correctness = {item: float(value) for item, value in work.groupby(item_col)["__label__"].mean().items()}
    if difficulty_col in work.columns:
        difficulty = {item: float(value) for item, value in work.groupby(item_col)[difficulty_col].mean().items()}
    else:
        difficulty = {item: float(1.0 - correctness.get(item, 0.5)) for item in support}
    if concept_col in work.columns:
        concept = {item: value for item, value in work.groupby(item_col)[concept_col].agg(_mode_or_first).items()}
    else:
        concept = {item: item for item in support}
    return ItemStats(
        support=support,
        correctness=correctness,
        difficulty=difficulty,
        concept=concept,
        delayed_gain=dict(delayed_gain_priors.get("item_gain_prior", {})),
        global_delayed_gain=float(delayed_gain_priors.get("global_gain_prior", 0.5)),
    )


def _choose_static(policy: str, candidates: list[Any], stats: ItemStats, target_correct: float) -> Any:
    if policy == "static_popularity":
        return max(candidates, key=lambda item: (stats.support.get(item, 0.0), str(item)))
    if policy == "static_easiest":
        return max(candidates, key=lambda item: (stats.correctness.get(item, 0.5), stats.support.get(item, 0.0), str(item)))
    if policy == "static_target_70":
        return max(
            candidates,
            key=lambda item: (
                -abs(stats.correctness.get(item, 0.5) - target_correct),
                stats.support.get(item, 0.0),
                str(item),
            ),
        )
    if policy == "static_delayed_gain_prior":
        return max(
            candidates,
            key=lambda item: (
                stats.delayed_gain.get(item, stats.global_delayed_gain),
                stats.support.get(item, 0.0),
                str(item),
            ),
        )
    raise ValueError(f"unknown static policy: {policy}")


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for scenario, policies in run["scenarios"].items():
            for policy, report in policies.items():
                if report["uplift"] is None:
                    continue
                rows.append(
                    {
                        "scenario": scenario,
                        "policy": policy,
                        "seed": run["seed"],
                        "value": report["value"],
                        "baseline_value": report["baseline_value"],
                        "uplift": report["uplift"],
                        "match_rate": report["match_rate"],
                        "effective_sample_size": report["effective_sample_size"],
                        "n_events": report["n_events"],
                    }
                )
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for (scenario, policy), group in frame.groupby(["scenario", "policy"], sort=True):
        uplifts = group["uplift"].to_numpy(dtype=float)
        ci_low, ci_high = _mean_ci(uplifts)
        n_events = float(group["n_events"].mean())
        ess = float(group["effective_sample_size"].mean())
        match = float(group["match_rate"].mean())
        summary.setdefault(scenario, {})[policy] = {
            "n_runs": int(len(group)),
            "target_value_mean": float(group["value"].mean()),
            "random_baseline_value_mean": float(group["baseline_value"].mean()),
            "uplift_mean": float(group["uplift"].mean()),
            "uplift_ci_low": ci_low,
            "uplift_ci_high": ci_high,
            "match_rate_mean": match,
            "effective_sample_size_mean": ess,
            "effective_sample_size_fraction_mean": float(ess / n_events) if n_events else 0.0,
            "n_events_mean": n_events,
            "gate_allowed": bool(ci_low > 0.0 and match >= 0.05 and (ess / n_events if n_events else 0.0) >= 0.05),
        }
    return summary


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean
    se = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return float(mean - 1.959963984540054 * se), float(mean + 1.959963984540054 * se)


def _ordered(frame: pd.DataFrame, *, user_col: str, timestamp_col: Optional[str]) -> pd.DataFrame:
    work = frame.copy()
    work["__orchid_order__"] = np.arange(len(work))
    sort_cols = [user_col]
    if timestamp_col is not None:
        sort_cols.append(timestamp_col)
    sort_cols.append("__orchid_order__")
    return work.sort_values(sort_cols, kind="mergesort").drop(columns=["__orchid_order__"])


def _mode_or_first(values: pd.Series) -> Any:
    modes = values.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return values.iloc[0]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
