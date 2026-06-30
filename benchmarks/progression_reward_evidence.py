#!/usr/bin/env python3
"""Evidence benchmark for the progression reward.

The progression policy is built on a hand-designed reward
(:func:`orchid_ranker.progression_reward.expected_progression_reward`) whose
weights encode pedagogical priors -- zone-of-proximal-development mastery gain,
stretch fit, easy/hard/repetition penalties. This benchmark asks whether that
reward actually carries signal about *realized* learning gain, and which of its
terms earn their weight.

Method (observational, confound-aware):

1. Fit a KT tracer on a chronological holdout split.
2. Replay each held-out decision. For the item the learner actually attempted,
   recover the reward inputs (p_correct, competence, difficulty, recent
   repetition) from the progression policy and compute the full reward
   breakdown *before* seeing the outcome.
3. Measure the realized forward learning gain for that decision as the
   normalized same-concept improvement over a future window
   (``attach_delayed_gain_rewards`` -- a monotone transform of
   future_same_concept_accuracy - prior_accuracy).
4. Correlate predicted reward with realized gain, but control for the obvious
   confounds:
   * overall Spearman,
   * **partial** Spearman of reward vs gain holding p_correct and competence
     fixed -- this isolates whether the reward's pedagogical *shape* (stretch /
     ZPD terms) adds signal beyond naive correctness and current ability,
   * within-competence-bin stratified calibration buckets.
5. Decompose per reward term, and run correctness-only / random ablations.

The headline number is the partial Spearman with a bootstrap CI: if it is
positive with a CI excluding zero, the reward is evidenced to add signal beyond
correctness; otherwise the per-term table shows which weights to retune.

Public tutoring logs rarely include true platform propensities, so realized
gain is an observational proxy, not a causal estimate. The partial-correlation
and stratification controls reduce -- but do not eliminate -- confounding.

Example
-------
    python benchmarks/progression_reward_evidence.py \
        --data benchmarks/fixtures/assistments_tiny_raw.csv \
        --concept-col skill_id --epochs 1 --max-events 500 --seeds 11
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as _sps

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orchid_ranker.kt_benchmark import _binary_labels, time_ordered_user_split  # noqa: E402
from orchid_ranker.learning_policy import ProgressionValuePolicy  # noqa: E402
from orchid_ranker.policy_benchmark import (  # noqa: E402
    _candidate_pool,
    _fit_tracer,
    attach_delayed_gain_rewards,
)
from orchid_ranker.progression_reward import (  # noqa: E402
    ProgressionRewardConfig,
    expected_progression_reward,
)

# Reward-breakdown terms and the sign with which each enters the reward sum.
# A term whose realized-gain correlation has the *wrong* sign relative to this
# is pulling the reward away from realized learning gain.
_TERMS = {
    "expected_outcome_value": +1,
    "mastery_gain": +1,
    "stretch_fit": +1,
    "difficulty_bonus": +1,
    "easy_penalty": -1,
    "hard_penalty": -1,
    "repetition_penalty": -1,
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--future-window", type=int, default=5)
    parser.add_argument("--competence-bins", type=int, default=4)
    parser.add_argument("--reward-buckets", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-correct", type=float, default=0.70)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.data)
    runs = [_run_seed(frame, args=args, seed=int(seed)) for seed in args.seeds]
    result = {
        "dataset": str(args.data),
        "assumptions": {
            "split": "chronological_by_user",
            "realized_gain": "normalized future same-concept improvement (observational proxy)",
            "future_window": float(args.future_window),
            "confound_controls": ["partial_spearman|p_correct,competence", "within_competence_bin_strata"],
            "max_events": float(args.max_events),
            "target_correct": float(args.target_correct),
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
    difficulty_by_item, concept_by_item = _difficulty_concept_maps(
        split.train,
        item_col=args.item_col,
        correct_col=args.correct_col,
        difficulty_col=args.item_difficulty_col,
        concept_col=args.concept_col,
        threshold=args.target_correct,
    )
    decisions = _collect_decisions(
        split,
        tracer=tracer,
        difficulty_by_item=difficulty_by_item,
        concept_by_item=concept_by_item,
        candidate_size=args.candidate_size,
        max_events=args.max_events,
        target_correct=args.target_correct,
        seed=seed,
    )
    realized = attach_delayed_gain_rewards(
        decisions,
        split,
        concept_col=args.concept_col,
        future_window=args.future_window,
    ).dropna(subset=["delayed_gain_reward"])

    analysis = _analyze(
        realized,
        competence_bins=args.competence_bins,
        reward_buckets=args.reward_buckets,
        bootstrap=args.bootstrap,
        seed=seed,
    )
    return {
        "seed": int(seed),
        "n_decisions": float(len(decisions)),
        "n_with_realized_gain": float(len(realized)),
        "split": {
            "train_events": float(len(split.train)),
            "test_events": float(len(split.test)),
            "test_users": float(split.test[args.user_col].nunique()),
        },
        **analysis,
    }


def _difficulty_concept_maps(
    train: pd.DataFrame,
    *,
    item_col: str,
    correct_col: str,
    difficulty_col: str,
    concept_col: str,
    threshold: float,
) -> tuple[dict[Any, float], dict[Any, Any]]:
    if difficulty_col in train.columns:
        difficulty = {
            item: float(np.clip(value, 0.0, 1.0))
            for item, value in train.groupby(item_col)[difficulty_col].mean().items()
        }
    else:
        labels = pd.Series(_binary_labels(train[correct_col].tolist(), threshold=threshold), index=train.index)
        work = pd.DataFrame({item_col: train[item_col].values, "__l__": labels.values})
        global_correct = float(work["__l__"].mean())
        grouped = work.groupby(item_col)["__l__"].agg(["sum", "count"])
        difficulty = {
            item: float(np.clip(1.0 - (row["sum"] + global_correct) / (row["count"] + 1.0), 0.0, 1.0))
            for item, row in grouped.iterrows()
        }
    if concept_col in train.columns:
        concept = {
            item: group[concept_col].mode(dropna=True).iloc[0]
            if not group[concept_col].mode(dropna=True).empty
            else group[concept_col].iloc[0]
            for item, group in train.groupby(item_col, sort=False)
        }
    else:
        concept = {item: item for item in train[item_col].drop_duplicates().tolist()}
    return difficulty, concept


def _collect_decisions(
    split: Any,
    *,
    tracer: Any,
    difficulty_by_item: dict[Any, float],
    concept_by_item: dict[Any, Any],
    candidate_size: int,
    max_events: int,
    target_correct: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    config = ProgressionRewardConfig(target_correct=target_correct)
    known_items = sorted(split.train[split.item_col].drop_duplicates().tolist(), key=lambda v: str(v))
    known_set = set(known_items)
    policy = ProgressionValuePolicy(
        tracer,
        difficulty_by_item=difficulty_by_item,
        concept_by_item=concept_by_item,
        config=config,
    ).seed_history(
        split.train,
        user_col=split.user_col,
        item_col=split.item_col,
        correct_col=split.correct_col,
        timestamp_col=split.timestamp_col,
    )
    test = split.test.sort_values([split.user_col, split.timestamp_col], kind="mergesort")
    if max_events:
        test = test.head(max_events).copy()

    rows: list[dict[str, Any]] = []
    for event_id, record in enumerate(test.itertuples(index=False), start=1):
        data = record._asdict()
        user_id = data[split.user_col]
        logged_item = data[split.item_col]
        if logged_item not in known_set:
            continue
        label = int(_binary_labels([data[split.correct_col]])[0])
        candidates = _candidate_pool(logged_item, known_items=known_items, candidate_size=candidate_size, rng=rng)
        ranked = {rec.item_id: rec for rec in policy.rank(user_id, candidates, top_k=len(candidates))}
        rec = ranked[logged_item]
        breakdown = expected_progression_reward(
            p_correct=rec.p_correct,
            difficulty=rec.difficulty,
            competence=rec.competence,
            recent_repetition=rec.recent_repetition,
            config=config,
        )
        out = {
            "event_id": int(event_id),
            "user_id": user_id,
            "logged_item_id": logged_item,
            "label": float(label),
            "p_correct": float(rec.p_correct),
            "competence": float(rec.competence),
            "difficulty": float(rec.difficulty if rec.difficulty is not None else 0.5),
            "expected_reward": float(breakdown.expected_reward),
        }
        for term in _TERMS:
            out[term] = float(getattr(breakdown, term))
        rows.append(out)

        tracer.observe(user_id, logged_item, label)
        policy.record_outcome(user_id, logged_item, label)

    if not rows:
        raise ValueError("replay produced no decisions (check column names / data size)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Correlation analysis
# --------------------------------------------------------------------------- #
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(_sps.spearmanr(x, y).statistic)


def _partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    """Spearman correlation of x and y after rank-residualizing on controls."""
    n = len(x)
    if n < 5:
        return float("nan")
    rx = _sps.rankdata(x)
    ry = _sps.rankdata(y)
    cols = [_sps.rankdata(c) for c in controls]
    design = np.column_stack([*cols, np.ones(n)])
    bx, *_ = np.linalg.lstsq(design, rx, rcond=None)
    by, *_ = np.linalg.lstsq(design, ry, rcond=None)
    res_x = rx - design @ bx
    res_y = ry - design @ by
    if np.std(res_x) == 0 or np.std(res_y) == 0:
        return float("nan")
    return float(np.corrcoef(res_x, res_y)[0, 1])


def _bootstrap_ci(
    fn: Any,
    df: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, float]:
    point = fn(df)
    if not np.isfinite(point) or bootstrap <= 0 or len(df) < 5:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    samples = []
    for _ in range(bootstrap):
        boot = df.iloc[rng.choice(idx, size=len(idx), replace=True)]
        val = fn(boot)
        if np.isfinite(val):
            samples.append(val)
    if len(samples) < 2:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan")}
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def _stratified_buckets(
    df: pd.DataFrame,
    *,
    competence_bins: int,
    reward_buckets: int,
) -> dict[str, Any]:
    """Within each competence bin, bucket by predicted reward and report mean realized gain."""
    out: list[dict[str, Any]] = []
    monotonic_scores: list[float] = []
    try:
        df = df.assign(_cbin=pd.qcut(df["competence"], q=min(competence_bins, df["competence"].nunique()), duplicates="drop"))
    except (ValueError, IndexError):
        return {"strata": [], "mean_within_stratum_monotonicity": float("nan")}
    for cbin, cgroup in df.groupby("_cbin", observed=True):
        if len(cgroup) < reward_buckets:
            continue
        try:
            cgroup = cgroup.assign(
                _rbin=pd.qcut(cgroup["expected_reward"], q=reward_buckets, labels=False, duplicates="drop")
            )
        except ValueError:
            continue
        bucket_means = cgroup.groupby("_rbin")["delayed_gain_reward"].mean()
        if len(bucket_means) < 3:
            continue
        mono = _spearman(bucket_means.index.to_numpy(dtype=float), bucket_means.to_numpy())
        if np.isfinite(mono):
            monotonic_scores.append(mono)
        out.append(
            {
                "competence_bin": str(cbin),
                "n": float(len(cgroup)),
                "bucket_mean_gain": [float(v) for v in bucket_means.to_numpy()],
                "reward_vs_gain_monotonicity": mono,
            }
        )
    return {
        "strata": out,
        "mean_within_stratum_monotonicity": float(np.mean(monotonic_scores)) if monotonic_scores else float("nan"),
    }


def _analyze(
    df: pd.DataFrame,
    *,
    competence_bins: int,
    reward_buckets: int,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if len(df) < 5:
        return {"insufficient_data": True, "n": float(len(df))}
    gain = df["delayed_gain_reward"].to_numpy()
    controls = [df["p_correct"].to_numpy(), df["competence"].to_numpy()]

    overall = _bootstrap_ci(
        lambda d: _spearman(d["expected_reward"].to_numpy(), d["delayed_gain_reward"].to_numpy()),
        df,
        bootstrap=bootstrap,
        seed=seed,
    )
    partial = _bootstrap_ci(
        lambda d: _partial_spearman(
            d["expected_reward"].to_numpy(),
            d["delayed_gain_reward"].to_numpy(),
            [d["p_correct"].to_numpy(), d["competence"].to_numpy()],
        ),
        df,
        bootstrap=bootstrap,
        seed=seed + 1,
    )
    ablations = {
        "correctness_only_spearman": _spearman(df["p_correct"].to_numpy(), gain),
        "competence_only_spearman": _spearman(df["competence"].to_numpy(), gain),
        "random_spearman": _spearman(np.random.default_rng(seed).random(len(df)), gain),
    }
    per_term = {}
    for term, sign in _TERMS.items():
        values = df[term].to_numpy()
        per_term[term] = {
            "sign_in_reward": sign,
            "spearman": _spearman(values, gain),
            "partial_spearman": _partial_spearman(values, gain, controls),
        }

    stratified = _stratified_buckets(df, competence_bins=competence_bins, reward_buckets=reward_buckets)

    partial_low = partial["ci_low"]
    overall_est = overall["estimate"]
    corr_only = ablations["correctness_only_spearman"]
    evidenced = bool(
        np.isfinite(partial_low)
        and partial_low > 0.0
        and np.isfinite(overall_est)
        and (not np.isfinite(corr_only) or overall_est >= corr_only)
    )
    return {
        "overall_spearman": overall,
        "partial_spearman_given_pcorrect_competence": partial,
        "ablations": ablations,
        "per_term": per_term,
        "stratified": stratified,
        "verdict": {
            "evidenced": evidenced,
            "rule": "partial Spearman CI excludes 0 (low>0) AND overall >= correctness-only Spearman",
        },
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in runs if not r.get("insufficient_data")]
    if not usable:
        return {"insufficient_data": True}

    def _mean(path: list[str]) -> float:
        vals = []
        for r in usable:
            node: Any = r
            ok = True
            for key in path:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    ok = False
                    break
            if ok and isinstance(node, (int, float)) and np.isfinite(node):
                vals.append(float(node))
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "mean_overall_spearman": _mean(["overall_spearman", "estimate"]),
        "mean_partial_spearman": _mean(["partial_spearman_given_pcorrect_competence", "estimate"]),
        "mean_correctness_only_spearman": _mean(["ablations", "correctness_only_spearman"]),
        "mean_within_stratum_monotonicity": _mean(["stratified", "mean_within_stratum_monotonicity"]),
        "evidenced_fraction": float(
            np.mean([1.0 if r.get("verdict", {}).get("evidenced") else 0.0 for r in usable])
        ),
        "n_seeds": float(len(usable)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
