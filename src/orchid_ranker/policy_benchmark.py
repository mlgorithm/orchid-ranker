"""Logged-policy benchmarks for KT-guided adaptive recommendation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .delayed_gain import fit_delayed_gain_reward_model
from .kt import AKTTracer, SAKTTracer
from .kt_benchmark import KTHoldoutSplit, _binary_labels, time_ordered_user_split
from .learning_policy import (
    DelayedGainValuePolicy,
    KTValuePolicy,
    ProgressionValuePolicy,
    SupportConstrainedDelayedGainPolicy,
)
from .ope import compare_logged_policies
from .progression_reward import ProgressionRewardConfig, observed_progression_reward

__all__ = [
    "KTPolicyOPEReport",
    "KTPolicyOPESweepReport",
    "attach_delayed_gain_rewards",
    "build_kt_policy_ope_events",
    "estimate_delayed_gain_priors",
    "run_kt_policy_ope_benchmark",
    "run_kt_policy_ope_seed_sweep",
]


@dataclass(frozen=True)
class KTPolicyOPEReport:
    """Summary for a KT-guided logged-policy OPE replay."""

    n_events: int
    candidate_size_mean: float
    logging_reward: float
    target_match_rate: float
    random_match_probability: float
    target_value_mean: Optional[float]
    random_value_mean: Optional[float]
    comparison: Dict[str, Any]
    split: Dict[str, float]
    assumptions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KTPolicyOPESweepReport:
    """Multi-seed summary for KT policy OPE benchmarks."""

    summary: Dict[str, Any]
    runs: list[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_kt_policy_ope_events(
    tracer: SAKTTracer,
    split: KTHoldoutSplit,
    *,
    candidate_item_ids: Optional[Sequence[Any]] = None,
    candidate_size: int = 20,
    max_events: Optional[int] = None,
    random_state: Optional[int] = 42,
    target_correct: float = 0.70,
    stretch_weight: float = 1.0,
    uncertainty_weight: float = 0.25,
    gain_weight: float = 0.50,
    policy: str = "kt_value",
    reward_mode: str = "correctness",
    difficulty_by_item: Optional[Dict[Any, float]] = None,
    concept_by_item: Optional[Dict[Any, Any]] = None,
    progression_config: Optional[ProgressionRewardConfig] = None,
    delayed_gain_priors: Optional[Dict[str, Any]] = None,
    delayed_gain_reward_model: Optional[Any] = None,
    support_by_item: Optional[Dict[Any, float]] = None,
    support_by_concept: Optional[Dict[Any, float]] = None,
    logging_propensity_col: Optional[str] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Build OPE rows for a KT-guided next-item policy.

    If ``logging_propensity_col`` is supplied, its held-out values are used as
    the logging probability of the observed action. Otherwise public KT logs are
    treated with a documented synthetic-uniform assumption: the logged action is
    assumed to have been drawn uniformly from the candidate set.
    """
    if not tracer.is_fitted:
        raise RuntimeError("tracer must be fitted before policy OPE replay")
    if candidate_size < 1:
        raise ValueError("candidate_size must be >= 1")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be >= 1 when provided")
    if not 0.0 <= target_correct <= 1.0:
        raise ValueError("target_correct must be in [0, 1]")
    if policy not in {"kt_value", "progression", "delayed_gain", "support_delayed_gain"}:
        raise ValueError("policy must be 'kt_value', 'progression', 'delayed_gain', or 'support_delayed_gain'")
    if reward_mode not in {"correctness", "progression"}:
        raise ValueError("reward_mode must be 'correctness' or 'progression'")
    if logging_propensity_col is not None and logging_propensity_col not in split.test.columns:
        raise ValueError(f"logging_propensity_col={logging_propensity_col!r} not present in test split")

    rng = np.random.default_rng(random_state)
    known_items = list(candidate_item_ids) if candidate_item_ids is not None else split.train[split.item_col].drop_duplicates().tolist()
    known_items = sorted(set(known_items), key=lambda value: str(value))
    if not known_items:
        raise ValueError("candidate_item_ids is empty")
    known_set = set(known_items)

    ranker = _make_policy(
        tracer,
        policy=policy,
        target_correct=target_correct,
        stretch_weight=stretch_weight,
        uncertainty_weight=uncertainty_weight,
        gain_weight=gain_weight,
        difficulty_by_item=difficulty_by_item or {},
        concept_by_item=concept_by_item or {},
        progression_config=progression_config,
        delayed_gain_priors=delayed_gain_priors,
        delayed_gain_reward_model=delayed_gain_reward_model,
        support_by_item=support_by_item or {},
        support_by_concept=support_by_concept or {},
    )
    rows: list[dict[str, Any]] = []
    test = _ordered(split.test, user_col=split.user_col, timestamp_col=split.timestamp_col)
    if max_events is not None:
        test = test.head(int(max_events)).copy()

    for event_idx, row in enumerate(test.itertuples(index=False), start=1):
        row_data = row._asdict()
        user_id = row_data[split.user_col]
        logged_item = row_data[split.item_col]
        if logged_item not in known_set:
            continue

        label = int(_binary_labels([row_data[split.correct_col]], threshold=threshold)[0])
        candidates = _candidate_pool(
            logged_item,
            known_items=known_items,
            candidate_size=candidate_size,
            rng=rng,
        )
        ranked = ranker.rank(user_id, candidates, top_k=len(candidates))
        if not ranked:
            continue
        by_item = {rec.item_id: rec for rec in ranked}
        target = ranked[0]
        logged_rec = by_item[logged_item]
        target_probability = float(target.item_id == logged_item)
        random_probability = 1.0 / float(len(candidates))
        logging_propensity = (
            float(row_data[logging_propensity_col])
            if logging_propensity_col is not None
            else random_probability
        )
        if not 0.0 < logging_propensity <= 1.0:
            raise ValueError("logging propensity values must be in (0, 1]")
        reward, target_value, random_value, logged_action_value = _policy_values(
            label=label,
            target=target,
            logged_rec=logged_rec,
            ranked=ranked,
            reward_mode=reward_mode,
            progression_config=progression_config,
        )

        rows.append(
            {
                "event_id": event_idx,
                "user_id": user_id,
                "logged_item_id": logged_item,
                "target_item_id": target.item_id,
                "reward": reward,
                "correct": float(label),
                "candidate_size": float(len(candidates)),
                "logging_propensity": logging_propensity,
                "target_probability": target_probability,
                "random_probability": random_probability,
                "target_value": target_value,
                "random_value": random_value,
                "logged_action_value": logged_action_value,
                "target_score": float(target.score),
            }
        )
        ranker.observe(user_id, logged_item, label)

    if not rows:
        raise ValueError("policy OPE replay produced no events")
    return pd.DataFrame(rows)


def run_kt_policy_ope_benchmark(
    interactions: pd.DataFrame,
    *,
    model: str = "akt",
    user_col: str = "user_id",
    item_col: str = "item_id",
    correct_col: str = "correct",
    timestamp_col: Optional[str] = None,
    item_difficulty_col: Optional[str] = None,
    test_fraction: float = 0.2,
    candidate_size: int = 20,
    max_events: Optional[int] = None,
    max_weight: Optional[float] = None,
    logging_propensity_col: Optional[str] = None,
    policy: str = "kt_value",
    reward_mode: str = "correctness",
    concept_col: Optional[str] = None,
    delayed_gain_window: int = 5,
    max_seq_len: int = 50,
    d_model: int = 64,
    n_heads: int = 4,
    epochs: int = 5,
    batch_size: int = 128,
    random_state: Optional[int] = 42,
    device: Optional[str] = None,
    target_correct: float = 0.70,
    reward_model_max_examples: int = 50000,
    reward_model_example_weighting: str = "uniform",
    reward_model_cross_fit_folds: int = 1,
    reward_model_max_sample_weight: float = 20.0,
) -> Dict[str, Any]:
    """Fit a KT tracer and evaluate a next-item policy with logged-policy OPE."""
    if reward_mode not in {"correctness", "progression", "delayed_gain"}:
        raise ValueError("reward_mode must be 'correctness', 'progression', or 'delayed_gain'")
    if delayed_gain_window < 1:
        raise ValueError("delayed_gain_window must be >= 1")
    if reward_mode == "delayed_gain" and concept_col is None:
        raise ValueError("reward_mode='delayed_gain' requires concept_col")
    if policy in {"delayed_gain", "support_delayed_gain"} and concept_col is None:
        raise ValueError(f"policy={policy!r} requires concept_col")

    split = time_ordered_user_split(
        interactions,
        user_col=user_col,
        item_col=item_col,
        correct_col=correct_col,
        timestamp_col=timestamp_col,
        test_fraction=test_fraction,
    )
    tracer = _fit_tracer(
        split,
        model=model,
        user_col=user_col,
        item_col=item_col,
        correct_col=correct_col,
        timestamp_col=timestamp_col,
        item_difficulty_col=item_difficulty_col,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        device=device,
    )
    difficulty_by_item = None
    if item_difficulty_col is not None and item_difficulty_col in split.train.columns:
        difficulty_by_item = {
            item_id: float(value)
            for item_id, value in split.train.groupby(item_col)[item_difficulty_col].mean().items()
        }
    concept_by_item = None
    if concept_col is not None and concept_col in split.train.columns:
        concept_by_item = {
            item_id: value
            for item_id, value in split.train.groupby(item_col)[concept_col].agg(_mode_or_first).items()
        }
    delayed_gain_priors = None
    delayed_gain_reward_model = None
    support_by_item = None
    support_by_concept = None
    if policy in {"delayed_gain", "support_delayed_gain"}:
        delayed_gain_priors = estimate_delayed_gain_priors(
            split,
            concept_col=concept_col or "",
            future_window=delayed_gain_window,
        )
    progression_config = ProgressionRewardConfig(target_correct=target_correct)
    if policy == "support_delayed_gain":
        support_by_item, support_by_concept = _support_tables(split, concept_col=concept_col or "")
        delayed_gain_reward_model = fit_delayed_gain_reward_model(
            split,
            concept_col=concept_col or "",
            item_difficulty_col=item_difficulty_col,
            item_gain_prior=delayed_gain_priors["item_gain_prior"] if delayed_gain_priors else None,
            concept_gain_prior=delayed_gain_priors["concept_gain_prior"] if delayed_gain_priors else None,
            global_gain_prior=float(delayed_gain_priors["global_gain_prior"]) if delayed_gain_priors else 0.5,
            future_window=delayed_gain_window,
            max_examples=reward_model_max_examples,
            example_weighting=reward_model_example_weighting,
            max_sample_weight=reward_model_max_sample_weight,
            cross_fit_folds=reward_model_cross_fit_folds,
            random_state=random_state,
            config=progression_config,
            tracer=tracer,
        )
    event_reward_mode = "progression" if reward_mode == "delayed_gain" else reward_mode
    events = build_kt_policy_ope_events(
        tracer,
        split,
        candidate_size=candidate_size,
        max_events=max_events,
        random_state=random_state,
        target_correct=target_correct,
        policy=policy,
        reward_mode=event_reward_mode,
        difficulty_by_item=difficulty_by_item,
        concept_by_item=concept_by_item,
        progression_config=progression_config,
        delayed_gain_priors=delayed_gain_priors,
        delayed_gain_reward_model=delayed_gain_reward_model,
        support_by_item=support_by_item,
        support_by_concept=support_by_concept,
        logging_propensity_col=logging_propensity_col,
    )
    delayed_gain_info: Optional[Dict[str, Any]] = None
    if reward_mode == "delayed_gain":
        events = attach_delayed_gain_rewards(
            events,
            split,
            concept_col=concept_col or "",
            future_window=delayed_gain_window,
        )
        before_filter = len(events)
        events = events.dropna(subset=["delayed_gain_reward"]).copy()
        if events.empty:
            raise ValueError("delayed-gain OPE produced no events with future same-concept outcomes")
        delayed_gain_info = {
            "future_window": float(delayed_gain_window),
            "dropped_no_future_same_concept": float(before_filter - len(events)),
            "future_same_concept_count_mean": float(events["future_same_concept_count"].mean()),
        }

    comparison_kwargs: Dict[str, Any] = {}
    if reward_mode == "delayed_gain":
        reward_col = "delayed_gain_reward"
        if delayed_gain_reward_model is not None:
            comparison_kwargs.update(
                target_value_col="target_value",
                baseline_value_col="random_value",
                logged_action_value_col="logged_action_value",
            )
    else:
        reward_col = "reward"
        comparison_kwargs.update(
            target_value_col="target_value",
            baseline_value_col="random_value",
            logged_action_value_col="logged_action_value",
        )
    comparison = compare_logged_policies(
        events,
        reward_col=reward_col,
        propensity_col="logging_propensity",
        target_probability_col="target_probability",
        baseline_probability_col="random_probability",
        max_weight=max_weight,
        **comparison_kwargs,
    )
    has_direct_values = reward_mode != "delayed_gain" or delayed_gain_reward_model is not None
    target_value_mean = float(events["target_value"].mean()) if has_direct_values else None
    random_value_mean = float(events["random_value"].mean()) if has_direct_values else None
    report = KTPolicyOPEReport(
        n_events=int(len(events)),
        candidate_size_mean=float(events["candidate_size"].mean()),
        logging_reward=float(events[reward_col].mean()),
        target_match_rate=float(events["target_probability"].mean()),
        random_match_probability=float(events["random_probability"].mean()),
        target_value_mean=target_value_mean,
        random_value_mean=random_value_mean,
        comparison=comparison.to_dict(),
        split={
            "train_events": float(len(split.train)),
            "test_events": float(len(split.test)),
            "train_users": float(split.train[user_col].nunique()),
            "test_users": float(split.test[user_col].nunique()),
            "train_items": float(split.train[item_col].nunique()),
            "test_items": float(split.test[item_col].nunique()),
        },
        assumptions={
            "logging": (
                f"provided_propensity_col:{logging_propensity_col}"
                if logging_propensity_col is not None
                else "synthetic_uniform_over_candidate_set"
            ),
            "baseline_policy": "random_uniform_candidate",
            "reward": _reward_name(reward_mode, correct_col),
            "reward_mode": reward_mode,
            "policy": policy,
            "target_correct": float(target_correct),
            "candidate_size": float(candidate_size),
            "max_events": None if max_events is None else float(max_events),
            "reward_model_example_weighting": reward_model_example_weighting,
            "reward_model_cross_fit_folds": float(reward_model_cross_fit_folds),
        },
    )
    data = report.to_dict()
    if delayed_gain_priors is not None:
        data["delayed_gain_policy"] = {
            "global_gain_prior": delayed_gain_priors["global_gain_prior"],
            "item_priors": float(len(delayed_gain_priors["item_gain_prior"])),
            "concept_priors": float(len(delayed_gain_priors["concept_gain_prior"])),
            "shrinkage": delayed_gain_priors["shrinkage"],
        }
    if delayed_gain_reward_model is not None:
        data["delayed_gain_reward_model"] = delayed_gain_reward_model.to_dict()
    if delayed_gain_info is not None:
        data["delayed_gain"] = delayed_gain_info
    return data


def attach_delayed_gain_rewards(
    events: pd.DataFrame,
    split: KTHoldoutSplit,
    *,
    concept_col: str,
    future_window: int = 5,
    threshold: float = 0.5,
    reward_col: str = "delayed_gain_reward",
) -> pd.DataFrame:
    """Attach delayed same-concept gain rewards to policy OPE rows.

    The reward is a bounded proxy:
    ``clip(0.5 + 0.5 * (future_same_concept_correctness - train_prior), 0, 1)``.
    Rows without a future same-concept outcome keep a missing reward so callers
    can decide whether to filter or inspect coverage.
    """
    if concept_col not in split.train.columns or concept_col not in split.test.columns:
        raise ValueError(f"concept_col={concept_col!r} must exist in train and test splits")
    if future_window < 1:
        raise ValueError("future_window must be >= 1")
    if "event_id" not in events.columns:
        raise ValueError("events must include event_id from build_kt_policy_ope_events")

    rewards, future_counts = _delayed_gain_reward_maps(
        split,
        concept_col=concept_col,
        future_window=future_window,
        threshold=threshold,
    )
    out = events.copy()
    out[reward_col] = out["event_id"].map(rewards)
    out["future_same_concept_count"] = out["event_id"].map(future_counts).fillna(0).astype(int)
    return out


def estimate_delayed_gain_priors(
    split: KTHoldoutSplit,
    *,
    concept_col: str,
    future_window: int = 5,
    threshold: float = 0.5,
    shrinkage: float = 10.0,
) -> Dict[str, Any]:
    """Estimate training-only delayed-gain priors for a ranking policy.

    The returned priors use the same bounded delayed-gain proxy as evaluation
    but are computed only from the training split. Item priors are shrunk toward
    their concept prior, and concept priors are shrunk toward the global delayed
    gain prior to avoid overreacting to rare items.
    """
    if concept_col not in split.train.columns:
        raise ValueError(f"concept_col={concept_col!r} must exist in the train split")
    if future_window < 1:
        raise ValueError("future_window must be >= 1")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")

    train = _ordered(split.train, user_col=split.user_col, timestamp_col=split.timestamp_col).reset_index(drop=True)
    train["__orchid_label__"] = _binary_labels(train[split.correct_col].tolist(), threshold=threshold)
    global_label_prior = float(train["__orchid_label__"].mean())
    concept_label_prior = train.groupby(concept_col)["__orchid_label__"].mean().to_dict()
    user_concept_prior = train.groupby([split.user_col, concept_col])["__orchid_label__"].mean().to_dict()

    item_stats: Dict[Any, list[float]] = {}
    concept_stats: Dict[Any, list[float]] = {}
    item_concept: Dict[Any, Any] = {}
    all_rewards: list[float] = []

    for _user_id, group in train.groupby(split.user_col, sort=False):
        rows = group.to_dict("records")
        for pos, row in enumerate(rows):
            concept = row[concept_col]
            future = []
            for later in rows[pos + 1:]:
                if later[concept_col] == concept:
                    future.append(float(later["__orchid_label__"]))
                    if len(future) >= future_window:
                        break
            if not future:
                continue
            prior = user_concept_prior.get(
                (row[split.user_col], concept),
                concept_label_prior.get(concept, global_label_prior),
            )
            reward = float(np.clip(0.5 + 0.5 * (float(np.mean(future)) - float(prior)), 0.0, 1.0))
            item_id = row[split.item_col]
            item_concept[item_id] = concept
            item_stats.setdefault(item_id, [0.0, 0.0])
            item_stats[item_id][0] += reward
            item_stats[item_id][1] += 1.0
            concept_stats.setdefault(concept, [0.0, 0.0])
            concept_stats[concept][0] += reward
            concept_stats[concept][1] += 1.0
            all_rewards.append(reward)

    global_gain_prior = float(np.mean(all_rewards)) if all_rewards else 0.5
    concept_gain_prior = {
        concept: _shrunk_mean(total, count, prior=global_gain_prior, shrinkage=shrinkage)
        for concept, (total, count) in concept_stats.items()
    }
    item_gain_prior = {
        item_id: _shrunk_mean(
            total,
            count,
            prior=concept_gain_prior.get(item_concept.get(item_id), global_gain_prior),
            shrinkage=shrinkage,
        )
        for item_id, (total, count) in item_stats.items()
    }
    return {
        "global_gain_prior": global_gain_prior,
        "item_gain_prior": item_gain_prior,
        "concept_gain_prior": concept_gain_prior,
        "shrinkage": float(shrinkage),
    }


def run_kt_policy_ope_seed_sweep(
    interactions: pd.DataFrame,
    *,
    seeds: Sequence[int],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run KT policy OPE for multiple seeds and aggregate the policy uplift."""
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    runs = []
    for seed in seeds:
        report = run_kt_policy_ope_benchmark(
            interactions,
            random_state=int(seed),
            **kwargs,
        )
        runs.append({"seed": int(seed), "report": report})
    summary = _sweep_summary(runs)
    return KTPolicyOPESweepReport(summary=summary, runs=runs).to_dict()


def _make_policy(
    tracer: SAKTTracer,
    *,
    policy: str,
    target_correct: float,
    stretch_weight: float,
    uncertainty_weight: float,
    gain_weight: float,
    difficulty_by_item: Dict[Any, float],
    concept_by_item: Dict[Any, Any],
    progression_config: Optional[ProgressionRewardConfig],
    delayed_gain_priors: Optional[Dict[str, Any]],
    delayed_gain_reward_model: Optional[Any],
    support_by_item: Dict[Any, float],
    support_by_concept: Dict[Any, float],
) -> Any:
    if policy == "support_delayed_gain":
        priors = delayed_gain_priors or {}
        return SupportConstrainedDelayedGainPolicy(
            tracer,
            reward_model=delayed_gain_reward_model,
            difficulty_by_item=difficulty_by_item,
            concept_by_item=concept_by_item,
            item_gain_prior=priors.get("item_gain_prior", {}),
            concept_gain_prior=priors.get("concept_gain_prior", {}),
            global_gain_prior=float(priors.get("global_gain_prior", 0.5)),
            item_support=support_by_item,
            concept_support=support_by_concept,
            config=progression_config or ProgressionRewardConfig(target_correct=target_correct),
        )
    if policy == "delayed_gain":
        priors = delayed_gain_priors or {}
        return DelayedGainValuePolicy(
            tracer,
            difficulty_by_item=difficulty_by_item,
            concept_by_item=concept_by_item,
            item_gain_prior=priors.get("item_gain_prior", {}),
            concept_gain_prior=priors.get("concept_gain_prior", {}),
            global_gain_prior=float(priors.get("global_gain_prior", 0.5)),
            config=progression_config or ProgressionRewardConfig(target_correct=target_correct),
        )
    if policy == "progression":
        return ProgressionValuePolicy(
            tracer,
            difficulty_by_item=difficulty_by_item,
            concept_by_item=concept_by_item,
            config=progression_config or ProgressionRewardConfig(target_correct=target_correct),
        )
    return KTValuePolicy(
        tracer,
        target_correct=target_correct,
        stretch_weight=stretch_weight,
        uncertainty_weight=uncertainty_weight,
        gain_weight=gain_weight,
        difficulty_by_item=difficulty_by_item,
    )


def _reward_name(reward_mode: str, correct_col: str) -> str:
    if reward_mode == "progression":
        return "observed_progression_reward"
    if reward_mode == "delayed_gain":
        return "delayed_same_concept_gain_proxy"
    return correct_col


def _delayed_gain_reward_maps(
    split: KTHoldoutSplit,
    *,
    concept_col: str,
    future_window: int,
    threshold: float,
) -> tuple[Dict[int, float], Dict[int, int]]:
    train = split.train.copy()
    train["__orchid_label__"] = _binary_labels(train[split.correct_col].tolist(), threshold=threshold)
    global_prior = float(train["__orchid_label__"].mean())
    concept_prior = train.groupby(concept_col)["__orchid_label__"].mean().to_dict()
    user_concept_prior = train.groupby([split.user_col, concept_col])["__orchid_label__"].mean().to_dict()

    test = _ordered(split.test, user_col=split.user_col, timestamp_col=split.timestamp_col).reset_index(drop=True)
    test["__orchid_label__"] = _binary_labels(test[split.correct_col].tolist(), threshold=threshold)
    test["__orchid_event_id__"] = np.arange(1, len(test) + 1)

    rewards: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for _user_id, group in test.groupby(split.user_col, sort=False):
        rows = group.to_dict("records")
        future_by_pos = _future_same_concept(
            rows,
            concept_col=concept_col,
            label_col="__orchid_label__",
            future_window=future_window,
        )
        for pos, row in enumerate(rows):
            future = future_by_pos.get(pos)
            if future is None:
                continue
            concept = row[concept_col]
            prior = user_concept_prior.get(
                (row[split.user_col], concept),
                concept_prior.get(concept, global_prior),
            )
            future_mean, future_count = future
            gain = future_mean - float(prior)
            event_id = int(row["__orchid_event_id__"])
            rewards[event_id] = float(np.clip(0.5 + 0.5 * gain, 0.0, 1.0))
            counts[event_id] = future_count
    return rewards, counts


def _policy_values(
    *,
    label: int,
    target: Any,
    logged_rec: Any,
    ranked: Sequence[Any],
    reward_mode: str,
    progression_config: Optional[ProgressionRewardConfig],
) -> tuple[float, float, float, float]:
    if reward_mode == "progression":
        if not hasattr(logged_rec, "expected_reward"):
            raise ValueError("reward_mode='progression' requires policy='progression'")
        reward = observed_progression_reward(
            correct=label,
            p_correct=logged_rec.p_correct,
            difficulty=logged_rec.difficulty,
            competence=logged_rec.competence,
            recent_repetition=logged_rec.recent_repetition,
            config=progression_config,
        )
        return (
            float(reward),
            float(target.expected_reward),
            float(np.mean([rec.expected_reward for rec in ranked])),
            float(logged_rec.expected_reward),
        )
    return (
        float(label),
        float(target.p_correct),
        float(np.mean([rec.p_correct for rec in ranked])),
        float(logged_rec.p_correct),
    )


def _candidate_pool(
    logged_item: Any,
    *,
    known_items: Sequence[Any],
    candidate_size: int,
    rng: np.random.Generator,
) -> list[Any]:
    pool = [item for item in known_items if item != logged_item]
    sample_size = min(max(0, int(candidate_size) - 1), len(pool))
    if sample_size:
        sampled = rng.choice(np.asarray(pool, dtype=object), size=sample_size, replace=False).tolist()
    else:
        sampled = []
    candidates = [logged_item, *sampled]
    return sorted(candidates, key=lambda value: str(value))


def _fit_tracer(
    split: KTHoldoutSplit,
    *,
    model: str,
    user_col: str,
    item_col: str,
    correct_col: str,
    timestamp_col: Optional[str],
    item_difficulty_col: Optional[str],
    max_seq_len: int,
    d_model: int,
    n_heads: int,
    epochs: int,
    batch_size: int,
    random_state: Optional[int],
    device: Optional[str],
) -> SAKTTracer:
    normalized = model.lower().replace("_", "-")
    if normalized == "sakt":
        return SAKTTracer(
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
        ).fit(
            split.train,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
        )
    if normalized in {"akt", "akt-inspired"}:
        return AKTTracer(
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
        ).fit(
            split.train,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
            item_difficulty_col=item_difficulty_col,
        )
    raise ValueError("model must be 'sakt' or 'akt'")


def _sweep_summary(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    uplifts = np.asarray([run["report"]["comparison"]["uplift"] for run in runs], dtype=float)
    target_values = np.asarray([run["report"]["comparison"]["target"]["value"] for run in runs], dtype=float)
    baseline_values = np.asarray([run["report"]["comparison"]["baseline"]["value"] for run in runs], dtype=float)
    target_ess = np.asarray([run["report"]["comparison"]["target"]["effective_sample_size"] for run in runs], dtype=float)
    target_match = np.asarray([run["report"]["target_match_rate"] for run in runs], dtype=float)
    n_events = np.asarray([run["report"]["n_events"] for run in runs], dtype=float)
    ci_low, ci_high = _mean_ci(uplifts)
    return {
        "n_runs": float(len(runs)),
        "seeds": [int(run["seed"]) for run in runs],
        "n_events_mean": float(np.mean(n_events)),
        "uplift_mean": float(np.mean(uplifts)),
        "uplift_std": float(np.std(uplifts, ddof=1)) if len(uplifts) > 1 else 0.0,
        "uplift_ci_low": ci_low,
        "uplift_ci_high": ci_high,
        "target_value_mean": float(np.mean(target_values)),
        "baseline_value_mean": float(np.mean(baseline_values)),
        "target_ess_mean": float(np.mean(target_ess)),
        "target_match_rate_mean": float(np.mean(target_match)),
    }


def _mode_or_first(values: pd.Series) -> Any:
    modes = values.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return values.iloc[0]


def _future_same_concept(
    records: list[dict[str, Any]],
    *,
    concept_col: str,
    label_col: str,
    future_window: int,
) -> Dict[int, tuple[float, int]]:
    by_concept: Dict[Any, list[int]] = {}
    for pos, row in enumerate(records):
        by_concept.setdefault(row[concept_col], []).append(pos)
    future_by_pos: Dict[int, tuple[float, int]] = {}
    for positions in by_concept.values():
        labels = [float(records[pos][label_col]) for pos in positions]
        for idx, pos in enumerate(positions[:-1]):
            future = labels[idx + 1: idx + 1 + future_window]
            if future:
                future_by_pos[pos] = (float(np.mean(future)), len(future))
    return future_by_pos


def _support_tables(split: KTHoldoutSplit, *, concept_col: str) -> tuple[Dict[Any, float], Dict[Any, float]]:
    item_counts = {item_id: float(value) for item_id, value in split.train.groupby(split.item_col).size().items()}
    concept_counts = {concept: float(value) for concept, value in split.train.groupby(concept_col).size().items()}
    return item_counts, concept_counts


def _shrunk_mean(total: float, count: float, *, prior: float, shrinkage: float) -> float:
    if count <= 0:
        return float(prior)
    return float((float(total) + float(shrinkage) * float(prior)) / (float(count) + float(shrinkage)))


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean
    se = float(np.std(values, ddof=1) / np.sqrt(values.size))
    # Normal 95% interval is adequate for small benchmark summaries; individual
    # run intervals remain the primary uncertainty signal.
    return float(mean - 1.959963984540054 * se), float(mean + 1.959963984540054 * se)


def _ordered(
    frame: pd.DataFrame,
    *,
    user_col: str,
    timestamp_col: Optional[str],
) -> pd.DataFrame:
    work = frame.copy()
    work["__orchid_order__"] = np.arange(len(work))
    sort_cols = [user_col]
    if timestamp_col is not None:
        sort_cols.append(timestamp_col)
    sort_cols.append("__orchid_order__")
    return work.sort_values(sort_cols, kind="mergesort").drop(columns=["__orchid_order__"])
