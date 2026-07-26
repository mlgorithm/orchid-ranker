"""Logged-policy benchmarks for KT-guided adaptive recommendation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .adaptive_schema import normalize_timestamps, parse_candidate_list
from .delayed_gain import fit_delayed_gain_reward_model
from .kt import AKTTracer, SAKTTracer
from .kt_benchmark import (
    KTHoldoutSplit,
    _binary_labels,
    derive_train_only_item_difficulty,
    time_ordered_user_split,
)
from .learning_policy import (
    DelayedGainValuePolicy,
    HybridAdaptivePolicy,
    KTValuePolicy,
    ProgressionValuePolicy,
    SupportConstrainedDelayedGainPolicy,
)
from .ope import bootstrap_compare_logged_policies, compare_logged_policies
from .progression_reward import ProgressionRewardConfig, observed_progression_reward

__all__ = [
    "KTPolicyOPEReport",
    "KTPolicyOPESweepReport",
    "attach_delayed_gain_rewards",
    "attach_mastery_transition_rewards",
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
    split: Dict[str, Any]
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
    candidate_context_cols: Optional[Sequence[str]] = None,
    candidate_set_col: Optional[str] = None,
    logging_propensity_col: Optional[str] = None,
    target_exploration: float = 0.0,
    hybrid_progression_weight: float = 0.45,
    hybrid_item_prior_weight: float = 0.25,
    hybrid_concept_prior_weight: float = 0.10,
    hybrid_kt_weight: float = 0.15,
    hybrid_support_weight: float = 0.05,
    hybrid_unsupported_penalty_weight: float = 0.05,
    hybrid_min_item_support: float = 20.0,
    hybrid_min_concept_support: float = 100.0,
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
    if policy not in {"kt_value", "hybrid", "progression", "delayed_gain", "support_delayed_gain"}:
        raise ValueError("policy must be 'kt_value', 'hybrid', 'progression', 'delayed_gain', or 'support_delayed_gain'")
    if reward_mode not in {"correctness", "progression"}:
        raise ValueError("reward_mode must be 'correctness' or 'progression'")
    if not 0.0 <= target_exploration <= 1.0:
        raise ValueError("target_exploration must be in [0, 1]")
    if logging_propensity_col is not None and logging_propensity_col not in split.test.columns:
        raise ValueError(f"logging_propensity_col={logging_propensity_col!r} not present in test split")
    if candidate_set_col is not None and candidate_set_col not in split.test.columns:
        raise ValueError(f"candidate_set_col={candidate_set_col!r} not present in test split")
    context_cols = [col for col in (candidate_context_cols or []) if col]
    missing_context = [col for col in context_cols if col not in split.train.columns or col not in split.test.columns]
    if missing_context:
        raise ValueError(f"candidate_context_cols missing from train/test split: {missing_context}")

    rng = np.random.default_rng(random_state)
    known_items = list(candidate_item_ids) if candidate_item_ids is not None else split.train[split.item_col].drop_duplicates().tolist()
    known_items = sorted(set(known_items), key=lambda value: str(value))
    if not known_items:
        raise ValueError("candidate_item_ids is empty")
    known_set = set(known_items)
    item_correct, concept_correct, global_correct = _correctness_tables(
        split,
        concept_by_item=concept_by_item or {},
        threshold=threshold,
    )
    if policy == "hybrid":
        if support_by_item is None:
            support_by_item = {item: float(value) for item, value in split.train.groupby(split.item_col).size().items()}
        if support_by_concept is None and concept_by_item:
            concept_work = split.train[[split.item_col]].copy()
            concept_work["__orchid_concept__"] = concept_work[split.item_col].map(concept_by_item)
            support_by_concept = {
                concept: float(value)
                for concept, value in concept_work.groupby("__orchid_concept__", dropna=False).size().items()
            }

    context_pools = _context_candidate_pools(split, context_cols=context_cols, known_set=known_set)
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
        item_correct=item_correct,
        concept_correct=concept_correct,
        global_correct=global_correct,
        hybrid_progression_weight=hybrid_progression_weight,
        hybrid_item_prior_weight=hybrid_item_prior_weight,
        hybrid_concept_prior_weight=hybrid_concept_prior_weight,
        hybrid_kt_weight=hybrid_kt_weight,
        hybrid_support_weight=hybrid_support_weight,
        hybrid_unsupported_penalty_weight=hybrid_unsupported_penalty_weight,
        hybrid_min_item_support=hybrid_min_item_support,
        hybrid_min_concept_support=hybrid_min_concept_support,
    )
    if hasattr(ranker, "seed_history"):
        ranker.seed_history(
            split.train,
            user_col=split.user_col,
            item_col=split.item_col,
            correct_col=split.correct_col,
            timestamp_col=split.timestamp_col,
            reset=True,
        )
    rows: list[dict[str, Any]] = []
    test = _sample_replay_events(
        split.test,
        user_col=split.user_col,
        timestamp_col=split.timestamp_col,
        max_events=max_events,
        rng=rng,
    )

    for row_data in test.to_dict("records"):
        event_idx = int(row_data["__orchid_event_id__"])
        user_id = row_data[split.user_col]
        logged_item = row_data[split.item_col]
        if logged_item not in known_set:
            continue

        label = int(_binary_labels([row_data[split.correct_col]], threshold=threshold)[0])
        if candidate_set_col is not None:
            candidates = _logged_candidate_pool(
                row_data[candidate_set_col],
                logged_item=logged_item,
                known_set=known_set,
            )
        else:
            candidates = _candidate_pool(
                logged_item,
                known_items=known_items,
                candidate_size=candidate_size,
                rng=rng,
                row=row_data,
                context_cols=context_cols,
                context_pools=context_pools,
            )
        ranked = ranker.rank(user_id, candidates, top_k=len(candidates))
        if not ranked:
            continue
        by_item = {rec.item_id: rec for rec in ranked}
        target = ranked[0]
        logged_rec = by_item[logged_item]
        random_probability = 1.0 / float(len(candidates))
        target_probability = (
            (1.0 - float(target_exploration)) * float(target.item_id == logged_item)
            + float(target_exploration) * random_probability
        )
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
        target_value = (
            (1.0 - float(target_exploration)) * float(target_value)
            + float(target_exploration) * float(random_value)
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
                "logged_competence": (
                    None
                    if not hasattr(logged_rec, "competence")
                    else float(logged_rec.competence)
                ),
                "logged_recent_repetition": (
                    None
                    if not hasattr(logged_rec, "recent_repetition")
                    else float(logged_rec.recent_repetition)
                ),
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
    derive_item_difficulty: bool = False,
    test_fraction: float = 0.2,
    candidate_size: int = 20,
    max_events: Optional[int] = None,
    max_weight: Optional[float] = None,
    logging_propensity_col: Optional[str] = None,
    candidate_set_col: Optional[str] = None,
    policy: str = "kt_value",
    reward_mode: str = "correctness",
    concept_col: Optional[str] = None,
    candidate_context_cols: Optional[Sequence[str]] = None,
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
    kt_item_mean_blend: float = 0.0,
    target_exploration: float = 0.0,
    hybrid_progression_weight: float = 0.45,
    hybrid_item_prior_weight: float = 0.25,
    hybrid_concept_prior_weight: float = 0.10,
    hybrid_kt_weight: float = 0.15,
    hybrid_support_weight: float = 0.05,
    hybrid_unsupported_penalty_weight: float = 0.05,
    hybrid_min_item_support: float = 20.0,
    hybrid_min_concept_support: float = 100.0,
    cluster_bootstrap_samples: int = 0,
) -> Dict[str, Any]:
    """Fit a KT tracer and evaluate a next-item policy with logged-policy OPE."""
    if reward_mode not in {"correctness", "progression", "delayed_gain", "mastery_transition"}:
        raise ValueError("reward_mode must be 'correctness', 'progression', 'delayed_gain', or 'mastery_transition'")
    if delayed_gain_window < 1:
        raise ValueError("delayed_gain_window must be >= 1")
    if reward_mode in {"delayed_gain", "mastery_transition"} and concept_col is None:
        raise ValueError(f"reward_mode={reward_mode!r} requires concept_col")
    if policy in {"delayed_gain", "support_delayed_gain"} and concept_col is None:
        raise ValueError(f"policy={policy!r} requires concept_col")
    if not 0.0 <= kt_item_mean_blend <= 1.0:
        raise ValueError("kt_item_mean_blend must be in [0, 1]")
    if not 0.0 <= target_exploration <= 1.0:
        raise ValueError("target_exploration must be in [0, 1]")
    if cluster_bootstrap_samples < 0:
        raise ValueError("cluster_bootstrap_samples must be non-negative")
    if (logging_propensity_col is None) != (candidate_set_col is None):
        raise ValueError(
            "decision-grade OPE requires logging_propensity_col and candidate_set_col together"
        )

    split = time_ordered_user_split(
        interactions,
        user_col=user_col,
        item_col=item_col,
        correct_col=correct_col,
        timestamp_col=timestamp_col,
        test_fraction=test_fraction,
    )
    difficulty_source = "external_metadata" if item_difficulty_col is not None else "none"
    if derive_item_difficulty:
        item_difficulty_col = item_difficulty_col or "__orchid_train_difficulty__"
        split = derive_train_only_item_difficulty(split, output_col=item_difficulty_col)
        difficulty_source = "train_labels_only"
    tracer: Any = _fit_tracer(
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
    if kt_item_mean_blend > 0.0:
        tracer = _ItemMeanBlendedTracer(
            tracer,
            split.train,
            item_col=item_col,
            correct_col=correct_col,
            threshold=0.5,
            blend=kt_item_mean_blend,
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
    event_reward_mode = "progression" if reward_mode in {"delayed_gain", "mastery_transition"} else reward_mode
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
        candidate_context_cols=candidate_context_cols,
        candidate_set_col=candidate_set_col,
        logging_propensity_col=logging_propensity_col,
        target_exploration=target_exploration,
        hybrid_progression_weight=hybrid_progression_weight,
        hybrid_item_prior_weight=hybrid_item_prior_weight,
        hybrid_concept_prior_weight=hybrid_concept_prior_weight,
        hybrid_kt_weight=hybrid_kt_weight,
        hybrid_support_weight=hybrid_support_weight,
        hybrid_unsupported_penalty_weight=hybrid_unsupported_penalty_weight,
        hybrid_min_item_support=hybrid_min_item_support,
        hybrid_min_concept_support=hybrid_min_concept_support,
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
    if reward_mode == "mastery_transition":
        events = attach_mastery_transition_rewards(
            events,
            split,
            concept_col=concept_col or "",
            future_window=delayed_gain_window,
        )
        before_filter = len(events)
        events = events.dropna(subset=["mastery_transition_reward"]).copy()
        if events.empty:
            raise ValueError("mastery-transition OPE produced no below-mastery events with future same-concept outcomes")
        delayed_gain_info = {
            "future_window": float(delayed_gain_window),
            "dropped_no_future_same_concept_or_already_mastered": float(before_filter - len(events)),
            "future_same_concept_count_mean": float(events["future_same_concept_count"].mean()),
            "prior_competence_mean": float(events["prior_competence"].mean()),
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
    elif reward_mode == "mastery_transition":
        reward_col = "mastery_transition_reward"
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
    cluster_bootstrap = None
    if cluster_bootstrap_samples > 0:
        cluster_bootstrap = bootstrap_compare_logged_policies(
            events,
            reward_col=reward_col,
            propensity_col="logging_propensity",
            target_probability_col="target_probability",
            baseline_probability_col="random_probability",
            max_weight=max_weight,
            n_bootstrap=cluster_bootstrap_samples,
            random_state=random_state,
            cluster_col="user_id",
            **comparison_kwargs,
        ).to_dict()
    has_direct_values = reward_mode not in {"delayed_gain", "mastery_transition"} or delayed_gain_reward_model is not None
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
            "replay_users": float(events["user_id"].nunique()),
            "replay_events": float(len(events)),
            "item_difficulty_source": difficulty_source,
        },
        assumptions={
            "logging": (
                f"provided_propensity_col:{logging_propensity_col}"
                if logging_propensity_col is not None
                else "synthetic_uniform_over_candidate_set"
            ),
            "logging_support": (
                "logged_candidate_set"
                if logging_propensity_col is not None and candidate_set_col is not None
                else "synthetic_candidate_set"
            ),
            "baseline_policy": "random_uniform_candidate",
            "reward": _reward_name(reward_mode, correct_col),
            "reward_mode": reward_mode,
            "policy": policy,
            "target_correct": float(target_correct),
            "candidate_size": float(candidate_size),
            "candidate_context_cols": list(candidate_context_cols or []),
            "candidate_set_col": candidate_set_col,
            "max_events": None if max_events is None else float(max_events),
            "replay_sampling": "random_learners_chronological_within_learner",
            "cluster_bootstrap_samples": float(cluster_bootstrap_samples),
            "reward_model_example_weighting": reward_model_example_weighting,
            "reward_model_cross_fit_folds": float(reward_model_cross_fit_folds),
            "kt_item_mean_blend": float(kt_item_mean_blend),
            "target_exploration": float(target_exploration),
            "hybrid_weights": {
                "progression": float(hybrid_progression_weight),
                "item_prior": float(hybrid_item_prior_weight),
                "concept_prior": float(hybrid_concept_prior_weight),
                "kt": float(hybrid_kt_weight),
                "support": float(hybrid_support_weight),
                "unsupported_penalty": float(hybrid_unsupported_penalty_weight),
                "min_item_support": float(hybrid_min_item_support),
                "min_concept_support": float(hybrid_min_concept_support),
            },
        },
    )
    data = report.to_dict()
    if cluster_bootstrap is not None:
        data["cluster_bootstrap_comparison"] = cluster_bootstrap
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
        data["delayed_gain" if reward_mode == "delayed_gain" else "mastery_transition"] = delayed_gain_info
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


def attach_mastery_transition_rewards(
    events: pd.DataFrame,
    split: KTHoldoutSplit,
    *,
    concept_col: str,
    future_window: int = 5,
    threshold: float = 0.5,
    mastery_threshold: float = 0.8,
    reward_col: str = "mastery_transition_reward",
) -> pd.DataFrame:
    """Attach binary below-mastery to future-mastery transition rewards.

    A row receives a reward only when the learner's train-split competence for
    the concept is below ``mastery_threshold`` and the held-out sequence has a
    future same-concept window. The reward is 1 when future same-concept
    correctness crosses ``mastery_threshold`` and 0 otherwise.
    """
    if concept_col not in split.train.columns or concept_col not in split.test.columns:
        raise ValueError(f"concept_col={concept_col!r} must exist in train and test splits")
    if future_window < 1:
        raise ValueError("future_window must be >= 1")
    if not 0.0 < mastery_threshold <= 1.0:
        raise ValueError("mastery_threshold must be in (0, 1]")
    if "event_id" not in events.columns:
        raise ValueError("events must include event_id from build_kt_policy_ope_events")

    rewards, future_counts, prior_competence = _mastery_transition_reward_maps(
        split,
        concept_col=concept_col,
        future_window=future_window,
        threshold=threshold,
        mastery_threshold=mastery_threshold,
    )
    out = events.copy()
    out[reward_col] = out["event_id"].map(rewards)
    out["future_same_concept_count"] = out["event_id"].map(future_counts).fillna(0).astype(int)
    out["prior_competence"] = out["event_id"].map(prior_competence)
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

    item_stats: Dict[Any, list[float]] = {}
    concept_stats: Dict[Any, list[float]] = {}
    item_concept: Dict[Any, Any] = {}
    all_rewards: list[float] = []

    for _user_id, group in train.groupby(split.user_col, sort=False):
        rows = group.to_dict("records")
        prior_totals: Dict[Any, float] = {}
        prior_counts: Dict[Any, int] = {}
        for pos, row in enumerate(rows):
            concept = row[concept_col]
            future = []
            for later in rows[pos + 1:]:
                if later[concept_col] == concept:
                    future.append(float(later["__orchid_label__"]))
                    if len(future) >= future_window:
                        break
            if not future:
                prior_totals[concept] = prior_totals.get(concept, 0.0) + float(row["__orchid_label__"])
                prior_counts[concept] = prior_counts.get(concept, 0) + 1
                continue
            if prior_counts.get(concept, 0) > 0:
                prior = prior_totals[concept] / float(prior_counts[concept])
            else:
                prior = concept_label_prior.get(concept, global_label_prior)
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
            prior_totals[concept] = prior_totals.get(concept, 0.0) + float(row["__orchid_label__"])
            prior_counts[concept] = prior_counts.get(concept, 0) + 1

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


class _ItemMeanBlendedTracer:
    """Blend neural KT probabilities with a smoothed item-correctness prior."""

    def __init__(
        self,
        tracer: Any,
        train: pd.DataFrame,
        *,
        item_col: str,
        correct_col: str,
        threshold: float,
        blend: float,
        smoothing: float = 5.0,
    ) -> None:
        self.tracer = tracer
        self.blend = float(blend)
        labels = _binary_labels(train[correct_col].tolist(), threshold=threshold)
        work = train[[item_col]].copy()
        work["__orchid_label__"] = labels
        self.global_mean = float(work["__orchid_label__"].mean())
        grouped = work.groupby(item_col)["__orchid_label__"].agg(["sum", "count"])
        self.item_totals: Dict[Any, float] = {item: float(row["sum"]) for item, row in grouped.iterrows()}
        self.item_counts: Dict[Any, float] = {item: float(row["count"]) for item, row in grouped.iterrows()}
        self.smoothing = float(smoothing)

    @property
    def is_fitted(self) -> bool:
        return bool(getattr(self.tracer, "is_fitted", False))

    @property
    def item_ids_(self) -> list[Any]:
        return list(getattr(self.tracer, "item_ids_", []))

    def predict_correct(self, user_id: Any, item_id: Any) -> float:
        return float(self.predict_many(user_id, [item_id])[item_id])

    def predict_many(self, user_id: Any, item_ids: Sequence[Any]) -> Dict[Any, float]:
        neural = self.tracer.predict_many(user_id, item_ids)
        out = {}
        for item_id in item_ids:
            prior = self._item_prior(item_id)
            out[item_id] = float((1.0 - self.blend) * float(neural[item_id]) + self.blend * prior)
        return out

    def observe(self, user_id: Any, item_id: Any, correct: Any, **kwargs: Any) -> Any:
        label = float(_binary_labels([correct])[0])
        self.item_totals[item_id] = self.item_totals.get(item_id, 0.0) + label
        self.item_counts[item_id] = self.item_counts.get(item_id, 0.0) + 1.0
        return self.tracer.observe(user_id, item_id, correct, **kwargs)

    def _item_prior(self, item_id: Any) -> float:
        total = self.item_totals.get(item_id, 0.0)
        count = self.item_counts.get(item_id, 0.0)
        return float((total + self.smoothing * self.global_mean) / (count + self.smoothing))


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
    item_correct: Dict[Any, float],
    concept_correct: Dict[Any, float],
    global_correct: float,
    hybrid_progression_weight: float,
    hybrid_item_prior_weight: float,
    hybrid_concept_prior_weight: float,
    hybrid_kt_weight: float,
    hybrid_support_weight: float,
    hybrid_unsupported_penalty_weight: float,
    hybrid_min_item_support: float,
    hybrid_min_concept_support: float,
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
    if policy == "hybrid":
        return HybridAdaptivePolicy(
            tracer,
            difficulty_by_item=difficulty_by_item,
            concept_by_item=concept_by_item,
            item_correct=item_correct,
            item_count=support_by_item,
            concept_correct=concept_correct,
            concept_count=support_by_concept,
            global_correct=global_correct,
            config=progression_config or ProgressionRewardConfig(target_correct=target_correct),
            progression_weight=hybrid_progression_weight,
            item_prior_weight=hybrid_item_prior_weight,
            concept_prior_weight=hybrid_concept_prior_weight,
            kt_weight=hybrid_kt_weight,
            support_weight=hybrid_support_weight,
            unsupported_penalty_weight=hybrid_unsupported_penalty_weight,
            min_item_support=hybrid_min_item_support,
            min_concept_support=hybrid_min_concept_support,
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
    if reward_mode == "mastery_transition":
        return "below_mastery_to_future_mastery"
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


def _mastery_transition_reward_maps(
    split: KTHoldoutSplit,
    *,
    concept_col: str,
    future_window: int,
    threshold: float,
    mastery_threshold: float,
) -> tuple[Dict[int, float], Dict[int, int], Dict[int, float]]:
    train = split.train.copy()
    train["__orchid_label__"] = _binary_labels(train[split.correct_col].tolist(), threshold=threshold)
    prior_totals = train.groupby([split.user_col, concept_col])["__orchid_label__"].sum().to_dict()
    prior_counts = train.groupby([split.user_col, concept_col])["__orchid_label__"].count().to_dict()
    concept_prior = train.groupby(concept_col)["__orchid_label__"].mean().to_dict()
    global_prior = float(train["__orchid_label__"].mean())

    test = _ordered(split.test, user_col=split.user_col, timestamp_col=split.timestamp_col).reset_index(drop=True)
    test["__orchid_label__"] = _binary_labels(test[split.correct_col].tolist(), threshold=threshold)
    test["__orchid_event_id__"] = np.arange(1, len(test) + 1)

    rewards: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    prior_competence: Dict[int, float] = {}
    running_totals: Dict[tuple[Any, Any], float] = {key: float(value) for key, value in prior_totals.items()}
    running_counts: Dict[tuple[Any, Any], float] = {key: float(value) for key, value in prior_counts.items()}

    for user_id, group in test.groupby(split.user_col, sort=False):
        rows = group.to_dict("records")
        future_by_pos = _future_same_concept(
            rows,
            concept_col=concept_col,
            label_col="__orchid_label__",
            future_window=future_window,
        )
        for pos, row in enumerate(rows):
            concept = row[concept_col]
            key = (user_id, concept)
            total = running_totals.get(key, 0.0)
            count = running_counts.get(key, 0.0)
            if count > 0:
                prior = float(total) / float(count)
            else:
                prior = float(concept_prior.get(concept, global_prior))

            future = future_by_pos.get(pos)
            event_id = int(row["__orchid_event_id__"])
            if future is not None and prior < float(mastery_threshold):
                future_mean, future_count = future
                rewards[event_id] = float(float(future_mean) >= float(mastery_threshold))
                counts[event_id] = int(future_count)
                prior_competence[event_id] = float(prior)

            running_totals[key] = running_totals.get(key, 0.0) + float(row["__orchid_label__"])
            running_counts[key] = running_counts.get(key, 0.0) + 1.0
    return rewards, counts, prior_competence


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


def _sample_replay_events(
    frame: pd.DataFrame,
    *,
    user_col: str,
    timestamp_col: Optional[str],
    max_events: Optional[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample learners uniformly, then retain chronological replay per learner."""
    ordered = _ordered(frame, user_col=user_col, timestamp_col=timestamp_col).reset_index(drop=True)
    ordered["__orchid_event_id__"] = np.arange(1, len(ordered) + 1)
    if max_events is None or len(ordered) <= int(max_events):
        return ordered

    users = np.asarray(ordered[user_col].drop_duplicates().tolist(), dtype=object)
    sampled_users = rng.permutation(users).tolist()
    by_user = {user_id: group for user_id, group in ordered.groupby(user_col, sort=False)}
    parts: list[pd.DataFrame] = []
    remaining = int(max_events)
    for user_id in sampled_users:
        if remaining <= 0:
            break
        group = by_user[user_id]
        take = min(remaining, len(group))
        parts.append(group.iloc[:take])
        remaining -= take
    return pd.concat(parts, ignore_index=True)


def _logged_candidate_pool(
    value: Any,
    *,
    logged_item: Any,
    known_set: set[Any],
) -> list[Any]:
    """Validate and return the exact candidate set recorded at decision time."""
    raw = parse_candidate_list(value)
    candidates: list[Any] = []
    seen: set[Any] = set()
    for item_id in raw:
        if item_id in seen:
            continue
        seen.add(item_id)
        candidates.append(item_id)
    if not candidates:
        raise ValueError("logged candidate set must be non-empty")
    if logged_item not in seen:
        raise ValueError("logged action must appear in the logged candidate set")
    unknown = [item_id for item_id in candidates if item_id not in known_set]
    if unknown:
        raise ValueError(
            "logged candidate set contains items unseen during KT training: "
            f"{unknown[:5]}"
        )
    return candidates


def _candidate_pool(
    logged_item: Any,
    *,
    known_items: Sequence[Any],
    candidate_size: int,
    rng: np.random.Generator,
    row: Optional[Dict[str, Any]] = None,
    context_cols: Optional[Sequence[str]] = None,
    context_pools: Optional[Dict[str, Dict[Any, list[Any]]]] = None,
) -> list[Any]:
    selected: list[Any] = []
    seen = {logged_item}
    row = row or {}
    context_pools = context_pools or {}
    for col in context_cols or ():
        value = row.get(col)
        if pd.isna(value):
            continue
        pool = [item for item in context_pools.get(col, {}).get(value, []) if item not in seen]
        need = max(0, int(candidate_size) - 1 - len(selected))
        if need <= 0:
            break
        if len(pool) > need:
            pool = rng.choice(np.asarray(pool, dtype=object), size=need, replace=False).tolist()
        selected.extend(pool)
        seen.update(pool)

    pool = [item for item in known_items if item not in seen]
    sample_size = min(max(0, int(candidate_size) - 1 - len(selected)), len(pool))
    if sample_size:
        selected.extend(rng.choice(np.asarray(pool, dtype=object), size=sample_size, replace=False).tolist())
    candidates = [logged_item, *selected]
    return sorted(candidates, key=lambda value: str(value))


def _context_candidate_pools(
    split: KTHoldoutSplit,
    *,
    context_cols: Sequence[str],
    known_set: set[Any],
) -> Dict[str, Dict[Any, list[Any]]]:
    pools: Dict[str, Dict[Any, list[Any]]] = {}
    if not context_cols:
        return pools
    train = split.train
    for col in context_cols:
        by_value: Dict[Any, list[Any]] = {}
        for value, group in train.groupby(col, sort=False, dropna=False):
            if pd.isna(value):
                continue
            items = sorted(
                {item for item in group[split.item_col].tolist() if item in known_set},
                key=lambda item: str(item),
            )
            if items:
                by_value[value] = items
        pools[col] = by_value
    return pools


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
    targets = [run["report"]["comparison"]["target"] for run in runs]
    target_ess = np.asarray([target["effective_sample_size"] for target in targets], dtype=float)
    target_coverage = np.asarray([target["coverage"] for target in targets], dtype=float)
    target_clipped = np.asarray([target["clipped_fraction"] for target in targets], dtype=float)
    target_ess_fraction = np.asarray(
        [
            float(target["effective_sample_size"]) / max(float(target["n_events"]), 1.0)
            for target in targets
        ],
        dtype=float,
    )
    target_match = np.asarray([run["report"]["target_match_rate"] for run in runs], dtype=float)
    n_events = np.asarray([run["report"]["n_events"] for run in runs], dtype=float)
    replay_users = np.asarray([run["report"]["split"]["replay_users"] for run in runs], dtype=float)
    bootstrap_reports = [
        run["report"].get("cluster_bootstrap_comparison")
        for run in runs
    ]
    if all(report is not None for report in bootstrap_reports):
        ci_low = float(min(report["bootstrap_ci_low"] for report in bootstrap_reports if report is not None))
        ci_high = float(max(report["bootstrap_ci_high"] for report in bootstrap_reports if report is not None))
        ci_method = "learner_cluster_bootstrap_seed_envelope"
    else:
        ci_low, ci_high = _mean_ci(uplifts)
        ci_method = "seed_normal"
    logging_support = sorted(
        {str(run["report"]["assumptions"].get("logging_support", "unknown")) for run in runs}
    )
    return {
        "n_runs": float(len(runs)),
        "seeds": [int(run["seed"]) for run in runs],
        "n_events_mean": float(np.mean(n_events)),
        "uplift_mean": float(np.mean(uplifts)),
        "uplift_std": float(np.std(uplifts, ddof=1)) if len(uplifts) > 1 else 0.0,
        "uplift_ci_low": ci_low,
        "uplift_ci_high": ci_high,
        "uplift_ci_method": ci_method,
        "target_value_mean": float(np.mean(target_values)),
        "baseline_value_mean": float(np.mean(baseline_values)),
        "target_ess_mean": float(np.mean(target_ess)),
        "target_ess_fraction_mean": float(np.mean(target_ess_fraction)),
        "target_coverage_mean": float(np.mean(target_coverage)),
        "target_clipped_fraction_mean": float(np.mean(target_clipped)),
        "target_match_rate_mean": float(np.mean(target_match)),
        "replay_users_mean": float(np.mean(replay_users)),
        "replay_users_min": float(np.min(replay_users)),
        "logging_support": logging_support,
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
    concept_counts = {concept: float(value) for concept, value in split.train.groupby(concept_col, dropna=False).size().items()}
    return item_counts, concept_counts


def _correctness_tables(
    split: KTHoldoutSplit,
    *,
    concept_by_item: Dict[Any, Any],
    threshold: float,
) -> tuple[Dict[Any, float], Dict[Any, float], float]:
    labels = _binary_labels(split.train[split.correct_col].tolist(), threshold=threshold).astype(float)
    work = split.train[[split.item_col]].copy()
    work["__orchid_label__"] = labels
    item_correct = {
        item_id: float(value)
        for item_id, value in work.groupby(split.item_col)["__orchid_label__"].sum().items()
    }
    concept_correct: Dict[Any, float] = {}
    if concept_by_item:
        work["__orchid_concept__"] = work[split.item_col].map(concept_by_item)
        concept_correct = {
            concept: float(value)
            for concept, value in work.groupby("__orchid_concept__", dropna=False)["__orchid_label__"].sum().items()
        }
    global_correct = float(labels.mean()) if len(labels) else 0.5
    return item_correct, concept_correct, global_correct


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
    if timestamp_col is not None:
        work[timestamp_col] = normalize_timestamps(work[timestamp_col], timestamp_col)
    work["__orchid_order__"] = np.arange(len(work))
    sort_cols = [user_col]
    if timestamp_col is not None:
        sort_cols.append(timestamp_col)
    sort_cols.append("__orchid_order__")
    return work.sort_values(sort_cols, kind="mergesort").drop(columns=["__orchid_order__"])
