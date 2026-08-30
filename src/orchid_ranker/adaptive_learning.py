"""First-class adaptive-learning recommender API.

This module composes Orchid's strongest in-repo adaptive-learning pieces:
sequence-aware knowledge tracing, progression reward scoring, delayed-gain
priors, support-aware direct reward modeling, and prerequisite gating.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .adaptive_schema import normalize_timestamps
from .delayed_gain import DelayedGainRewardModel, fit_delayed_gain_reward_model
from .kt_benchmark import KTHoldoutSplit
from .learning_policy import (
    DelayedGainValuePolicy,
    HybridAdaptivePolicy,
    KTValuePolicy,
    ProgressionValuePolicy,
    SupportConstrainedDelayedGainPolicy,
)
from .policy_benchmark import estimate_delayed_gain_priors
from .progression_reward import ProgressionRewardConfig

__all__ = [
    "AdaptiveLearningConfig",
    "AdaptiveLearningReadinessReport",
    "AdaptiveLearningRecommendation",
    "AdaptiveLearningRecommender",
]


@dataclass(frozen=True)
class AdaptiveLearningConfig:
    """Configuration for :class:`AdaptiveLearningRecommender`."""

    tracer_model: str = "akt"
    policy: str = "auto"
    target_correct: float = 0.70
    max_seq_len: int = 50
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 128
    correct_threshold: float = 0.5
    delayed_gain_window: int = 5
    delayed_gain_shrinkage: float = 10.0
    reward_model_max_examples: Optional[int] = 50000
    reward_model_example_weighting: str = "support_inverse"
    reward_model_cross_fit_folds: int = 2
    reward_model_max_sample_weight: float = 20.0
    progression_competence_blend: float = 0.35
    hybrid_progression_weight: float = 0.45
    hybrid_item_prior_weight: float = 0.25
    hybrid_concept_prior_weight: float = 0.10
    hybrid_kt_weight: float = 0.15
    hybrid_support_weight: float = 0.05
    hybrid_unsupported_penalty_weight: float = 0.05
    hybrid_prior_smoothing: float = 8.0
    hybrid_concept_smoothing: float = 20.0
    hybrid_min_item_support: float = 20.0
    hybrid_min_concept_support: float = 100.0
    mastery_threshold: float = 0.80
    enforce_prerequisites: bool = True
    allow_prerequisite_fallback: bool = False
    fallback_to_empirical: bool = True
    min_kt_events: int = 500
    min_kt_users: int = 50
    min_kt_items: int = 10
    min_kt_median_events_per_user: float = 3.0
    device: Optional[str] = None
    random_state: Optional[int] = 42


@dataclass(frozen=True)
class AdaptiveLearningReadinessReport:
    """Whether observed learning data supports a sequence-model deployment.

    The report never blocks fitting. It allows a pilot to begin with a
    transparent empirical tracer and makes the conditions for graduating to
    knowledge tracing explicit.
    """

    n_events: int
    n_users: int
    n_items: int
    n_categories: int
    median_events_per_user: float
    median_events_per_item: float
    outcome_rate: float
    outcome_entropy: float
    has_categories: bool
    has_difficulty: bool
    knowledge_tracing_ready: bool
    active_tracer: str
    reasons: tuple[str, ...]
    recommendations: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveLearningRecommendation:
    """Normalized recommendation returned by :class:`AdaptiveRanker`."""

    item_id: Any
    score: float
    outcome_probability: float
    policy: str
    difficulty: Optional[float] = None
    category_id: Optional[Any] = None
    competence: Optional[float] = None
    expected_reward: Optional[float] = None
    stretch_fit: Optional[float] = None
    expected_gain: Optional[float] = None
    uncertainty: Optional[float] = None
    delayed_gain_prior: Optional[float] = None
    model_prediction: Optional[float] = None
    support_penalty: float = 0.0
    item_support: float = 0.0
    concept_support: float = 0.0
    recent_repetition: int = 0
    prerequisites_met: bool = True
    feedback_supported: bool = True
    reward_breakdown: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveLearningRecommender:
    """Adaptive-learning recommender with a production-oriented default stack.

    ``policy="auto"`` resolves to ``HybridAdaptivePolicy``: empirical priors,
    KT state, support confidence, and progression reward. Delayed-gain and
    support-constrained delayed-gain policies are available as explicit opt-ins
    because they require stronger reward-model and logged-support assumptions.

    The fitted object exposes ``rank``/``recommend`` for serving and ``observe``
    for live learner updates.
    """

    def __init__(self, config: Optional[AdaptiveLearningConfig] = None, **overrides: Any) -> None:
        valid = {field.name for field in fields(AdaptiveLearningConfig)}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(f"Unknown AdaptiveLearningConfig fields: {unknown}")
        self.config = replace(config or AdaptiveLearningConfig(), **overrides)
        self.tracer_: Any = None
        self.policy_: Any = None
        self._state_policy: Optional[ProgressionValuePolicy] = None
        self.policy_name_: Optional[str] = None
        self.progression_config_: Optional[ProgressionRewardConfig] = None
        self.delayed_gain_priors_: Optional[Dict[str, Any]] = None
        self.delayed_gain_reward_model_: Optional[DelayedGainRewardModel] = None
        self.difficulty_by_item_: Dict[Any, float] = {}
        self.concept_by_item_: Dict[Any, Any] = {}
        self.prerequisite_by_concept_: Dict[Any, set[Any]] = {}
        self.item_support_: Dict[Any, float] = {}
        self.concept_support_: Dict[Any, float] = {}
        self.item_correct_: Dict[Any, float] = {}
        self.concept_correct_: Dict[Any, float] = {}
        self.global_correct_: float = 0.5
        self.item_ids_: list[Any] = []
        self._item_id_set: set[Any] = set()
        self.user_ids_: list[Any] = []
        self._training_concept_col: Optional[str] = None
        self._item_col: str = "item_id"
        self._global_correct_total: float = 0.0
        self._global_outcome_count: float = 0.0
        self.readiness_report_: Optional[AdaptiveLearningReadinessReport] = None

    @property
    def is_fitted(self) -> bool:
        """Whether the recommender has been fitted."""
        return self.tracer_ is not None and self.policy_ is not None

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        correct_col: str = "correct",
        timestamp_col: Optional[str] = None,
        concept_col: Optional[str] = None,
        item_difficulty_col: Optional[str] = None,
        item_difficulty_map: Optional[Mapping[Any, float]] = None,
        concept_by_item: Optional[Mapping[Any, Any]] = None,
        prerequisite_by_concept: Optional[Mapping[Any, Sequence[Any]]] = None,
    ) -> "AdaptiveLearningRecommender":
        """Fit the adaptive-learning stack from learner outcome history."""
        if interactions.empty:
            raise ValueError("interactions DataFrame is empty")
        required = {user_col, item_col, correct_col}
        if timestamp_col is not None:
            required.add(timestamp_col)
        if concept_col is not None:
            required.add(concept_col)
        if item_difficulty_col is not None:
            required.add(item_difficulty_col)
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing required columns: {sorted(missing)}")

        _validate_config(self.config)
        work = _ordered(interactions, user_col=user_col, timestamp_col=timestamp_col).reset_index(drop=True)
        self._item_col = item_col
        self.item_ids_ = sorted(work[item_col].drop_duplicates().tolist(), key=lambda value: str(value))
        self._item_id_set = set(self.item_ids_)
        self.user_ids_ = sorted(work[user_col].drop_duplicates().tolist(), key=lambda value: str(value))
        self.difficulty_by_item_ = _difficulty_by_item(
            work,
            item_col=item_col,
            correct_col=correct_col,
            difficulty_col=item_difficulty_col,
            difficulty_map=item_difficulty_map,
            threshold=self.config.correct_threshold,
        )
        self.concept_by_item_, training_concept_col = _concept_by_item(
            work,
            item_col=item_col,
            concept_col=concept_col,
            concept_map=concept_by_item,
        )
        if training_concept_col == "__orchid_concept__" or training_concept_col not in work.columns:
            work[training_concept_col] = work[item_col].map(self.concept_by_item_)
        self._training_concept_col = training_concept_col
        self.prerequisite_by_concept_ = {
            concept: set(prerequisites)
            for concept, prerequisites in dict(prerequisite_by_concept or {}).items()
        }
        readiness = _assess_learning_readiness(
            work,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            category_col=concept_col,
            difficulty_col=item_difficulty_col,
            config=self.config,
        )
        active_tracer = self._resolve_active_tracer(readiness)
        self.readiness_report_ = replace(readiness, active_tracer=active_tracer)

        self.tracer_ = self._fit_tracer(
            work,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
            item_difficulty_col=item_difficulty_col,
            item_difficulty_map=item_difficulty_map,
            tracer_model=active_tracer,
        )
        # Every bundled tracer reserves an OOV embedding at fit time. Enable it
        # only through this catalog-registration-aware facade.
        if hasattr(self.tracer_, "_allow_unknown_items"):
            self.tracer_._allow_unknown_items = True
        split = KTHoldoutSplit(
            train=work,
            test=work.iloc[0:0].copy(),
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
        )
        self.item_support_ = {item: float(value) for item, value in work.groupby(item_col).size().items()}
        self.concept_support_ = {
            concept: float(value)
            for concept, value in work.groupby(training_concept_col, dropna=False).size().items()
        }
        labels = (work[correct_col].astype(float) >= self.config.correct_threshold).astype(float)
        label_work = work[[item_col, training_concept_col]].copy()
        label_work["__orchid_label__"] = labels
        self.item_correct_ = {
            item: float(value)
            for item, value in label_work.groupby(item_col)["__orchid_label__"].sum().items()
        }
        self.concept_correct_ = {
            concept: float(value)
            for concept, value in label_work.groupby(training_concept_col, dropna=False)["__orchid_label__"].sum().items()
        }
        self.global_correct_ = float(labels.mean()) if len(labels) else 0.5
        self._global_correct_total = float(labels.sum())
        self._global_outcome_count = float(len(labels))
        self.progression_config_ = ProgressionRewardConfig(target_correct=self.config.target_correct)
        has_concept_signal = concept_col is not None or concept_by_item is not None
        resolved_policy = self._resolve_policy(has_concept_signal=has_concept_signal)

        if resolved_policy in {"delayed_gain", "support_delayed_gain"}:
            self.delayed_gain_priors_ = estimate_delayed_gain_priors(
                split,
                concept_col=training_concept_col,
                future_window=self.config.delayed_gain_window,
                threshold=self.config.correct_threshold,
                shrinkage=self.config.delayed_gain_shrinkage,
            )
        else:
            self.delayed_gain_priors_ = None

        if resolved_policy == "support_delayed_gain":
            self.delayed_gain_reward_model_ = fit_delayed_gain_reward_model(
                split,
                concept_col=training_concept_col,
                item_difficulty_col=item_difficulty_col,
                item_gain_prior=(self.delayed_gain_priors_ or {}).get("item_gain_prior", {}),
                concept_gain_prior=(self.delayed_gain_priors_ or {}).get("concept_gain_prior", {}),
                global_gain_prior=float((self.delayed_gain_priors_ or {}).get("global_gain_prior", 0.5)),
                future_window=self.config.delayed_gain_window,
                threshold=self.config.correct_threshold,
                max_examples=self.config.reward_model_max_examples,
                example_weighting=self.config.reward_model_example_weighting,
                max_sample_weight=self.config.reward_model_max_sample_weight,
                cross_fit_folds=self.config.reward_model_cross_fit_folds,
                random_state=self.config.random_state,
                config=self.progression_config_,
                tracer=self.tracer_,
            )
        else:
            self.delayed_gain_reward_model_ = None

        self.policy_ = self._make_policy(resolved_policy)
        self.policy_name_ = resolved_policy
        if hasattr(self.policy_, "seed_history"):
            self.policy_.seed_history(
                work,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                timestamp_col=timestamp_col,
                reset=True,
            )
            self._state_policy = self.policy_
        else:
            self._state_policy = ProgressionValuePolicy(
                self.tracer_,
                difficulty_by_item=self.difficulty_by_item_,
                concept_by_item=self.concept_by_item_,
                config=self.progression_config_,
                correct_threshold=self.config.correct_threshold,
                competence_blend=self.config.progression_competence_blend,
            ).seed_history(
                work,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                timestamp_col=timestamp_col,
                reset=True,
            )
        return self

    @classmethod
    def from_interactions(cls, interactions: pd.DataFrame, **kwargs: Any) -> "AdaptiveLearningRecommender":
        """Create and fit an adaptive-learning recommender in one call."""
        fit_keys = {
            "user_col",
            "item_col",
            "correct_col",
            "timestamp_col",
            "concept_col",
            "item_difficulty_col",
            "item_difficulty_map",
            "concept_by_item",
            "prerequisite_by_concept",
        }
        fit_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in fit_keys}
        return cls(**kwargs).fit(interactions, **fit_kwargs)

    def rank(
        self,
        user_id: Any,
        candidate_item_ids: Sequence[Any],
        *,
        top_k: int = 5,
        enforce_prerequisites: Optional[bool] = None,
        allow_prerequisite_fallback: Optional[bool] = None,
    ) -> list[AdaptiveLearningRecommendation]:
        """Rank candidate items for the next adaptive action."""
        self._require_fitted()
        if top_k <= 0 or not candidate_item_ids:
            return []
        candidates = self._known_candidates(candidate_item_ids)
        if not candidates:
            return []

        enforce = self.config.enforce_prerequisites if enforce_prerequisites is None else bool(enforce_prerequisites)
        allow_fallback = (
            self.config.allow_prerequisite_fallback
            if allow_prerequisite_fallback is None
            else bool(allow_prerequisite_fallback)
        )
        ranked_candidates = candidates
        if enforce and self.prerequisite_by_concept_:
            eligible = [item_id for item_id in candidates if self._prerequisites_met(user_id, item_id)]
            if eligible or not allow_fallback:
                ranked_candidates = eligible
        if not ranked_candidates:
            return []

        raw = self.policy_.rank(user_id, ranked_candidates, top_k=min(int(top_k), len(ranked_candidates)))
        return [self._normalize_recommendation(user_id, rec) for rec in raw]

    def register_items(
        self,
        catalog: pd.DataFrame,
        *,
        item_col: str = "item_id",
        category_col: Optional[str] = None,
        difficulty_col: Optional[str] = None,
    ) -> "AdaptiveLearningRecommender":
        """Register catalog items for OOV serving and live feedback.

        Registered items are assigned the tracer's learned OOV representation
        until a future offline fit gives them item-specific parameters.
        """
        self._require_fitted()
        if item_col not in catalog.columns:
            raise ValueError(f"catalog missing item_col={item_col!r}")
        if category_col is not None and category_col not in catalog.columns:
            raise ValueError(f"catalog missing category_col={category_col!r}")
        if difficulty_col is not None and difficulty_col not in catalog.columns:
            raise ValueError(f"catalog missing difficulty_col={difficulty_col!r}")
        for _, row in catalog.drop_duplicates(subset=[item_col], keep="last").iterrows():
            item_id = row[item_col]
            if pd.isna(item_id):
                raise ValueError("catalog item_id values must not be missing")
            if item_id in self._item_id_set:
                continue
            concept = item_id
            if category_col is not None and not pd.isna(row[category_col]):
                concept = row[category_col]
            difficulty = 0.5
            if difficulty_col is not None and not pd.isna(row[difficulty_col]):
                difficulty = _clamp01(row[difficulty_col])
            self._item_id_set.add(item_id)
            self.item_ids_.append(item_id)
            self.concept_by_item_[item_id] = concept
            self.difficulty_by_item_[item_id] = difficulty
            self.item_support_[item_id] = 0.0
            self.item_correct_[item_id] = 0.0
            self.concept_support_.setdefault(concept, 0.0)
            self.concept_correct_.setdefault(concept, 0.0)
            self._register_policy_item(item_id, concept, difficulty)
        return self

    def registered_candidates(self, candidate_item_ids: Sequence[Any]) -> list[Any]:
        """Return deduplicated catalog items eligible for model feedback."""
        candidates: list[Any] = []
        seen: set[Any] = set()
        for item_id in candidate_item_ids:
            if item_id in self._item_id_set and item_id not in seen:
                candidates.append(item_id)
                seen.add(item_id)
        return candidates

    def feedback_supported(self, item_id: Any) -> bool:
        """Whether an item can be passed to :meth:`observe` in this deployment."""
        return item_id in self._item_id_set

    def observe(
        self,
        user_id: Any,
        item_id: Any,
        correct: Any,
        *,
        timestamp: Optional[Any] = None,
    ) -> Any:
        """Observe one live outcome and update user state."""
        self._require_fitted()
        if item_id not in self._item_id_set:
            raise KeyError(f"Unknown item_id={item_id!r}")
        result = self.policy_.observe(user_id, item_id, correct, timestamp=timestamp)
        if self._state_policy is not None and self._state_policy is not self.policy_:
            self._state_policy.record_outcome(user_id, item_id, correct)
        label = float(float(correct) >= self.config.correct_threshold)
        self.item_support_[item_id] = self.item_support_.get(item_id, 0.0) + 1.0
        self.item_correct_[item_id] = self.item_correct_.get(item_id, 0.0) + label
        concept = self.concept_by_item_.get(item_id, item_id)
        self.concept_support_[concept] = self.concept_support_.get(concept, 0.0) + 1.0
        self.concept_correct_[concept] = self.concept_correct_.get(concept, 0.0) + label
        self._global_correct_total += label
        self._global_outcome_count += 1.0
        self.global_correct_ = self._global_correct_total / self._global_outcome_count
        return result

    def predict_correct(self, user_id: Any, item_id: Any) -> float:
        """Predict the probability that a learner answers an item correctly."""
        self._require_fitted()
        return float(self.tracer_.predict_correct(user_id, item_id))

    def competence_for(self, user_id: Any, concept: Any) -> float:
        """Return rolling competence for a concept when the active policy tracks it."""
        self._require_fitted()
        if self._state_policy is not None:
            return float(self._state_policy.competence_for(user_id, concept))
        return float(self.progression_config_.default_competence if self.progression_config_ else 0.5)

    def mastered_concepts(self, user_id: Any, *, threshold: Optional[float] = None) -> set[Any]:
        """Return concepts above the mastery threshold."""
        self._require_fitted()
        value = self.config.mastery_threshold if threshold is None else float(threshold)
        if self._state_policy is not None:
            return set(self._state_policy.mastered_concepts(user_id, threshold=value))
        return set()

    def diagnostics(self) -> Dict[str, Any]:
        """Return fit and policy diagnostics for logging or model cards."""
        self._require_fitted()
        return {
            "tracer_model": self.config.tracer_model,
            "active_tracer": None if self.readiness_report_ is None else self.readiness_report_.active_tracer,
            "requested_policy": self.config.policy,
            "policy": self.policy_name_,
            "n_users": len(self.user_ids_),
            "n_items": len(self.item_ids_),
            "n_concepts": len(set(self.concept_by_item_.values())),
            "target_correct": self.config.target_correct,
            "has_prerequisites": bool(self.prerequisite_by_concept_),
            "delayed_gain_priors": None
            if self.delayed_gain_priors_ is None
            else {
                "global_gain_prior": self.delayed_gain_priors_["global_gain_prior"],
                "item_priors": len(self.delayed_gain_priors_["item_gain_prior"]),
                "concept_priors": len(self.delayed_gain_priors_["concept_gain_prior"]),
                "shrinkage": self.delayed_gain_priors_["shrinkage"],
            },
            "delayed_gain_reward_model": None
            if self.delayed_gain_reward_model_ is None
            else self.delayed_gain_reward_model_.to_dict(),
            "readiness": None if self.readiness_report_ is None else self.readiness_report_.to_dict(),
        }

    def readiness_report(self) -> AdaptiveLearningReadinessReport:
        """Return the pilot-data assessment used to select the active tracer."""
        self._require_fitted()
        assert self.readiness_report_ is not None
        return self.readiness_report_

    def _fit_tracer(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str,
        item_col: str,
        correct_col: str,
        timestamp_col: Optional[str],
        item_difficulty_col: Optional[str],
        item_difficulty_map: Optional[Mapping[Any, float]],
        tracer_model: Optional[str] = None,
    ) -> Any:
        normalized = (tracer_model or self.config.tracer_model).lower().replace("_", "-")
        if normalized in {"empirical", "baseline"}:
            from .empirical import EmpiricalTracer

            return EmpiricalTracer(correct_threshold=self.config.correct_threshold).fit(
                interactions,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                timestamp_col=timestamp_col,
            )
        if normalized == "sakt":
            from .kt import SAKTTracer

            return SAKTTracer(
                max_seq_len=self.config.max_seq_len,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                correct_threshold=self.config.correct_threshold,
                device=self.config.device,
                random_state=self.config.random_state,
            ).fit(
                interactions,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                timestamp_col=timestamp_col,
            )
        if normalized == "dkt":
            from .kt import DKTTracer

            return DKTTracer(
                max_seq_len=self.config.max_seq_len,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                correct_threshold=self.config.correct_threshold,
                device=self.config.device,
                random_state=self.config.random_state,
            ).fit(
                interactions,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                timestamp_col=timestamp_col,
            )
        if normalized in {"dkvmn", "dkvmn-style"}:
            from .kt import DKVMNTracer

            return DKVMNTracer(
                max_seq_len=self.config.max_seq_len,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                correct_threshold=self.config.correct_threshold,
                device=self.config.device,
                random_state=self.config.random_state,
            ).fit(
                interactions,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                timestamp_col=timestamp_col,
            )
        if normalized in {"akt", "akt-inspired"}:
            from .kt import AKTTracer

            return AKTTracer(
                max_seq_len=self.config.max_seq_len,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                correct_threshold=self.config.correct_threshold,
                device=self.config.device,
                random_state=self.config.random_state,
            ).fit(
                interactions,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                timestamp_col=timestamp_col,
                item_difficulty_col=item_difficulty_col,
                item_difficulty_map=dict(item_difficulty_map or {}),
            )
        if normalized == "saint":
            from .kt import SAINTTracer

            return SAINTTracer(
                max_seq_len=self.config.max_seq_len,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                correct_threshold=self.config.correct_threshold,
                device=self.config.device,
                random_state=self.config.random_state,
            ).fit(
                interactions,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                timestamp_col=timestamp_col,
            )
        if normalized in {"saint+", "saint-plus"}:
            from .kt import SAINTPlusTracer

            return SAINTPlusTracer(
                max_seq_len=self.config.max_seq_len,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                correct_threshold=self.config.correct_threshold,
                device=self.config.device,
                random_state=self.config.random_state,
            ).fit(
                interactions,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                timestamp_col=timestamp_col,
            )
        raise ValueError("tracer_model must be one of 'empirical', 'akt', 'sakt', 'dkt', 'dkvmn', 'saint', or 'saint+'")

    def _resolve_active_tracer(self, readiness: AdaptiveLearningReadinessReport) -> str:
        """Select the transparent pilot baseline unless KT has adequate support."""
        requested = self.config.tracer_model.lower().replace("_", "-")
        if requested in {"empirical", "baseline"}:
            return "empirical"
        if self.config.fallback_to_empirical and not readiness.knowledge_tracing_ready:
            return "empirical"
        return requested

    def _resolve_policy(self, *, has_concept_signal: bool) -> str:
        policy = self.config.policy.lower()
        valid = {"auto", "kt_value", "hybrid", "progression", "canary_progression", "delayed_gain", "support_delayed_gain"}
        if policy not in valid:
            raise ValueError(f"policy must be one of {sorted(valid)}")
        if policy == "auto":
            return "hybrid"
        if policy == "canary_progression":
            return "progression"
        if policy in {"delayed_gain", "support_delayed_gain"} and not has_concept_signal:
            raise ValueError(f"policy={policy!r} requires concept_col or concept_by_item")
        return policy

    def _make_policy(self, policy: str) -> Any:
        if policy == "support_delayed_gain":
            priors = self.delayed_gain_priors_ or {}
            return SupportConstrainedDelayedGainPolicy(
                self.tracer_,
                reward_model=self.delayed_gain_reward_model_,
                difficulty_by_item=self.difficulty_by_item_,
                concept_by_item=self.concept_by_item_,
                item_gain_prior=priors.get("item_gain_prior", {}),
                concept_gain_prior=priors.get("concept_gain_prior", {}),
                global_gain_prior=float(priors.get("global_gain_prior", 0.5)),
                item_support=self.item_support_,
                concept_support=self.concept_support_,
                config=self.progression_config_,
                correct_threshold=self.config.correct_threshold,
                competence_blend=self.config.progression_competence_blend,
            )
        if policy == "delayed_gain":
            priors = self.delayed_gain_priors_ or {}
            return DelayedGainValuePolicy(
                self.tracer_,
                difficulty_by_item=self.difficulty_by_item_,
                concept_by_item=self.concept_by_item_,
                item_gain_prior=priors.get("item_gain_prior", {}),
                concept_gain_prior=priors.get("concept_gain_prior", {}),
                global_gain_prior=float(priors.get("global_gain_prior", 0.5)),
                config=self.progression_config_,
                correct_threshold=self.config.correct_threshold,
                competence_blend=self.config.progression_competence_blend,
            )
        if policy == "hybrid":
            return HybridAdaptivePolicy(
                self.tracer_,
                difficulty_by_item=self.difficulty_by_item_,
                concept_by_item=self.concept_by_item_,
                item_correct=self.item_correct_,
                item_count=self.item_support_,
                concept_correct=self.concept_correct_,
                concept_count=self.concept_support_,
                global_correct=self.global_correct_,
                prior_smoothing=self.config.hybrid_prior_smoothing,
                concept_smoothing=self.config.hybrid_concept_smoothing,
                min_item_support=self.config.hybrid_min_item_support,
                min_concept_support=self.config.hybrid_min_concept_support,
                config=self.progression_config_,
                correct_threshold=self.config.correct_threshold,
                competence_blend=self.config.progression_competence_blend,
                progression_weight=self.config.hybrid_progression_weight,
                item_prior_weight=self.config.hybrid_item_prior_weight,
                concept_prior_weight=self.config.hybrid_concept_prior_weight,
                kt_weight=self.config.hybrid_kt_weight,
                support_weight=self.config.hybrid_support_weight,
                unsupported_penalty_weight=self.config.hybrid_unsupported_penalty_weight,
            )
        if policy == "progression":
            return ProgressionValuePolicy(
                self.tracer_,
                difficulty_by_item=self.difficulty_by_item_,
                concept_by_item=self.concept_by_item_,
                config=self.progression_config_,
                correct_threshold=self.config.correct_threshold,
                competence_blend=self.config.progression_competence_blend,
            )
        return KTValuePolicy(
            self.tracer_,
            target_correct=self.config.target_correct,
            difficulty_by_item=self.difficulty_by_item_,
        )

    def _known_candidates(self, candidate_item_ids: Sequence[Any]) -> list[Any]:
        candidates = []
        seen = set()
        for item_id in candidate_item_ids:
            if item_id in self._item_id_set and item_id not in seen:
                candidates.append(item_id)
                seen.add(item_id)
        return candidates

    def _register_policy_item(self, item_id: Any, concept: Any, difficulty: float) -> None:
        """Extend mutable policy metadata without changing learned weights."""
        policies = [self.policy_, self._state_policy]
        seen: set[int] = set()
        for policy in policies:
            if policy is None or id(policy) in seen:
                continue
            seen.add(id(policy))
            for attribute, value in (
                ("concept_by_item", concept),
                ("difficulty_by_item", difficulty),
            ):
                mapping = getattr(policy, attribute, None)
                if isinstance(mapping, dict):
                    mapping[item_id] = value
            for attribute in ("item_count", "item_correct", "item_support"):
                mapping = getattr(policy, attribute, None)
                if isinstance(mapping, dict):
                    mapping.setdefault(item_id, 0.0)
            for attribute in ("concept_count", "concept_correct", "concept_support"):
                mapping = getattr(policy, attribute, None)
                if isinstance(mapping, dict):
                    mapping.setdefault(concept, 0.0)

    def _prerequisites_met(self, user_id: Any, item_id: Any) -> bool:
        concept = self.concept_by_item_.get(item_id, item_id)
        requirements = self.prerequisite_by_concept_.get(concept, set())
        if not requirements:
            return True
        mastered = self.mastered_concepts(user_id)
        return set(requirements).issubset(mastered)

    def _normalize_recommendation(self, user_id: Any, rec: Any) -> AdaptiveLearningRecommendation:
        item_id = rec.item_id
        concept = getattr(rec, "concept_id", self.concept_by_item_.get(item_id))
        reward: Any = getattr(rec, "reward", None)
        reward_breakdown = reward.to_dict() if hasattr(reward, "to_dict") else None
        return AdaptiveLearningRecommendation(
            item_id=item_id,
            score=float(rec.score),
            outcome_probability=float(rec.p_correct),
            policy=str(self.policy_name_),
            difficulty=_optional_float(getattr(rec, "difficulty", self.difficulty_by_item_.get(item_id))),
            category_id=concept,
            competence=_optional_float(getattr(rec, "competence", None)),
            expected_reward=_optional_float(getattr(rec, "expected_reward", None)),
            stretch_fit=_optional_float(getattr(rec, "stretch_fit", getattr(reward, "stretch_fit", None))),
            expected_gain=_optional_float(getattr(rec, "expected_gain", getattr(reward, "mastery_gain", None))),
            uncertainty=_optional_float(getattr(rec, "uncertainty", None)),
            delayed_gain_prior=_optional_float(getattr(rec, "delayed_gain_prior", None)),
            model_prediction=_optional_float(getattr(rec, "model_prediction", None)),
            support_penalty=float(getattr(rec, "support_penalty", 0.0)),
            item_support=float(getattr(rec, "item_support", self.item_support_.get(item_id, 0.0))),
            concept_support=float(getattr(rec, "concept_support", self.concept_support_.get(concept, 0.0))),
            recent_repetition=int(getattr(rec, "recent_repetition", 0)),
            prerequisites_met=self._prerequisites_met(user_id, item_id),
            feedback_supported=item_id in self._item_id_set,
            reward_breakdown=reward_breakdown,
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("AdaptiveLearningRecommender must be fitted before use")


def _validate_config(config: AdaptiveLearningConfig) -> None:
    if not 0.0 <= config.target_correct <= 1.0:
        raise ValueError("target_correct must be in [0, 1]")
    if not 0.0 <= config.correct_threshold <= 1.0:
        raise ValueError("correct_threshold must be in [0, 1]")
    if not 0.0 <= config.mastery_threshold <= 1.0:
        raise ValueError("mastery_threshold must be in [0, 1]")
    if not 0.0 <= config.progression_competence_blend <= 1.0:
        raise ValueError("progression_competence_blend must be in [0, 1]")
    hybrid_weights = [
        config.hybrid_progression_weight,
        config.hybrid_item_prior_weight,
        config.hybrid_concept_prior_weight,
        config.hybrid_kt_weight,
        config.hybrid_support_weight,
        config.hybrid_unsupported_penalty_weight,
    ]
    if min(hybrid_weights) < 0.0:
        raise ValueError("hybrid policy weights must be non-negative")
    if config.hybrid_prior_smoothing < 0.0 or config.hybrid_concept_smoothing < 0.0:
        raise ValueError("hybrid smoothing values must be non-negative")
    if config.hybrid_min_item_support <= 0.0 or config.hybrid_min_concept_support <= 0.0:
        raise ValueError("hybrid support thresholds must be positive")
    if config.delayed_gain_window < 1:
        raise ValueError("delayed_gain_window must be >= 1")
    if config.min_kt_events < 1 or config.min_kt_users < 1 or config.min_kt_items < 1:
        raise ValueError("knowledge-tracing readiness minimums must be positive")
    if not np.isfinite(float(config.min_kt_median_events_per_user)) or config.min_kt_median_events_per_user < 1.0:
        raise ValueError("min_kt_median_events_per_user must be a finite value >= 1")


def _assess_learning_readiness(
    frame: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    correct_col: str,
    category_col: Optional[str],
    difficulty_col: Optional[str],
    config: AdaptiveLearningConfig,
) -> AdaptiveLearningReadinessReport:
    """Assess support for sequence modeling without inventing a quality claim.

    These are deliberately conservative *starting checks*, not universal data
    requirements. A team should validate its chosen tracer against an authored
    baseline on a chronological holdout before treating the result as ready for
    a live learning pilot.
    """
    labels = (frame[correct_col].astype(float) >= config.correct_threshold).astype(float)
    n_events = int(len(frame))
    n_users = int(frame[user_col].nunique())
    n_items = int(frame[item_col].nunique())
    per_user = frame.groupby(user_col, dropna=False).size()
    per_item = frame.groupby(item_col, dropna=False).size()
    outcome_rate = float(labels.mean())
    outcome_entropy = 0.0
    if 0.0 < outcome_rate < 1.0:
        outcome_entropy = float(
            -outcome_rate * math.log2(outcome_rate) - (1.0 - outcome_rate) * math.log2(1.0 - outcome_rate)
        )
    has_categories = category_col is not None
    n_categories = int(frame[category_col].nunique()) if category_col is not None else 0
    has_difficulty = difficulty_col is not None
    median_per_user = float(per_user.median()) if not per_user.empty else 0.0
    median_per_item = float(per_item.median()) if not per_item.empty else 0.0
    reasons: list[str] = []
    if n_events < config.min_kt_events:
        reasons.append(f"only {n_events} completed interactions; configured KT starting check is {config.min_kt_events}")
    if n_users < config.min_kt_users:
        reasons.append(f"only {n_users} learners; configured KT starting check is {config.min_kt_users}")
    if n_items < config.min_kt_items:
        reasons.append(f"only {n_items} exercises; configured KT starting check is {config.min_kt_items}")
    if median_per_user < config.min_kt_median_events_per_user:
        reasons.append(
            "median learner history is "
            f"{median_per_user:.1f}; configured KT starting check is {config.min_kt_median_events_per_user:.1f}"
        )
    if outcome_rate <= 0.02 or outcome_rate >= 0.98:
        reasons.append("outcomes are almost one-sided, limiting correctness-model discrimination")
    recommendations: list[str] = []
    if reasons:
        recommendations.append(
            "Use the empirical pilot baseline, continue collecting completed learner attempts, and compare future KT "
            "against an authored sequence on a chronological holdout."
        )
    else:
        recommendations.append(
            "Knowledge tracing has enough basic support to evaluate, but it still needs a chronological calibration "
            "comparison against an authored or empirical baseline before live use."
        )
    if not has_categories:
        recommendations.append(
            "Add a stable skill/category mapping before claiming skill-level mastery or using prerequisite rules."
        )
    if not has_difficulty:
        recommendations.append(
            "Add author-reviewed difficulty when available; inferred difficulty is only an outcome-rate proxy."
        )
    return AdaptiveLearningReadinessReport(
        n_events=n_events,
        n_users=n_users,
        n_items=n_items,
        n_categories=n_categories,
        median_events_per_user=median_per_user,
        median_events_per_item=median_per_item,
        outcome_rate=outcome_rate,
        outcome_entropy=outcome_entropy,
        has_categories=has_categories,
        has_difficulty=has_difficulty,
        knowledge_tracing_ready=not reasons,
        active_tracer="unresolved",
        reasons=tuple(reasons),
        recommendations=tuple(recommendations),
    )


def _ordered(frame: pd.DataFrame, *, user_col: str, timestamp_col: Optional[str]) -> pd.DataFrame:
    work = frame.copy()
    if timestamp_col is not None:
        work[timestamp_col] = normalize_timestamps(work[timestamp_col], timestamp_col)
    work["__orchid_order__"] = np.arange(len(work))
    sort_cols = [user_col]
    if timestamp_col is not None:
        sort_cols.append(timestamp_col)
    sort_cols.append("__orchid_order__")
    return work.sort_values(sort_cols, kind="mergesort").drop(columns=["__orchid_order__"])


def _difficulty_by_item(
    frame: pd.DataFrame,
    *,
    item_col: str,
    correct_col: str,
    difficulty_col: Optional[str],
    difficulty_map: Optional[Mapping[Any, float]],
    threshold: float,
) -> Dict[Any, float]:
    if difficulty_col is not None:
        values = {item: _clamp01(value) for item, value in frame.groupby(item_col)[difficulty_col].mean().items()}
    else:
        labels = (frame[correct_col].astype(float) >= float(threshold)).astype(float)
        work = frame[[item_col]].copy()
        work["__label__"] = labels
        global_correct = float(work["__label__"].mean())
        grouped = work.groupby(item_col)["__label__"].agg(["sum", "count"])
        values = {
            item: _clamp01(1.0 - float((row["sum"] + global_correct) / (row["count"] + 1.0)))
            for item, row in grouped.iterrows()
        }
    for item, value in dict(difficulty_map or {}).items():
        values[item] = _clamp01(value)
    return values


def _concept_by_item(
    frame: pd.DataFrame,
    *,
    item_col: str,
    concept_col: Optional[str],
    concept_map: Optional[Mapping[Any, Any]],
) -> tuple[Dict[Any, Any], str]:
    if concept_map is not None:
        values = {item: concept_map.get(item, item) for item in frame[item_col].drop_duplicates().tolist()}
        values.update(dict(concept_map))
        return values, "__orchid_concept__"
    if concept_col is not None:
        values = {
            item: _mode_or_first(group[concept_col])
            for item, group in frame.groupby(item_col, sort=False)
        }
        return values, concept_col
    values = {item: item for item in frame[item_col].drop_duplicates().tolist()}
    return values, "__orchid_concept__"


def _mode_or_first(values: pd.Series) -> Any:
    modes = values.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return values.iloc[0]


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _clamp01(value: Any) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("adaptive-learning numeric inputs must be finite")
    return max(0.0, min(1.0, numeric))
