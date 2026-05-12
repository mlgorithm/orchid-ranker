"""First-class adaptive-learning recommender API.

This module composes Orchid's strongest in-repo adaptive-learning pieces:
sequence-aware knowledge tracing, progression reward scoring, delayed-gain
priors, support-aware direct reward modeling, and prerequisite gating.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .delayed_gain import DelayedGainRewardModel, fit_delayed_gain_reward_model
from .kt_benchmark import KTHoldoutSplit
from .learning_policy import (
    DelayedGainValuePolicy,
    KTValuePolicy,
    ProgressionValuePolicy,
    SupportConstrainedDelayedGainPolicy,
)
from .policy_benchmark import estimate_delayed_gain_priors
from .progression_reward import ProgressionRewardConfig

__all__ = [
    "AdaptiveLearningConfig",
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
    mastery_threshold: float = 0.80
    enforce_prerequisites: bool = True
    allow_prerequisite_fallback: bool = False
    device: Optional[str] = None
    random_state: Optional[int] = 42


@dataclass(frozen=True)
class AdaptiveLearningRecommendation:
    """Normalized recommendation from the adaptive-learning policy stack."""

    item_id: Any
    score: float
    p_correct: float
    policy: str
    difficulty: Optional[float] = None
    concept_id: Optional[Any] = None
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
    reward_breakdown: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveLearningRecommender:
    """Adaptive-learning recommender with a production-oriented default stack.

    ``policy="auto"`` resolves to ``ProgressionValuePolicy``. Delayed-gain and
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
        self.item_ids_: list[Any] = []
        self.user_ids_: list[Any] = []
        self._training_concept_col: Optional[str] = None
        self._item_col: str = "item_id"

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

        self.tracer_ = self._fit_tracer(
            work,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
            item_difficulty_col=item_difficulty_col,
            item_difficulty_map=item_difficulty_map,
        )
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
            for concept, value in work.groupby(training_concept_col).size().items()
        }
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
        """Rank candidate items for the next adaptive-learning action."""
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

    recommend = rank

    def observe(self, user_id: Any, item_id: Any, correct: Any) -> Any:
        """Observe one live outcome and update learner state."""
        self._require_fitted()
        if item_id not in set(self.item_ids_):
            raise KeyError(f"Unknown item_id={item_id!r}")
        result = self.policy_.observe(user_id, item_id, correct)
        if self._state_policy is not None and self._state_policy is not self.policy_:
            self._state_policy.record_outcome(user_id, item_id, correct)
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
        }

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
    ) -> Any:
        normalized = self.config.tracer_model.lower().replace("_", "-")
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
        raise ValueError("tracer_model must be 'akt' or 'sakt'")

    def _resolve_policy(self, *, has_concept_signal: bool) -> str:
        policy = self.config.policy.lower()
        valid = {"auto", "kt_value", "progression", "delayed_gain", "support_delayed_gain"}
        if policy not in valid:
            raise ValueError(f"policy must be one of {sorted(valid)}")
        if policy == "auto":
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
            )
        if policy == "progression":
            return ProgressionValuePolicy(
                self.tracer_,
                difficulty_by_item=self.difficulty_by_item_,
                concept_by_item=self.concept_by_item_,
                config=self.progression_config_,
                correct_threshold=self.config.correct_threshold,
            )
        return KTValuePolicy(
            self.tracer_,
            target_correct=self.config.target_correct,
            difficulty_by_item=self.difficulty_by_item_,
        )

    def _known_candidates(self, candidate_item_ids: Sequence[Any]) -> list[Any]:
        known = set(self.item_ids_)
        candidates = []
        seen = set()
        for item_id in candidate_item_ids:
            if item_id in known and item_id not in seen:
                candidates.append(item_id)
                seen.add(item_id)
        return candidates

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
        reward = getattr(rec, "reward", None)
        reward_breakdown = reward.to_dict() if hasattr(reward, "to_dict") else None
        return AdaptiveLearningRecommendation(
            item_id=item_id,
            score=float(rec.score),
            p_correct=float(rec.p_correct),
            policy=str(self.policy_name_),
            difficulty=_optional_float(getattr(rec, "difficulty", self.difficulty_by_item_.get(item_id))),
            concept_id=concept,
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
    if config.delayed_gain_window < 1:
        raise ValueError("delayed_gain_window must be >= 1")


def _ordered(frame: pd.DataFrame, *, user_col: str, timestamp_col: Optional[str]) -> pd.DataFrame:
    work = frame.copy()
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
