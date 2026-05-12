"""Learning policies that turn KT predictions into next-item rankings."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from .delayed_gain import make_delayed_gain_features
from .progression_reward import (
    ProgressionRewardBreakdown,
    ProgressionRewardConfig,
    expected_progression_reward,
)

__all__ = [
    "DelayedGainPolicyRecommendation",
    "DelayedGainValuePolicy",
    "KTPolicyRecommendation",
    "KTValuePolicy",
    "ProgressionPolicyRecommendation",
    "ProgressionValuePolicy",
    "SupportConstrainedDelayedGainPolicy",
]


@dataclass(frozen=True)
class KTPolicyRecommendation:
    """Recommendation from a KT-guided learning policy."""

    item_id: Any
    score: float
    p_correct: float
    stretch_fit: float
    uncertainty: float
    expected_gain: float
    difficulty: Optional[float] = None


class KTValuePolicy:
    """Rank eligible learning items from predicted correctness.

    The policy is deliberately conservative: it does not claim to learn a full
    reinforcement-learning value function. It combines three transparent terms:

    - stretch fit around a target correctness probability
    - uncertainty, highest near 0.5
    - expected gain proxy, ``p_correct * (1 - p_correct)``
    """

    def __init__(
        self,
        tracer: Any,
        *,
        target_correct: float = 0.70,
        stretch_weight: float = 1.0,
        uncertainty_weight: float = 0.25,
        gain_weight: float = 0.50,
        difficulty_by_item: Optional[Mapping[Any, float]] = None,
    ) -> None:
        if not 0.0 <= target_correct <= 1.0:
            raise ValueError("target_correct must be in [0, 1]")
        if stretch_weight < 0 or uncertainty_weight < 0 or gain_weight < 0:
            raise ValueError("policy weights must be non-negative")
        self.tracer = tracer
        self.target_correct = float(target_correct)
        self.stretch_weight = float(stretch_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.gain_weight = float(gain_weight)
        self.difficulty_by_item = dict(difficulty_by_item or {})

    def rank(
        self,
        user_id: Any,
        candidate_item_ids: Sequence[Any],
        *,
        top_k: int = 5,
    ) -> list[KTPolicyRecommendation]:
        """Rank candidate items for a learner."""
        if top_k <= 0 or not candidate_item_ids:
            return []
        predictions: Dict[Any, float] = self.tracer.predict_many(user_id, list(candidate_item_ids))
        recs = []
        normalizer = max(self.target_correct, 1.0 - self.target_correct, 1e-6)
        for item_id in candidate_item_ids:
            p_correct = float(predictions[item_id])
            stretch_fit = max(0.0, 1.0 - abs(p_correct - self.target_correct) / normalizer)
            uncertainty = max(0.0, 1.0 - 2.0 * abs(p_correct - 0.5))
            expected_gain = max(0.0, p_correct * (1.0 - p_correct))
            score = (
                self.stretch_weight * stretch_fit
                + self.uncertainty_weight * uncertainty
                + self.gain_weight * expected_gain
            )
            recs.append(
                KTPolicyRecommendation(
                    item_id=item_id,
                    score=float(score),
                    p_correct=p_correct,
                    stretch_fit=float(stretch_fit),
                    uncertainty=float(uncertainty),
                    expected_gain=float(expected_gain),
                    difficulty=self.difficulty_by_item.get(item_id),
                )
            )
        recs.sort(key=lambda rec: (rec.score, rec.stretch_fit, str(rec.item_id)), reverse=True)
        return recs[: min(int(top_k), len(recs))]

    def observe(self, user_id: Any, item_id: Any, correct: Any) -> Any:
        """Forward a live outcome into the underlying tracer."""
        return self.tracer.observe(user_id, item_id, correct)


@dataclass(frozen=True)
class ProgressionPolicyRecommendation:
    """Recommendation from a progression-aware learning policy."""

    item_id: Any
    score: float
    p_correct: float
    difficulty: float
    concept_id: Any
    competence: float
    expected_reward: float
    reward: ProgressionRewardBreakdown
    recent_repetition: int = 0


class ProgressionValuePolicy:
    """Rank items by expected progression value, not only correctness.

    The policy uses predicted correctness from a tracer but scores candidates
    with a progression reward that includes mastery-gain potential, stretch fit,
    item difficulty, and repetition/easy-item penalties.
    """

    def __init__(
        self,
        tracer: Any,
        *,
        difficulty_by_item: Optional[Mapping[Any, float]] = None,
        concept_by_item: Optional[Mapping[Any, Any]] = None,
        config: Optional[ProgressionRewardConfig] = None,
        correct_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= float(correct_threshold) <= 1.0:
            raise ValueError("correct_threshold must be in [0, 1]")
        self.tracer = tracer
        self.difficulty_by_item = dict(difficulty_by_item or {})
        self.concept_by_item = dict(concept_by_item or {})
        self.config = config or ProgressionRewardConfig()
        self.correct_threshold = float(correct_threshold)
        self._correct_by_user_concept: Dict[Any, Dict[Any, list[int]]] = {}
        self._recent_concepts_by_user: Dict[Any, list[Any]] = {}

    def seed_history(
        self,
        interactions: Any,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        correct_col: str = "correct",
        timestamp_col: Optional[str] = None,
        reset: bool = True,
    ) -> "ProgressionValuePolicy":
        """Warm-start progression state from historical learner outcomes.

        This updates only the policy's competence and repetition summaries. It
        does not forward events into the tracer, which should already have been
        fitted on the same history.
        """
        required = {user_col, item_col, correct_col}
        if timestamp_col is not None:
            required.add(timestamp_col)
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions missing required columns: {sorted(missing)}")
        if reset:
            self._correct_by_user_concept = {}
            self._recent_concepts_by_user = {}

        work = interactions.copy()
        work["__orchid_order__"] = range(len(work))
        sort_cols = [user_col]
        if timestamp_col is not None:
            sort_cols.append(timestamp_col)
        sort_cols.append("__orchid_order__")
        work = work.sort_values(sort_cols, kind="mergesort")
        for user_id, item_id, correct in work[[user_col, item_col, correct_col]].itertuples(index=False, name=None):
            self.record_outcome(user_id, item_id, correct)
        return self

    def rank(
        self,
        user_id: Any,
        candidate_item_ids: Sequence[Any],
        *,
        top_k: int = 5,
    ) -> list[ProgressionPolicyRecommendation]:
        """Rank candidate items by expected progression reward."""
        if top_k <= 0 or not candidate_item_ids:
            return []
        predictions: Dict[Any, float] = self.tracer.predict_many(user_id, list(candidate_item_ids))
        recs = []
        for item_id in candidate_item_ids:
            p_correct = float(predictions[item_id])
            difficulty = self._difficulty_for(item_id)
            concept = self._concept_for(item_id)
            competence = self._competence_for(user_id, concept)
            recent_repetition = self._recent_repetition(user_id, concept)
            reward = expected_progression_reward(
                p_correct=p_correct,
                difficulty=difficulty,
                competence=competence,
                recent_repetition=recent_repetition,
                config=self.config,
            )
            recs.append(
                ProgressionPolicyRecommendation(
                    item_id=item_id,
                    score=reward.expected_reward,
                    p_correct=p_correct,
                    difficulty=reward.difficulty,
                    concept_id=concept,
                    competence=competence,
                    expected_reward=reward.expected_reward,
                    reward=reward,
                    recent_repetition=recent_repetition,
                )
            )
        recs.sort(
            key=lambda rec: (
                rec.score,
                rec.reward.stretch_fit,
                rec.reward.mastery_gain,
                -rec.reward.easy_penalty,
                str(rec.item_id),
            ),
            reverse=True,
        )
        return recs[: min(int(top_k), len(recs))]

    def observe(self, user_id: Any, item_id: Any, correct: Any) -> Any:
        """Forward a live outcome and update progression history."""
        result = self.tracer.observe(user_id, item_id, correct)
        self.record_outcome(user_id, item_id, correct)
        return result

    def record_outcome(self, user_id: Any, item_id: Any, correct: Any) -> None:
        """Update progression summaries without forwarding into the tracer."""
        self._record_progression_outcome(user_id, item_id, correct)

    def competence_for(self, user_id: Any, concept: Any) -> float:
        """Return the current rolling competence estimate for a concept."""
        return self._competence_for(user_id, concept)

    def mastered_concepts(self, user_id: Any, *, threshold: float = 0.80) -> set[Any]:
        """Return concepts whose rolling correctness is above ``threshold``."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        concepts = self._correct_by_user_concept.get(user_id, {})
        return {
            concept
            for concept in concepts
            if self._competence_for(user_id, concept) >= float(threshold)
        }

    def recent_repetition_for(self, user_id: Any, concept: Any) -> int:
        """Return how often ``concept`` appears in the recent repetition window."""
        return self._recent_repetition(user_id, concept)

    def _record_progression_outcome(self, user_id: Any, item_id: Any, correct: Any) -> None:
        concept = self._concept_for(item_id)
        label = int(float(correct) >= self.correct_threshold)
        by_concept = self._correct_by_user_concept.setdefault(user_id, {})
        history = by_concept.setdefault(concept, [])
        history.append(label)
        if len(history) > 20:
            del history[: len(history) - 20]
        recent = self._recent_concepts_by_user.setdefault(user_id, [])
        recent.append(concept)
        if len(recent) > max(1, self.config.repetition_window):
            del recent[: len(recent) - max(1, self.config.repetition_window)]

    def _difficulty_for(self, item_id: Any) -> float:
        return float(self.difficulty_by_item.get(item_id, 0.5))

    def _concept_for(self, item_id: Any) -> Any:
        return self.concept_by_item.get(item_id, "__global__")

    def _competence_for(self, user_id: Any, concept: Any) -> float:
        history = self._correct_by_user_concept.get(user_id, {}).get(concept, [])
        if not history:
            return float(self.config.default_competence)
        return float(sum(history[-10:]) / len(history[-10:]))

    def _recent_repetition(self, user_id: Any, concept: Any) -> int:
        recent = self._recent_concepts_by_user.get(user_id, [])
        return sum(1 for seen in recent if seen == concept)


@dataclass(frozen=True)
class DelayedGainPolicyRecommendation:
    """Recommendation from a delayed-gain-aware learning policy."""

    item_id: Any
    score: float
    p_correct: float
    difficulty: float
    concept_id: Any
    competence: float
    expected_reward: float
    delayed_gain_prior: float
    reward: ProgressionRewardBreakdown
    model_prediction: Optional[float] = None
    support_penalty: float = 0.0
    item_support: float = 0.0
    concept_support: float = 0.0
    recent_repetition: int = 0


class DelayedGainValuePolicy(ProgressionValuePolicy):
    """Rank items with historical delayed same-concept gain priors.

    Public KT logs rarely expose the platform policy, so this remains a
    transparent value heuristic rather than a learned RL policy. It combines the
    progression reward with training-only delayed-gain priors by item/concept so
    the policy can prefer exercises that historically preceded future same-skill
    improvement instead of only optimizing immediate progression score.
    """

    def __init__(
        self,
        tracer: Any,
        *,
        difficulty_by_item: Optional[Mapping[Any, float]] = None,
        concept_by_item: Optional[Mapping[Any, Any]] = None,
        item_gain_prior: Optional[Mapping[Any, float]] = None,
        concept_gain_prior: Optional[Mapping[Any, float]] = None,
        global_gain_prior: float = 0.5,
        config: Optional[ProgressionRewardConfig] = None,
        correct_threshold: float = 0.5,
        progression_weight: float = 0.35,
        delayed_gain_weight: float = 0.55,
        stretch_weight: float = 0.10,
    ) -> None:
        if progression_weight < 0 or delayed_gain_weight < 0 or stretch_weight < 0:
            raise ValueError("delayed-gain policy weights must be non-negative")
        super().__init__(
            tracer,
            difficulty_by_item=difficulty_by_item,
            concept_by_item=concept_by_item,
            config=config,
            correct_threshold=correct_threshold,
        )
        self.item_gain_prior = {item: _clamp01(value) for item, value in dict(item_gain_prior or {}).items()}
        self.concept_gain_prior = {
            concept: _clamp01(value) for concept, value in dict(concept_gain_prior or {}).items()
        }
        self.global_gain_prior = _clamp01(global_gain_prior)
        self.progression_weight = float(progression_weight)
        self.delayed_gain_weight = float(delayed_gain_weight)
        self.policy_stretch_weight = float(stretch_weight)

    def rank(
        self,
        user_id: Any,
        candidate_item_ids: Sequence[Any],
        *,
        top_k: int = 5,
    ) -> list[DelayedGainPolicyRecommendation]:
        """Rank candidate items by delayed-gain-aware expected value."""
        if top_k <= 0 or not candidate_item_ids:
            return []
        predictions: Dict[Any, float] = self.tracer.predict_many(user_id, list(candidate_item_ids))
        recs = []
        for item_id in candidate_item_ids:
            p_correct = float(predictions[item_id])
            difficulty = self._difficulty_for(item_id)
            concept = self._concept_for(item_id)
            competence = self._competence_for(user_id, concept)
            recent_repetition = self._recent_repetition(user_id, concept)
            reward = expected_progression_reward(
                p_correct=p_correct,
                difficulty=difficulty,
                competence=competence,
                recent_repetition=recent_repetition,
                config=self.config,
            )
            delayed_prior = self._delayed_gain_prior(item_id, concept)
            score = (
                self.delayed_gain_weight * delayed_prior
                + self.progression_weight * reward.expected_reward
                + self.policy_stretch_weight * reward.stretch_fit
            )
            recs.append(
                DelayedGainPolicyRecommendation(
                    item_id=item_id,
                    score=float(score),
                    p_correct=p_correct,
                    difficulty=reward.difficulty,
                    concept_id=concept,
                    competence=competence,
                    expected_reward=float(score),
                    delayed_gain_prior=delayed_prior,
                    reward=reward,
                    model_prediction=None,
                    support_penalty=0.0,
                    item_support=0.0,
                    concept_support=0.0,
                    recent_repetition=recent_repetition,
                )
            )
        recs.sort(
            key=lambda rec: (
                rec.score,
                rec.delayed_gain_prior,
                rec.reward.stretch_fit,
                rec.reward.mastery_gain,
                str(rec.item_id),
            ),
            reverse=True,
        )
        return recs[: min(int(top_k), len(recs))]

    def _delayed_gain_prior(self, item_id: Any, concept: Any) -> float:
        return float(
            self.item_gain_prior.get(
                item_id,
                self.concept_gain_prior.get(concept, self.global_gain_prior),
            )
        )


class SupportConstrainedDelayedGainPolicy(DelayedGainValuePolicy):
    """Rank by learned delayed-gain value while penalizing weak log support."""

    def __init__(
        self,
        tracer: Any,
        *,
        reward_model: Any,
        difficulty_by_item: Optional[Mapping[Any, float]] = None,
        concept_by_item: Optional[Mapping[Any, Any]] = None,
        item_gain_prior: Optional[Mapping[Any, float]] = None,
        concept_gain_prior: Optional[Mapping[Any, float]] = None,
        global_gain_prior: float = 0.5,
        item_support: Optional[Mapping[Any, float]] = None,
        concept_support: Optional[Mapping[Any, float]] = None,
        config: Optional[ProgressionRewardConfig] = None,
        correct_threshold: float = 0.5,
        model_weight: float = 0.65,
        prior_weight: float = 0.20,
        progression_weight: float = 0.10,
        stretch_weight: float = 0.05,
        support_penalty_weight: float = 0.15,
        min_item_support: float = 5.0,
        min_concept_support: float = 20.0,
    ) -> None:
        if reward_model is None:
            raise ValueError("reward_model is required")
        if min(model_weight, prior_weight, progression_weight, stretch_weight, support_penalty_weight) < 0:
            raise ValueError("support-constrained policy weights must be non-negative")
        super().__init__(
            tracer,
            difficulty_by_item=difficulty_by_item,
            concept_by_item=concept_by_item,
            item_gain_prior=item_gain_prior,
            concept_gain_prior=concept_gain_prior,
            global_gain_prior=global_gain_prior,
            config=config,
            correct_threshold=correct_threshold,
            progression_weight=progression_weight,
            delayed_gain_weight=prior_weight,
            stretch_weight=stretch_weight,
        )
        self.reward_model = reward_model
        self.item_support = {item: float(value) for item, value in dict(item_support or {}).items()}
        self.concept_support = {concept: float(value) for concept, value in dict(concept_support or {}).items()}
        self.model_weight = float(model_weight)
        self.prior_weight = float(prior_weight)
        self.support_progression_weight = float(progression_weight)
        self.support_stretch_weight = float(stretch_weight)
        self.support_penalty_weight = float(support_penalty_weight)
        self.min_item_support = float(min_item_support)
        self.min_concept_support = float(min_concept_support)

    def rank(
        self,
        user_id: Any,
        candidate_item_ids: Sequence[Any],
        *,
        top_k: int = 5,
    ) -> list[DelayedGainPolicyRecommendation]:
        """Rank candidate items by learned delayed-gain value and support."""
        if top_k <= 0 or not candidate_item_ids:
            return []
        predictions: Dict[Any, float] = self.tracer.predict_many(user_id, list(candidate_item_ids))
        payloads = []
        feature_rows = []
        for item_id in candidate_item_ids:
            p_correct = float(predictions[item_id])
            difficulty = self._difficulty_for(item_id)
            concept = self._concept_for(item_id)
            competence = self._competence_for(user_id, concept)
            recent_repetition = self._recent_repetition(user_id, concept)
            reward = expected_progression_reward(
                p_correct=p_correct,
                difficulty=difficulty,
                competence=competence,
                recent_repetition=recent_repetition,
                config=self.config,
            )
            delayed_prior = self._delayed_gain_prior(item_id, concept)
            item_count = self.item_support.get(item_id, 0.0)
            concept_count = self.concept_support.get(concept, 0.0)
            features = make_delayed_gain_features(
                p_correct=p_correct,
                difficulty=reward.difficulty,
                competence=competence,
                recent_repetition=recent_repetition,
                progression_expected_reward=reward.expected_reward,
                progression_stretch_fit=reward.stretch_fit,
                progression_mastery_gain=reward.mastery_gain,
                delayed_gain_prior=delayed_prior,
                item_support=item_count,
                concept_support=concept_count,
                min_item_support=self.min_item_support,
                min_concept_support=self.min_concept_support,
            )
            payloads.append(
                {
                    "item_id": item_id,
                    "p_correct": p_correct,
                    "concept": concept,
                    "competence": competence,
                    "recent_repetition": recent_repetition,
                    "reward": reward,
                    "delayed_prior": delayed_prior,
                    "item_count": item_count,
                    "concept_count": concept_count,
                    "support_penalty": self._support_penalty(item_count, concept_count),
                }
            )
            feature_rows.append(features)
        model_predictions = _predict_many(self.reward_model, feature_rows)
        recs = []
        for payload, model_prediction in zip(payloads, model_predictions):
            item_id = payload["item_id"]
            reward = payload["reward"]
            delayed_prior = payload["delayed_prior"]
            support_penalty = payload["support_penalty"]
            score = (
                self.model_weight * model_prediction
                + self.prior_weight * delayed_prior
                + self.support_progression_weight * reward.expected_reward
                + self.support_stretch_weight * reward.stretch_fit
                - self.support_penalty_weight * support_penalty
            )
            recs.append(
                DelayedGainPolicyRecommendation(
                    item_id=item_id,
                    score=float(score),
                    p_correct=payload["p_correct"],
                    difficulty=reward.difficulty,
                    concept_id=payload["concept"],
                    competence=payload["competence"],
                    expected_reward=model_prediction,
                    delayed_gain_prior=delayed_prior,
                    reward=reward,
                    model_prediction=model_prediction,
                    support_penalty=support_penalty,
                    item_support=payload["item_count"],
                    concept_support=payload["concept_count"],
                    recent_repetition=payload["recent_repetition"],
                )
            )
        recs.sort(
            key=lambda rec: (
                rec.score,
                -rec.support_penalty,
                rec.model_prediction if rec.model_prediction is not None else 0.0,
                rec.delayed_gain_prior,
                str(rec.item_id),
            ),
            reverse=True,
        )
        return recs[: min(int(top_k), len(recs))]

    def _support_penalty(self, item_support: float, concept_support: float) -> float:
        item_gap = max(0.0, self.min_item_support - float(item_support)) / max(self.min_item_support, 1.0)
        concept_gap = max(0.0, self.min_concept_support - float(concept_support)) / max(self.min_concept_support, 1.0)
        return min(1.0, 0.7 * item_gap + 0.3 * concept_gap)


def _clamp01(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.5
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _predict_many(model: Any, feature_rows: Sequence[Mapping[str, float]]) -> list[float]:
    if hasattr(model, "predict_many"):
        return [_clamp01(value) for value in model.predict_many(feature_rows)]
    return [_clamp01(model.predict_one(features)) for features in feature_rows]
