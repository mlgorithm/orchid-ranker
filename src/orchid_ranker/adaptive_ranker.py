"""Adaptive-first facade for learner-state, policy learning, OPE, and sketch mode."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Optional, Sequence

import pandas as pd

from .adaptive_learning import (
    AdaptiveLearningConfig,
    AdaptiveLearningRecommendation,
    AdaptiveLearningRecommender,
)
from .adaptive_schema import (
    LearnerEvent,
    parse_candidate_list,
    stable_context_hash,
    validate_learner_events,
    validate_logged_decisions,
)
from .offline_policy import CQLDiscretePolicy, CQLTrainingReport
from .ope import LoggedPolicyReport, evaluate_logged_policy

__all__ = [
    "AdaptiveRanker",
    "AdaptiveRankerConfig",
]


@dataclass(frozen=True)
class AdaptiveRankerConfig:
    """Configuration for the adaptive-first Orchid product facade."""

    kt_backbone: str = "akt"
    mode: str = "full"
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
    device: Optional[str] = None
    random_state: Optional[int] = 42
    offline_policy_weight: float = 1.0
    semantic_cold_start_weight: float = 0.50


class AdaptiveRanker:
    """Narrow adaptive-learning interface for Orchid's roadmap target.

    The facade stages the system explicitly: fit KT, optionally fit a
    delayed-gain reward model, fit a conservative logged policy, serve
    recommendations, observe outcomes, and run OPE.
    """

    def __init__(self, config: Optional[AdaptiveRankerConfig] = None, **overrides: Any) -> None:
        valid = {field.name for field in fields(AdaptiveRankerConfig)}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(f"Unknown AdaptiveRankerConfig fields: {unknown}")
        self.config = replace(config or AdaptiveRankerConfig(), **overrides)
        self.recommender_: Optional[AdaptiveLearningRecommender] = None
        self.offline_policy_: Optional[CQLDiscretePolicy] = None
        self.sketch_generator_: Optional[Any] = None
        self.semantic_encoder_: Optional[Any] = None
        self._events: Optional[pd.DataFrame] = None
        self._fit_kwargs: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.recommender_ is not None and self.recommender_.is_fitted

    def fit_kt(
        self,
        events: pd.DataFrame,
        *,
        learner_col: str = "learner_id",
        item_col: str = "item_id",
        correct_col: str = "correct",
        timestamp_col: str = "ts",
        concept_col: Optional[str] = "concept_id",
        item_difficulty_col: Optional[str] = None,
        **fit_kwargs: Any,
    ) -> "AdaptiveRanker":
        """Fit learner-state tracing and the default adaptive policy."""
        backbone = self.config.kt_backbone.lower().replace("_", "-")
        if backbone not in {"akt", "sakt", "akt-inspired", "dkt", "dkvmn", "dkvmn-style", "saint", "saint+", "saint-plus"}:
            raise ValueError("kt_backbone must be one of 'akt', 'sakt', 'dkt', 'dkvmn', 'saint', or 'saint+'")
        work = validate_learner_events(
            events,
            learner_col=learner_col,
            ts_col=timestamp_col,
            item_col=item_col,
            correct_col=correct_col,
        )
        adaptive_config = self._adaptive_config(policy=self.config.policy)
        concept_arg = concept_col if concept_col is not None and concept_col in work.columns else None
        self.recommender_ = AdaptiveLearningRecommender(adaptive_config).fit(
            work,
            user_col=learner_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
            concept_col=concept_arg,
            item_difficulty_col=item_difficulty_col,
            **fit_kwargs,
        )
        self._events = work.copy()
        self._fit_kwargs = {
            "user_col": learner_col,
            "item_col": item_col,
            "correct_col": correct_col,
            "timestamp_col": timestamp_col,
            "concept_col": concept_arg,
            "item_difficulty_col": item_difficulty_col,
            **fit_kwargs,
        }
        return self

    def fit_reward_model(self) -> "AdaptiveRanker":
        """Refit the adaptive stack with support-constrained delayed-gain modeling."""
        if self._events is None:
            raise RuntimeError("fit_kt must be called before fit_reward_model")
        if self._fit_kwargs.get("concept_col") is None:
            raise ValueError("fit_reward_model requires concept_id/concept_col data")
        adaptive_config = self._adaptive_config(policy="support_delayed_gain")
        self.recommender_ = AdaptiveLearningRecommender(adaptive_config).fit(self._events, **self._fit_kwargs)
        return self

    def fit_policy(
        self,
        logged_decisions: pd.DataFrame,
        *,
        algo: str = "cql",
        reward_col: str = "reward",
        **policy_kwargs: Any,
    ) -> CQLTrainingReport:
        """Fit a conservative offline policy from logged decisions."""
        normalized = algo.lower()
        if normalized not in {"cql", "conservative", "tabular_cql"}:
            raise NotImplementedError("Only the tabular CQL policy learner is implemented in this roadmap slice")
        validate_logged_decisions(logged_decisions, reward_col=reward_col)
        self.offline_policy_ = CQLDiscretePolicy(
            random_state=self.config.random_state,
            **policy_kwargs,
        ).fit(logged_decisions, reward_col=reward_col)
        assert self.offline_policy_.report_ is not None
        return self.offline_policy_.report_

    def attach_sketch_generator(self, generator: Any) -> "AdaptiveRanker":
        """Attach a sketch-mode candidate generator with a ``candidates`` method."""
        if not hasattr(generator, "candidates"):
            raise TypeError("generator must expose a candidates(...) method")
        self.sketch_generator_ = generator
        return self

    def fit_semantic_items(
        self,
        catalog: pd.DataFrame,
        *,
        item_col: str = "item_id",
        text_col: str = "item_text",
        metadata_cols: Optional[Sequence[str]] = None,
        **encoder_kwargs: Any,
    ) -> "AdaptiveRanker":
        """Fit the lightweight semantic item encoder used for cold-start retrieval."""
        from .semantic import SemanticItemEncoder

        self.semantic_encoder_ = SemanticItemEncoder(**encoder_kwargs).fit(
            catalog,
            item_col=item_col,
            text_col=text_col,
            metadata_cols=metadata_cols,
        )
        return self

    def attach_semantic_encoder(self, encoder: Any) -> "AdaptiveRanker":
        """Attach a pre-fitted semantic encoder exposing ``similar_items``."""
        if not getattr(encoder, "is_fitted", False):
            raise RuntimeError("semantic encoder must be fitted before attachment")
        if not hasattr(encoder, "similar_items"):
            raise TypeError("semantic encoder must expose similar_items(...)")
        self.semantic_encoder_ = encoder
        return self

    def recommend(
        self,
        learner_id: Any,
        candidate_item_ids: Optional[Sequence[Any]] = None,
        *,
        top_k: int = 10,
        mode: Optional[str] = None,
        context_hash: Optional[str] = None,
        concept_goal: Optional[Any] = None,
        item_query_vec: Optional[Sequence[float]] = None,
        item_query_text: Optional[str] = None,
    ) -> list[AdaptiveLearningRecommendation]:
        """Recommend next items using exact or sketch-mode candidates."""
        self._require_fitted()
        assert self.recommender_ is not None
        active_mode = self.config.mode if mode is None else mode
        candidates = list(candidate_item_ids or [])
        if not candidates and item_query_text is not None and self.semantic_encoder_ is not None:
            candidates = self.semantic_encoder_.similar_items(
                item_query_text,
                top_k=max(top_k, 50),
            )
        if not candidates and active_mode == "sketch":
            if self.sketch_generator_ is None:
                raise RuntimeError("sketch mode requires an attached SketchCandidateGenerator")
            candidates = self.sketch_generator_.candidates(
                learner_id,
                concept_goal,
                item_query_vec=item_query_vec,
                top_m=max(top_k, 50),
            )
        if not candidates:
            candidates = list(self.recommender_.item_ids_)
        known_items = set(self.recommender_.item_ids_)
        known_candidates = [item_id for item_id in candidates if item_id in known_items]
        cold_candidates = [item_id for item_id in candidates if item_id not in known_items]
        recs: list[AdaptiveLearningRecommendation] = []
        if known_candidates:
            recs.extend(self.recommender_.rank(learner_id, known_candidates, top_k=max(top_k, len(known_candidates))))
        recs.extend(
            self._semantic_cold_start_recommendations(
                learner_id,
                cold_candidates,
                query_text=item_query_text,
                top_k=max(top_k, len(cold_candidates)),
            )
        )
        if self.offline_policy_ is not None:
            ctx = context_hash or stable_context_hash(learner_id, concept_goal)
            q_scores = self.offline_policy_.score(ctx, [rec.item_id for rec in recs])
            recs = sorted(
                recs,
                key=lambda rec: (
                    rec.score + self.config.offline_policy_weight * q_scores.get(rec.item_id, 0.0),
                    rec.score,
                    str(rec.item_id),
                ),
                reverse=True,
            )
        else:
            recs = sorted(recs, key=lambda rec: (rec.score, str(rec.item_id)), reverse=True)
        return recs[: min(int(top_k), len(recs))]

    def observe(self, event: LearnerEvent | None = None, **kwargs: Any) -> Any:
        """Observe one learner event and update live state."""
        self._require_fitted()
        assert self.recommender_ is not None
        learner_event = event or LearnerEvent(**kwargs)
        return self.recommender_.observe(
            learner_event.learner_id,
            learner_event.item_id,
            learner_event.correct,
            timestamp=learner_event.ts,
        )

    def ope_report(
        self,
        logged_decisions: pd.DataFrame,
        *,
        reward_col: str = "reward",
        propensity_col: str = "propensity",
        target_probability_col: str = "target_probability",
        max_weight: Optional[float] = None,
    ) -> LoggedPolicyReport:
        """Evaluate the fitted conservative policy from logged decisions."""
        work = validate_logged_decisions(logged_decisions, reward_col=reward_col).copy()
        if target_probability_col not in work.columns:
            if self.offline_policy_ is None:
                raise RuntimeError("fit_policy or target_probability_col is required before ope_report")
            work[target_probability_col] = [
                self._target_probability(row)
                for _, row in work.iterrows()
            ]
        return evaluate_logged_policy(
            work,
            reward_col=reward_col,
            propensity_col=propensity_col,
            target_probability_col=target_probability_col,
            max_weight=max_weight,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return adaptive, reward-model, policy, and sketch diagnostics."""
        self._require_fitted()
        assert self.recommender_ is not None
        data = self.recommender_.diagnostics()
        data["adaptive_ranker"] = {
            "mode": self.config.mode,
            "kt_backbone": self.config.kt_backbone,
            "offline_policy": None if self.offline_policy_ is None else self.offline_policy_.to_dict(),
            "has_sketch_generator": self.sketch_generator_ is not None,
            "semantic_encoder": None if self.semantic_encoder_ is None else self.semantic_encoder_.diagnostics(),
        }
        return data

    def _semantic_cold_start_recommendations(
        self,
        learner_id: Any,
        candidate_item_ids: Sequence[Any],
        *,
        query_text: Optional[str],
        top_k: int,
    ) -> list[AdaptiveLearningRecommendation]:
        if top_k <= 0 or not candidate_item_ids or self.semantic_encoder_ is None or self.recommender_ is None:
            return []
        encoded_items = set(getattr(self.semantic_encoder_, "item_ids_", []))
        cold_items = []
        seen = set()
        for item_id in candidate_item_ids:
            if item_id in encoded_items and item_id not in seen:
                cold_items.append(item_id)
                seen.add(item_id)
        if not cold_items:
            return []

        if query_text:
            semantic_scores = self.semantic_encoder_.scores(query_text, candidate_item_ids=cold_items)
        else:
            semantic_scores = {item_id: 0.5 for item_id in cold_items}

        recs: list[AdaptiveLearningRecommendation] = []
        semantic_weight = _clamp01(self.config.semantic_cold_start_weight)
        structure_weight = 1.0 - semantic_weight
        for item_id in cold_items:
            if item_id not in semantic_scores:
                continue
            metadata = self.semantic_encoder_.metadata(item_id) if hasattr(self.semantic_encoder_, "metadata") else {}
            concept = _first_metadata_value(metadata, ("concept_id", "concept", "skill_id", "skill"))
            difficulty = _optional_float(_first_metadata_value(metadata, ("difficulty", "item_difficulty", "difficulty_score")))
            prerequisites_met = self._cold_start_prerequisites_met(learner_id, concept)
            if (
                not prerequisites_met
                and self.recommender_.config.enforce_prerequisites
                and not self.recommender_.config.allow_prerequisite_fallback
            ):
                continue
            competence = self.recommender_.competence_for(learner_id, concept) if concept is not None else None
            p_correct = _cold_start_correctness_prior(competence=competence, difficulty=difficulty)
            normalizer = max(self.config.target_correct, 1.0 - self.config.target_correct, 1e-6)
            stretch_fit = max(0.0, 1.0 - abs(p_correct - self.config.target_correct) / normalizer)
            uncertainty = max(0.0, 1.0 - 2.0 * abs(p_correct - 0.5))
            structure_score = 0.7 * stretch_fit + 0.3 * uncertainty
            score = semantic_weight * _clamp01(semantic_scores[item_id]) + structure_weight * structure_score
            recs.append(
                AdaptiveLearningRecommendation(
                    item_id=item_id,
                    score=float(score),
                    p_correct=float(p_correct),
                    policy="semantic_cold_start",
                    difficulty=difficulty,
                    concept_id=concept,
                    competence=competence,
                    expected_reward=float(score),
                    stretch_fit=float(stretch_fit),
                    uncertainty=float(uncertainty),
                    support_penalty=1.0,
                    item_support=0.0,
                    concept_support=0.0,
                    prerequisites_met=prerequisites_met,
                )
            )
        recs.sort(key=lambda rec: (rec.score, str(rec.item_id)), reverse=True)
        return recs[: min(int(top_k), len(recs))]

    def _cold_start_prerequisites_met(self, learner_id: Any, concept: Any) -> bool:
        if self.recommender_ is None or concept is None:
            return True
        requirements = self.recommender_.prerequisite_by_concept_.get(concept, set())
        if not requirements:
            return True
        return set(requirements).issubset(self.recommender_.mastered_concepts(learner_id))

    def _adaptive_config(self, *, policy: str) -> AdaptiveLearningConfig:
        return AdaptiveLearningConfig(
            tracer_model=self.config.kt_backbone,
            policy=policy,
            target_correct=self.config.target_correct,
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
        )

    def _target_probability(self, row: pd.Series) -> float:
        assert self.offline_policy_ is not None
        candidates = parse_candidate_list(row["candidate_item_ids"])
        chosen = row["chosen_item_id"]
        selected = self.offline_policy_.recommend(row["context_hash"], candidates, top_k=1)
        return float(bool(selected and selected[0] == chosen))

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("AdaptiveRanker.fit_kt must be called before serving")


def _first_metadata_value(metadata: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _clamp01(value: Any) -> float:
    numeric = _optional_float(value)
    if numeric is None:
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _cold_start_correctness_prior(*, competence: Optional[float], difficulty: Optional[float]) -> float:
    if competence is None and difficulty is None:
        return 0.5
    if competence is None:
        return _clamp01(1.0 - float(difficulty))
    if difficulty is None:
        return _clamp01(float(competence))
    return _clamp01(0.5 + 0.5 * (float(competence) - float(difficulty)))
