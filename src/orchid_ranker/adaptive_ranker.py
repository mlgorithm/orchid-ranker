"""The single public interface for outcome-driven adaptive recommendation."""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .adaptive_learning import (
    AdaptiveLearningConfig,
    AdaptiveLearningRecommendation,
    AdaptiveLearningRecommender,
)
from .adaptive_schema import (
    DecisionOutcome,
    LoggedDecision,
    normalize_timestamp,
    parse_candidate_list,
    stable_context_hash,
    validate_decision_outcomes,
    validate_logged_decisions,
    validate_user_events,
)
from .decision_store import DecisionOutcomeStore, InMemoryDecisionStore
from .offline_policy import CQLDiscretePolicy, CQLTrainingReport
from .ope import (
    BootstrapLoggedPolicyReport,
    BootstrapPolicyComparisonReport,
    LoggedPolicyReport,
    OPERolloutGateReport,
    PolicyComparisonReport,
    bootstrap_compare_logged_policies,
    bootstrap_logged_policy,
    compare_logged_policies,
    evaluate_logged_policy,
    evaluate_rollout_gate,
)

__all__ = [
    "AdaptiveRanker",
    "AdaptiveRankerConfig",
    "RollingPolicyUpdateReport",
    "ShadowDeploymentReport",
]


@dataclass(frozen=True)
class AdaptiveRankerConfig:
    """Configuration for the adaptive-first Orchid product facade."""

    kt_backbone: str = "akt"
    mode: str = "full"
    policy: str = "auto"
    target_outcome: float = 0.70
    max_seq_len: int = 50
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 128
    outcome_threshold: float = 0.5
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
    offline_policy_weight: float = 1.0
    offline_policy_max_weight: float = 0.35
    offline_policy_normalize: bool = True
    offline_policy_min_effect: float = 0.0
    offline_policy_min_coverage: float = 0.05
    offline_policy_min_ess_fraction: float = 0.05
    offline_policy_max_clipped_fraction: float = 0.20
    semantic_cold_start_weight: float = 0.50
    allow_catalog_fallback: bool = False
    exploration_min_item_support: float = 1.0
    exploration_min_outcome_probability: float = 0.20
    exploration_max_outcome_probability: float = 0.95


@dataclass(frozen=True)
class RollingPolicyUpdateReport:
    """One policy update trained on the past and gated on a future window."""

    training: CQLTrainingReport
    gate: OPERolloutGateReport
    bootstrap_ope: Optional[BootstrapPolicyComparisonReport]
    train_events: int
    evaluation_events: int
    train_users: int
    evaluation_users: int
    train_start: Any
    train_end: Any
    evaluation_start: Any
    evaluation_end: Any

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["training"] = self.training.to_dict()
        data["gate"] = self.gate.to_dict()
        data["bootstrap_ope"] = None if self.bootstrap_ope is None else self.bootstrap_ope.to_dict()
        return data


@dataclass(frozen=True)
class ShadowDeploymentReport:
    """Operational and outcome diagnostics for shadow-serving decisions."""

    n_decisions: int
    n_outcomes: int
    outcome_coverage: float
    unique_users: int
    unique_items_chosen: int
    candidate_count_mean: float
    exploration_fraction: float
    propensity_mean: float
    propensity_min: float
    reward_mean: Optional[float]
    outcome_mean: Optional[float]
    score_regret_mean: Optional[float]
    calibration_brier: Optional[float]
    calibration_bias: Optional[float]
    reward_drift: Optional[float]
    calibration_drift: Optional[float]
    policy_versions: tuple[str, ...]
    bootstrap_ope: Optional[BootstrapPolicyComparisonReport]
    rollout_gate: Optional[OPERolloutGateReport]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bootstrap_ope"] = None if self.bootstrap_ope is None else self.bootstrap_ope.to_dict()
        data["rollout_gate"] = None if self.rollout_gate is None else self.rollout_gate.to_dict()
        return data


class AdaptiveRanker:
    """Choose the next item, observe the outcome, and adapt.

    Most applications only need :meth:`fit`, :meth:`recommend`, and
    :meth:`observe`. Orchid selects its internal adaptive policy; callers do not
    need to choose a model.
    """

    def __init__(
        self,
        config: Optional[AdaptiveRankerConfig] = None,
        *,
        decision_store: Optional[DecisionOutcomeStore] = None,
        **overrides: Any,
    ) -> None:
        valid = {field.name for field in fields(AdaptiveRankerConfig)}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(f"Unknown AdaptiveRankerConfig fields: {unknown}")
        self.config = replace(config or AdaptiveRankerConfig(), **overrides)
        self.recommender_: Optional[AdaptiveLearningRecommender] = None
        self.offline_policy_: Optional[CQLDiscretePolicy] = None
        self.offline_policy_gate_: Optional[OPERolloutGateReport] = None
        self.offline_policy_bootstrap_: Optional[BootstrapPolicyComparisonReport] = None
        self.last_policy_gate_: Optional[OPERolloutGateReport] = None
        self.last_policy_evidence_: Optional[PolicyComparisonReport | BootstrapPolicyComparisonReport] = None
        self.rolling_policy_report_: Optional[RollingPolicyUpdateReport] = None
        self.sketch_generator_: Optional[Any] = None
        self.semantic_encoder_: Optional[Any] = None
        self._events: Optional[pd.DataFrame] = None
        self._fit_kwargs: dict[str, Any] = {}
        self._catalog_registration: Optional[tuple[pd.DataFrame, str, Optional[str], Optional[str]]] = None
        self.decision_store: DecisionOutcomeStore = decision_store or InMemoryDecisionStore()
        _require_decision_store(self.decision_store)
        # One ranker owns mutable learner state, decision records, and deployed
        # policy references. Serializing public operations makes that contract
        # explicit and prevents readers from observing a partially updated model.
        self._state_lock = threading.RLock()
        self._decision_lock = self._state_lock
        self._rng = np.random.default_rng(self.config.random_state)
        self._deployment_version: Optional[str] = None

    @property
    def is_fitted(self) -> bool:
        with self._state_lock:
            return self.recommender_ is not None and self.recommender_.is_fitted

    def fit(
        self,
        events: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        outcome_col: str = "outcome",
        timestamp_col: str = "timestamp",
        category_col: Optional[str] = None,
        difficulty_col: Optional[str] = None,
        catalog: Optional[pd.DataFrame] = None,
        catalog_item_col: str = "item_id",
        catalog_category_col: str = "category_id",
        catalog_difficulty_col: str = "difficulty",
        prerequisite_by_concept: Optional[Mapping[Any, Sequence[Any]]] = None,
        **fit_kwargs: Any,
    ) -> "AdaptiveRanker":
        """Fit from a chronological table of user outcomes.

        The schema is ``user_id``, ``item_id``, ``outcome``, and ``timestamp``.
        Outcomes must be binary ``0`` or ``1``.

        ``category_col`` and ``difficulty_col`` are optional. They can improve
        ranking when the application has meaningful item categories or an
        externally defined difficulty signal. A separate exercise ``catalog``
        is the preferred learning-product path: it supplies category and
        difficulty metadata for both historical and newly registered exercises.
        """
        with self._state_lock:
            for role, column in (
                ("user", user_col),
                ("item", item_col),
                ("outcome", outcome_col),
                ("timestamp", timestamp_col),
            ):
                if column not in events.columns:
                    raise ValueError(f"events must include {role} column {column!r}")
            try:
                outcomes = pd.to_numeric(events[outcome_col], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError("outcome values must be binary 0 or 1") from exc
            if outcomes.isna().any() or not outcomes.isin([0, 1]).all():
                raise ValueError("outcome values must be binary 0 or 1")
            resolved_category_col = category_col
            if resolved_category_col is None and "category_id" in events.columns:
                resolved_category_col = "category_id"
            if resolved_category_col is not None and resolved_category_col not in events.columns:
                raise ValueError(f"events must include category column {resolved_category_col!r}")
            if difficulty_col is not None and difficulty_col not in events.columns:
                raise ValueError(f"events must include difficulty column {difficulty_col!r}")
            if prerequisite_by_concept is not None:
                if "prerequisite_by_concept" in fit_kwargs:
                    raise TypeError("pass prerequisite_by_concept only once")
                fit_kwargs["prerequisite_by_concept"] = prerequisite_by_concept
            prepared_events, resolved_category_col, resolved_difficulty_col, prepared_catalog = _prepare_learning_catalog(
                events,
                item_col=item_col,
                category_col=resolved_category_col,
                difficulty_col=difficulty_col,
                catalog=catalog,
                catalog_item_col=catalog_item_col,
                catalog_category_col=catalog_category_col,
                catalog_difficulty_col=catalog_difficulty_col,
            )
            return self._fit_events(
                prepared_events,
                user_col=user_col,
                item_col=item_col,
                outcome_col=outcome_col,
                timestamp_col=timestamp_col,
                category_col=resolved_category_col,
                difficulty_col=resolved_difficulty_col,
                catalog=prepared_catalog,
                catalog_item_col=catalog_item_col,
                catalog_category_col=catalog_category_col,
                catalog_difficulty_col=catalog_difficulty_col,
                **fit_kwargs,
            )

    def _fit_events(
        self,
        events: pd.DataFrame,
        *,
        user_col: str,
        item_col: str,
        outcome_col: str,
        timestamp_col: str,
        category_col: Optional[str],
        difficulty_col: Optional[str],
        catalog: Optional[pd.DataFrame] = None,
        catalog_item_col: str = "item_id",
        catalog_category_col: str = "category_id",
        catalog_difficulty_col: str = "difficulty",
        **fit_kwargs: Any,
    ) -> "AdaptiveRanker":
        """Fit the internal outcome-tracing and ranking stack."""
        backbone = self.config.kt_backbone.lower().replace("_", "-")
        if backbone not in {
            "empirical",
            "baseline",
            "akt",
            "sakt",
            "akt-inspired",
            "dkt",
            "dkvmn",
            "dkvmn-style",
            "saint",
            "saint+",
            "saint-plus",
        }:
            raise ValueError("kt_backbone must be one of 'empirical', 'akt', 'sakt', 'dkt', 'dkvmn', 'saint', or 'saint+'")
        work = validate_user_events(
            events,
            user_col=user_col,
            timestamp_col=timestamp_col,
            item_col=item_col,
            outcome_col=outcome_col,
        )
        adaptive_config = self._adaptive_config(policy=self.config.policy)
        category_arg = category_col if category_col is not None and category_col in work.columns else None
        candidate_recommender = AdaptiveLearningRecommender(adaptive_config).fit(
            work,
            user_col=user_col,
            item_col=item_col,
            correct_col=outcome_col,
            timestamp_col=timestamp_col,
            concept_col=category_arg,
            item_difficulty_col=difficulty_col,
            **fit_kwargs,
        )
        if catalog is not None:
            candidate_recommender.register_items(
                catalog,
                item_col=catalog_item_col,
                category_col=catalog_category_col if catalog_category_col in catalog.columns else None,
                difficulty_col=catalog_difficulty_col if catalog_difficulty_col in catalog.columns else None,
            )
            self._catalog_registration = (
                catalog.copy(),
                catalog_item_col,
                catalog_category_col if catalog_category_col in catalog.columns else None,
                catalog_difficulty_col if catalog_difficulty_col in catalog.columns else None,
            )
        else:
            self._catalog_registration = None
        self.recommender_ = candidate_recommender
        self._events = work.copy()
        self._fit_kwargs = {
            "user_col": user_col,
            "item_col": item_col,
            "correct_col": outcome_col,
            "timestamp_col": timestamp_col,
            "concept_col": category_arg,
            "item_difficulty_col": difficulty_col,
            **fit_kwargs,
        }
        self.offline_policy_ = None
        self.offline_policy_gate_ = None
        self.offline_policy_bootstrap_ = None
        self.rolling_policy_report_ = None
        self._deployment_version = self._derive_deployment_version()
        return self

    def fit_reward_model(self) -> "AdaptiveRanker":
        """Refit the adaptive stack with support-constrained delayed-gain modeling."""
        with self._state_lock:
            if self._events is None:
                raise RuntimeError("fit must be called before fit_reward_model")
            if self._fit_kwargs.get("concept_col") is None:
                raise ValueError("fit_reward_model requires category data")
            adaptive_config = self._adaptive_config(policy="support_delayed_gain")
            self.recommender_ = AdaptiveLearningRecommender(adaptive_config).fit(self._events, **self._fit_kwargs)
            if self._catalog_registration is not None:
                catalog, item_col, category_col, difficulty_col = self._catalog_registration
                self.recommender_.register_items(
                    catalog,
                    item_col=item_col,
                    category_col=category_col,
                    difficulty_col=difficulty_col,
                )
            self.offline_policy_ = None
            self.offline_policy_gate_ = None
            self.offline_policy_bootstrap_ = None
            self._deployment_version = self._derive_deployment_version()
            return self

    def fit_policy(
        self,
        logged_decisions: pd.DataFrame,
        *,
        algo: str = "cql",
        reward_col: str = "reward",
        evaluation_decisions: Optional[pd.DataFrame] = None,
        cluster_bootstrap_samples: int = 300,
        cluster_col: str = "user_id",
        min_evaluation_events: int = 30,
        min_evaluation_users: int = 30,
        **policy_kwargs: Any,
    ) -> CQLTrainingReport:
        """Fit, evaluate, and atomically promote a conservative offline policy.

        Promotion always requires chronologically held-out logs. The candidate
        is compared with the explicit historical logging-policy baseline and
        replaces the active policy only when the rollout gate passes.
        """
        normalized = algo.lower()
        if normalized not in {"cql", "conservative", "tabular_cql"}:
            raise NotImplementedError("Only the tabular CQL policy learner is implemented in this roadmap slice")
        self._require_fitted()
        if evaluation_decisions is None:
            raise ValueError("fit_policy requires held-out evaluation_decisions before a policy can be promoted")
        if cluster_bootstrap_samples < 1:
            raise ValueError("fit_policy requires cluster_bootstrap_samples >= 1")
        if min_evaluation_events < 1 or min_evaluation_users < 1:
            raise ValueError("minimum evaluation event and user counts must be >= 1")
        training = validate_logged_decisions(logged_decisions, reward_col=reward_col)
        evaluation = validate_logged_decisions(evaluation_decisions, reward_col=reward_col)
        if cluster_col not in evaluation.columns:
            raise ValueError(f"cluster_col={cluster_col!r} is not present in evaluation_decisions")
        if len(evaluation) < min_evaluation_events:
            raise ValueError(
                f"evaluation_decisions has {len(evaluation)} events; requires at least {min_evaluation_events}"
            )
        evaluation_users = int(evaluation[cluster_col].nunique())
        if evaluation_users < min_evaluation_users:
            raise ValueError(
                f"evaluation_decisions has {evaluation_users} {cluster_col} values; "
                f"requires at least {min_evaluation_users}"
            )
        if training["timestamp"].max() >= evaluation["timestamp"].min():
            raise ValueError("evaluation_decisions must be strictly later than policy training")
        _require_disjoint_policy_logs(training, evaluation)

        candidate_policy = CQLDiscretePolicy(
            random_state=self.config.random_state,
            **policy_kwargs,
        ).fit(training, reward_col=reward_col)
        evidence = self._evaluate_candidate_policy(
            candidate_policy,
            evaluation,
            reward_col=reward_col,
            cluster_bootstrap_samples=cluster_bootstrap_samples,
            cluster_col=cluster_col,
        )
        gate = evaluate_rollout_gate(
            evidence,
            min_effect=self.config.offline_policy_min_effect,
            min_ess_fraction=self.config.offline_policy_min_ess_fraction,
            min_coverage=self.config.offline_policy_min_coverage,
            max_clipped_fraction=self.config.offline_policy_max_clipped_fraction,
        )
        with self._state_lock:
            self.last_policy_evidence_ = evidence
            self.last_policy_gate_ = gate
            if gate.allowed:
                self.offline_policy_ = candidate_policy
                self.offline_policy_gate_ = gate
                self.offline_policy_bootstrap_ = (
                    evidence if isinstance(evidence, BootstrapPolicyComparisonReport) else None
                )
                self._deployment_version = self._derive_deployment_version()
        assert candidate_policy.report_ is not None
        return candidate_policy.report_

    def fit_policy_rolling(
        self,
        logged_decisions: Optional[pd.DataFrame] = None,
        *,
        reward_col: str = "reward",
        train_window: Optional[int] = None,
        evaluation_window: int = 100,
        min_train_events: int = 100,
        min_evaluation_events: int = 30,
        min_evaluation_users: int = 30,
        cluster_bootstrap_samples: int = 300,
        cluster_col: str = "user_id",
        **policy_kwargs: Any,
    ) -> RollingPolicyUpdateReport:
        """Fit on a trailing past window and gate on a strictly future window."""
        if evaluation_window < 1:
            raise ValueError("evaluation_window must be >= 1")
        if train_window is not None and train_window < 1:
            raise ValueError("train_window must be >= 1 when supplied")
        if min_train_events < 1 or min_evaluation_events < 1 or min_evaluation_users < 1:
            raise ValueError("minimum rolling-window event and user counts must be >= 1")
        if cluster_bootstrap_samples < 1:
            raise ValueError("rolling updates require cluster_bootstrap_samples >= 1")

        source = (
            self.decision_log_frame(completed_only=True)
            if logged_decisions is None
            else logged_decisions.copy()
        )
        work = validate_logged_decisions(source, reward_col=reward_col).copy()
        if cluster_col not in work.columns:
            raise ValueError(f"cluster_col={cluster_col!r} is not present in logged decisions")
        work = work.sort_values(
            ["timestamp", "decision_id"] if "decision_id" in work.columns else ["timestamp"],
            kind="mergesort",
        )
        timestamps = work["timestamp"].drop_duplicates().sort_values(kind="mergesort").tolist()
        if len(timestamps) < 2:
            raise ValueError("rolling policy updates require at least two distinct decision timestamps")

        boundary: Any = None
        evaluation = work.iloc[0:0]
        for candidate_boundary in reversed(timestamps[1:]):
            candidate_evaluation = work[work["timestamp"] >= candidate_boundary]
            if len(candidate_evaluation) >= evaluation_window:
                boundary = candidate_boundary
                evaluation = candidate_evaluation
                break
        if boundary is None:
            boundary = timestamps[-1]
            evaluation = work[work["timestamp"] >= boundary]
        training = work[work["timestamp"] < boundary]
        if train_window is not None:
            training = training.tail(int(train_window))
        if len(training) < min_train_events:
            raise ValueError(
                f"rolling training window has {len(training)} events; requires at least {min_train_events}"
            )
        if len(evaluation) < min_evaluation_events:
            raise ValueError(
                f"rolling evaluation window has {len(evaluation)} events; requires at least {min_evaluation_events}"
            )
        evaluation_users = int(evaluation[cluster_col].nunique())
        if evaluation_users < min_evaluation_users:
            raise ValueError(
                f"rolling evaluation window has {evaluation_users} users; "
                f"requires at least {min_evaluation_users}"
            )
        if training["timestamp"].max() >= evaluation["timestamp"].min():
            raise RuntimeError("rolling policy split is not strictly chronological")

        training_report = self.fit_policy(
            training,
            reward_col=reward_col,
            evaluation_decisions=evaluation,
            cluster_bootstrap_samples=cluster_bootstrap_samples,
            cluster_col=cluster_col,
            min_evaluation_events=min_evaluation_events,
            min_evaluation_users=min_evaluation_users,
            **policy_kwargs,
        )
        assert self.last_policy_gate_ is not None
        report = RollingPolicyUpdateReport(
            training=training_report,
            gate=self.last_policy_gate_,
            bootstrap_ope=(
                self.last_policy_evidence_
                if isinstance(self.last_policy_evidence_, BootstrapPolicyComparisonReport)
                else None
            ),
            train_events=int(len(training)),
            evaluation_events=int(len(evaluation)),
            train_users=int(training[cluster_col].nunique()),
            evaluation_users=evaluation_users,
            train_start=training["timestamp"].min(),
            train_end=training["timestamp"].max(),
            evaluation_start=evaluation["timestamp"].min(),
            evaluation_end=evaluation["timestamp"].max(),
        )
        self.rolling_policy_report_ = report
        return report

    def attach_sketch_generator(self, generator: Any) -> "AdaptiveRanker":
        """Attach a sketch-mode candidate generator with a ``candidates`` method."""
        if not hasattr(generator, "candidates"):
            raise TypeError("generator must expose a candidates(...) method")
        with self._state_lock:
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
        """Fit semantic retrieval and register its catalog for live feedback.

        Catalog items absent from fitting history use Orchid's OOV item embedding
        until the next offline refit. They are nevertheless registered, so a
        served item can be logged and observed safely in the same process.
        """
        from .semantic import SemanticItemEncoder

        with self._state_lock:
            self._require_fitted()
            self.semantic_encoder_ = SemanticItemEncoder(**encoder_kwargs).fit(
                catalog,
                item_col=item_col,
                text_col=text_col,
                metadata_cols=metadata_cols,
            )
            self.register_items(catalog, item_col=item_col)
            return self

    def register_items(
        self,
        catalog: pd.DataFrame,
        *,
        item_col: str = "item_id",
        category_col: str = "category_id",
        difficulty_col: str = "difficulty",
    ) -> "AdaptiveRanker":
        """Register catalog items that may be served and observed before refitting.

        Newly registered items use the fitted tracer's OOV item representation;
        this is deliberately conservative. Their feedback is retained in the
        live learner state and incorporated as item-specific model parameters on
        the next offline ``fit``.
        """
        with self._state_lock:
            self._require_fitted()
            assert self.recommender_ is not None
            self.recommender_.register_items(
                catalog,
                item_col=item_col,
                category_col=category_col if category_col in catalog.columns else None,
                difficulty_col=difficulty_col if difficulty_col in catalog.columns else None,
            )
            return self

    def attach_semantic_encoder(self, encoder: Any) -> "AdaptiveRanker":
        """Attach a pre-fitted semantic encoder exposing ``similar_items``."""
        if not getattr(encoder, "is_fitted", False):
            raise RuntimeError("semantic encoder must be fitted before attachment")
        if not hasattr(encoder, "similar_items"):
            raise TypeError("semantic encoder must expose similar_items(...)")
        with self._state_lock:
            self.semantic_encoder_ = encoder
            return self

    def recommend(
        self,
        user_id: Any,
        candidate_item_ids: Optional[Sequence[Any]] = None,
        *,
        top_k: int = 10,
        mode: Optional[str] = None,
        context_hash: Optional[str] = None,
        concept_goal: Optional[Any] = None,
        item_query_vec: Optional[Sequence[float]] = None,
        item_query_text: Optional[str] = None,
    ) -> list[AdaptiveLearningRecommendation]:
        """Rank an exact eligible candidate set.

        ``None`` permits configured candidate generation. An explicit empty
        sequence is an empty eligible set and always returns ``[]``; it never
        expands to the training catalog.
        """
        with self._state_lock:
            base_recs = self._base_recommendations(
                user_id,
                candidate_item_ids,
                top_k=top_k,
                mode=mode,
                concept_goal=concept_goal,
                item_query_vec=item_query_vec,
                item_query_text=item_query_text,
            )
            if not base_recs:
                return []
            context = context_hash or stable_context_hash(user_id, concept_goal)
            recs = self._apply_offline_overlay(base_recs, context_hash=context)
            return recs[: min(int(top_k), len(recs))]

    def _base_recommendations(
        self,
        user_id: Any,
        candidate_item_ids: Optional[Sequence[Any]],
        *,
        top_k: int,
        mode: Optional[str],
        concept_goal: Optional[Any],
        item_query_vec: Optional[Sequence[float]],
        item_query_text: Optional[str],
        enforce_prerequisites: Optional[bool] = None,
    ) -> list[AdaptiveLearningRecommendation]:
        """Produce base adaptive scores before an optional CQL overlay.

        The unblended scores are retained in decision logs.  They are the
        historical component needed to evaluate the exact hybrid+CQL action
        rule during a later policy promotion.
        """
        self._require_fitted()
        assert self.recommender_ is not None
        active_mode = self.config.mode if mode is None else mode
        top_k = int(top_k)
        if top_k <= 0 or (candidate_item_ids is not None and len(candidate_item_ids) == 0):
            return []
        candidates = list(candidate_item_ids) if candidate_item_ids is not None else []
        if not candidates and item_query_text is not None and self.semantic_encoder_ is not None:
            candidates = self.semantic_encoder_.similar_items(
                item_query_text,
                top_k=max(top_k, 50),
            )
        if not candidates and active_mode == "sketch":
            if self.sketch_generator_ is None:
                raise RuntimeError("sketch mode requires an attached SketchCandidateGenerator")
            candidates = self.sketch_generator_.candidates(
                user_id,
                concept_goal,
                item_query_vec=item_query_vec,
                top_m=max(top_k, 50),
            )
        if not candidates:
            if candidate_item_ids is None and self.config.allow_catalog_fallback:
                candidates = list(self.recommender_.item_ids_)
            else:
                return []
        known_candidates = self.recommender_.registered_candidates(candidates)
        known_set = set(known_candidates)
        cold_candidates = [item_id for item_id in candidates if item_id not in known_set]
        recs: list[AdaptiveLearningRecommendation] = []
        if known_candidates:
            recs.extend(
                self.recommender_.rank(
                    user_id,
                    known_candidates,
                    top_k=max(top_k, len(known_candidates)),
                    enforce_prerequisites=enforce_prerequisites,
                )
            )
        recs.extend(
            self._semantic_cold_start_recommendations(
                user_id,
                cold_candidates,
                query_text=item_query_text,
                top_k=max(top_k, len(cold_candidates)),
            )
        )
        return sorted(recs, key=lambda rec: (rec.score, str(rec.item_id)), reverse=True)

    def recommend_and_log(
        self,
        user_id: Any,
        candidate_item_ids: Sequence[Any],
        *,
        timestamp: Any,
        top_k: int = 10,
        exploration: float = 0.0,
        min_item_support: Optional[float] = None,
        min_outcome_probability: Optional[float] = None,
        max_outcome_probability: Optional[float] = None,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None,
        require_prerequisites: bool = True,
        allow_unsupported_feedback: bool = False,
        policy_version: Optional[str] = None,
        context_hash: Optional[str] = None,
        concept_goal: Optional[Any] = None,
        decision_id: Optional[str] = None,
        decision_metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[list[AdaptiveLearningRecommendation], LoggedDecision]:
        """Choose, return, and register one decision with exact action probabilities.

        Exploration is epsilon-uniform over the candidates that survive
        prerequisite, support, correctness, and optional difficulty constraints.
        The chosen item is returned first; the immutable log contains the entire
        safe decision-time candidate set.  Supplying ``decision_id`` makes a
        retried serving request idempotent: the original decision is returned
        and never re-sampled or overwritten. ``decision_metadata`` records
        application-owned immutable context, such as an experiment arm or
        catalog version, alongside Orchid's derived policy metadata.
        """
        self._require_fitted()
        normalized_timestamp = normalize_timestamp(timestamp)
        if int(top_k) < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= float(exploration) <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        if min_item_support is None:
            min_item_support = (
                self.config.exploration_min_item_support if float(exploration) > 0.0 else 0.0
            )
        else:
            min_item_support = float(min_item_support)
        min_outcome_probability = (
            self.config.exploration_min_outcome_probability
            if min_outcome_probability is None
            else float(min_outcome_probability)
        )
        max_outcome_probability = (
            self.config.exploration_max_outcome_probability
            if max_outcome_probability is None
            else float(max_outcome_probability)
        )
        if float(min_item_support) < 0.0:
            raise ValueError("min_item_support must be non-negative")
        if not 0.0 <= float(min_outcome_probability) <= float(max_outcome_probability) <= 1.0:
            raise ValueError("outcome probability constraints must satisfy 0 <= min <= max <= 1")
        if min_difficulty is not None and not 0.0 <= float(min_difficulty) <= 1.0:
            raise ValueError("min_difficulty must be in [0, 1]")
        if max_difficulty is not None and not 0.0 <= float(max_difficulty) <= 1.0:
            raise ValueError("max_difficulty must be in [0, 1]")
        if min_difficulty is not None and max_difficulty is not None and min_difficulty > max_difficulty:
            raise ValueError("min_difficulty must be <= max_difficulty")
        candidates = list(dict.fromkeys(candidate_item_ids))
        if not candidates:
            raise ValueError("recommend_and_log requires an explicit non-empty candidate set")
        resolved_decision_metadata = _normalize_decision_metadata(decision_metadata)
        ctx = context_hash or stable_context_hash(user_id, concept_goal)
        with self._decision_lock:
            assert self.recommender_ is not None
            resolved_policy_version = self._deployment_version if policy_version is None else str(policy_version)
            if not resolved_policy_version:
                raise ValueError("policy_version must be non-empty")
            resolved_decision_id = None if decision_id is None else str(decision_id)
            if resolved_decision_id == "":
                raise ValueError("decision_id must be non-empty when supplied")
            request_fingerprint = _decision_request_fingerprint(
                user_id=user_id,
                timestamp=normalized_timestamp,
                candidate_item_ids=candidates,
                exploration=exploration,
                min_item_support=min_item_support,
                min_outcome_probability=min_outcome_probability,
                max_outcome_probability=max_outcome_probability,
                min_difficulty=min_difficulty,
                max_difficulty=max_difficulty,
                require_prerequisites=require_prerequisites,
                allow_unsupported_feedback=allow_unsupported_feedback,
                policy_version=None if policy_version is None else str(policy_version),
                context_hash=ctx,
                concept_goal=concept_goal,
                decision_metadata=resolved_decision_metadata,
            )
            if resolved_decision_id is not None:
                existing = self.decision_store.get_decision(resolved_decision_id)
                if existing is not None:
                    _require_matching_decision_request(existing, request_fingerprint)
                    return self._recommendations_from_logged_decision(existing, top_k=top_k), existing
            base_ranked = self._base_recommendations(
                user_id,
                candidates,
                top_k=len(candidates),
                mode=None,
                concept_goal=concept_goal,
                item_query_vec=None,
                item_query_text=None,
                enforce_prerequisites=require_prerequisites,
            )
            base_scores = {rec.item_id: float(rec.score) for rec in base_ranked}
            ranked = self._apply_offline_overlay(base_ranked, context_hash=ctx)
            safe = [
                rec
                for rec in ranked
                if self._is_safe_exploration_candidate(
                    rec,
                    min_item_support=min_item_support,
                    min_outcome_probability=min_outcome_probability,
                    max_outcome_probability=max_outcome_probability,
                    min_difficulty=min_difficulty,
                    max_difficulty=max_difficulty,
                    require_prerequisites=require_prerequisites,
                )
            ]
            if not safe:
                raise ValueError("no candidate satisfies the configured adaptive serving constraints")
            unsupported = [rec.item_id for rec in safe if not rec.feedback_supported]
            if unsupported and not allow_unsupported_feedback:
                raise ValueError(
                    "recommend_and_log refuses items without local feedback support; "
                    "register the items or set allow_unsupported_feedback=True for an external feedback path"
                )

            epsilon = float(exploration)
            probabilities = np.full(len(safe), epsilon / len(safe), dtype=float)
            probabilities[0] += 1.0 - epsilon
            was_exploration = bool(epsilon > 0.0 and len(safe) > 1 and self._rng.random() < epsilon)
            chosen_index = int(self._rng.integers(len(safe))) if was_exploration else 0
            chosen = safe[chosen_index]
            served = [chosen, *(rec for idx, rec in enumerate(safe) if idx != chosen_index)]
            served = served[: min(int(top_k), len(served))]
            decision = LoggedDecision(
                user_id=user_id,
                timestamp=normalized_timestamp,
                candidate_item_ids=tuple(rec.item_id for rec in safe),
                chosen_item_id=chosen.item_id,
                propensity=float(probabilities[chosen_index]),
                policy_name=self._resolved_policy_name(),
                policy_version=resolved_policy_version,
                scores=tuple(float(rec.score) for rec in safe),
                context_hash=ctx,
                decision_id=resolved_decision_id or uuid.uuid4().hex,
                action_probabilities=tuple(float(value) for value in probabilities),
                predicted_outcomes=tuple(float(rec.outcome_probability) for rec in safe),
                exploration_rate=epsilon,
                was_exploration=was_exploration,
                exploration_bonus=tuple([epsilon / len(safe)] * len(safe)),
                policy_metadata={
                    "concept_goal": concept_goal,
                    "min_item_support": float(min_item_support),
                    "min_outcome_probability": float(min_outcome_probability),
                    "max_outcome_probability": float(max_outcome_probability),
                    "min_difficulty": min_difficulty,
                    "max_difficulty": max_difficulty,
                    "require_prerequisites": bool(require_prerequisites),
                    "feedback_supported": all(rec.feedback_supported for rec in safe),
                    "base_scores": tuple(base_scores[rec.item_id] for rec in safe),
                    "external_feedback_required": bool(unsupported),
                    "_orchid_request_fingerprint": request_fingerprint,
                    "decision_metadata": resolved_decision_metadata,
                },
            )
            stored_decision, created = self.decision_store.create_decision(decision)
            if not created:
                _require_matching_decision_request(stored_decision, request_fingerprint)
                return self._recommendations_from_logged_decision(stored_decision, top_k=top_k), stored_decision
            return served, stored_decision

    def observe(
        self,
        *,
        user_id: Any,
        item_id: Any,
        outcome: Any,
        timestamp: Any,
        category_id: Optional[Any] = None,
        update_global: bool = True,
    ) -> Any:
        """Observe one user outcome and update live state.

        Set ``update_global=False`` for a frozen experiment: the learner's
        own adaptive state advances while aggregate item/global statistics
        remain unchanged.  This is useful when a treatment artifact must not
        learn from concurrent control or pilot traffic.
        """
        with self._state_lock:
            self._require_fitted()
            assert self.recommender_ is not None
            if not isinstance(update_global, (bool, np.bool_)):
                raise TypeError("update_global must be boolean")
            normalized_outcome = _binary_outcome(outcome)
            normalized_timestamp = normalize_timestamp(timestamp)
            del category_id
            if bool(update_global):
                return self.recommender_.observe(
                    user_id,
                    item_id,
                    normalized_outcome,
                    timestamp=normalized_timestamp,
                )
            return _observe_recommender_locally(
                self.recommender_,
                user_id=user_id,
                item_id=item_id,
                outcome=normalized_outcome,
                timestamp=normalized_timestamp,
            )

    def persist_decision_outcome(
        self,
        decision_id: str,
        *,
        outcome: Optional[Any] = None,
        reward: Optional[float] = None,
        timestamp: Optional[Any] = None,
        category_id: Optional[Any] = None,
        outcome_event_id: Optional[str] = None,
        apply_state: bool = True,
        update_global: bool = True,
    ) -> DecisionOutcome:
        """Attach one delayed outcome without changing live learner state.

        This is the persistence half of the durable outcome protocol. Most
        integrations should use :meth:`observe_decision`; the method is
        public so a lifecycle adapter can persist all score evidence before
        rebuilding a fresh ranker state projection.
        """
        self._require_fitted()
        with self._decision_lock:
            decision = self.decision_store.get_decision(decision_id)
            if decision is None:
                raise KeyError(f"unknown decision_id: {decision_id}")
            if outcome is None and reward is None:
                raise ValueError("observe_decision requires outcome or reward")
            if outcome_event_id is not None and not str(outcome_event_id):
                raise ValueError("outcome_event_id must be non-empty when supplied")
            if not isinstance(apply_state, (bool, np.bool_)):
                raise TypeError("apply_state must be boolean")
            if not isinstance(update_global, (bool, np.bool_)):
                raise TypeError("update_global must be boolean")
            normalized_outcome = None if outcome is None else _binary_outcome(outcome)
            if reward is not None and not np.isfinite(float(reward)):
                raise ValueError("reward must be finite")

            outcome_timestamp = decision.timestamp if timestamp is None else normalize_timestamp(timestamp)
            if outcome_timestamp < decision.timestamp:
                raise ValueError("timestamp must not precede the serving decision")
            if reward is not None:
                resolved_reward = float(reward)
            else:
                assert normalized_outcome is not None
                resolved_reward = float(normalized_outcome)
            resolved_category = category_id
            if resolved_category is None and self.recommender_ is not None:
                resolved_category = self.recommender_.concept_by_item_.get(decision.chosen_item_id)
            linked_outcome = DecisionOutcome(
                decision_id=decision_id,
                user_id=decision.user_id,
                item_id=decision.chosen_item_id,
                outcome_timestamp=outcome_timestamp,
                outcome=normalized_outcome,
                reward=resolved_reward,
                category_id=resolved_category,
                outcome_event_id=None if outcome_event_id is None else str(outcome_event_id),
                apply_state=bool(apply_state),
                update_global=bool(update_global),
            )
            validate_decision_outcomes(pd.DataFrame([linked_outcome.to_dict()]))
            stored_outcome, _created = self.decision_store.attach_outcome(linked_outcome)
            return stored_outcome

    def observe_decision(
        self,
        decision_id: str,
        *,
        outcome: Optional[Any] = None,
        reward: Optional[float] = None,
        timestamp: Optional[Any] = None,
        category_id: Optional[Any] = None,
        outcome_event_id: Optional[str] = None,
        apply_state: bool = True,
        update_global: bool = True,
    ) -> DecisionOutcome:
        """Attach one delayed outcome and apply it to live learner state.

        Outcome evidence is durably attached before it is applied. If the
        application raises, the outcome remains pending and can be repaired by
        retrying this exact call or by :meth:`replay_pending_outcomes`.
        """
        stored_outcome = self.persist_decision_outcome(
            decision_id,
            outcome=outcome,
            reward=reward,
            timestamp=timestamp,
            category_id=category_id,
            outcome_event_id=outcome_event_id,
            apply_state=apply_state,
            update_global=update_global,
        )
        with self._decision_lock:
            self._apply_stored_outcome(stored_outcome)
        return stored_outcome

    def replay_pending_outcomes(self) -> list[DecisionOutcome]:
        """Apply durable outcomes that were stored before live-state application.

        Call this after restoring a ranker from the same baseline/model-state
        checkpoint following a crash.  Applied outcomes are durably marked, so
        repeated recovery calls are no-ops.  Outcomes are replayed in
        chronological order within each learner.
        """
        self._require_fitted()
        with self._decision_lock:
            pending = self.decision_store.pending_outcomes()
            pending.sort(key=lambda value: (str(value.user_id), value.outcome_timestamp, value.decision_id))
            for stored_outcome in pending:
                self._apply_stored_outcome(stored_outcome)
            return pending

    def replay_all_outcomes_from_baseline(self) -> list[DecisionOutcome]:
        """Reapply all stateful durable outcomes to a fresh model baseline.

        The store's application checkpoints describe a previous in-memory
        projection, so they are deliberately ignored here. Only call this on
        a newly restored/fitted baseline; calling it on a live projection
        would apply the same outcomes twice.
        """
        self._require_fitted()
        with self._decision_lock:
            outcomes = self.decision_store.outcomes()
            outcomes.sort(key=lambda value: (str(value.user_id), value.outcome_timestamp, value.decision_id))
            for stored_outcome in outcomes:
                if stored_outcome.apply_state and stored_outcome.outcome is not None:
                    self.observe(
                        user_id=stored_outcome.user_id,
                        timestamp=stored_outcome.outcome_timestamp,
                        item_id=stored_outcome.item_id,
                        category_id=stored_outcome.category_id,
                        outcome=stored_outcome.outcome,
                        update_global=stored_outcome.update_global,
                    )
                self.decision_store.mark_outcome_applied(stored_outcome.decision_id)
            return outcomes

    def _apply_stored_outcome(self, stored_outcome: DecisionOutcome) -> bool:
        """Apply one pending durable outcome, then checkpoint its application."""
        if self.decision_store.is_outcome_applied(stored_outcome.decision_id):
            return False
        if stored_outcome.apply_state and stored_outcome.outcome is not None:
            self.observe(
                user_id=stored_outcome.user_id,
                timestamp=stored_outcome.outcome_timestamp,
                item_id=stored_outcome.item_id,
                category_id=stored_outcome.category_id,
                outcome=stored_outcome.outcome,
                update_global=stored_outcome.update_global,
            )
        self.decision_store.mark_outcome_applied(stored_outcome.decision_id)
        return True

    def decision_log_frame(self, *, completed_only: bool = False) -> pd.DataFrame:
        """Return immutable serving decisions, optionally joined to delayed outcomes."""
        with self._decision_lock:
            decisions = pd.DataFrame([decision.to_dict() for decision in self.decision_store.decisions()])
            if decisions.empty:
                return decisions
            stored_outcomes = self.decision_store.outcomes()
            if not stored_outcomes:
                return decisions.iloc[0:0].copy() if completed_only else decisions.copy()
            outcomes = pd.DataFrame([outcome.to_dict() for outcome in stored_outcomes]).rename(
                columns={"reward": "__outcome_reward__"}
            )
            joined = decisions.merge(
                outcomes[
                    [
                        "decision_id",
                        "outcome_timestamp",
                        "outcome",
                        "__outcome_reward__",
                        "category_id",
                        "outcome_event_id",
                        "apply_state",
                        "update_global",
                    ]
                ],
                on="decision_id",
                how="left",
                validate="one_to_one",
            )
            joined["reward"] = [
                outcome_reward if pd.notna(outcome_reward) else logged_reward
                for outcome_reward, logged_reward in zip(
                    joined["__outcome_reward__"],
                    joined["reward"],
                )
            ]
            joined = joined.drop(columns=["__outcome_reward__"])
            if completed_only:
                joined = joined[joined["outcome_timestamp"].notna() & joined["reward"].notna()]
            return joined.reset_index(drop=True)

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
        work = validate_logged_decisions(
            logged_decisions,
            reward_col=reward_col,
            propensity_col=propensity_col,
        ).copy()
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

    def bootstrap_ope_report(
        self,
        logged_decisions: pd.DataFrame,
        *,
        reward_col: str = "reward",
        propensity_col: str = "propensity",
        target_probability_col: str = "target_probability",
        max_weight: Optional[float] = None,
        n_bootstrap: int = 300,
        cluster_col: str = "user_id",
    ) -> BootstrapLoggedPolicyReport:
        """Evaluate a fitted policy with user-cluster bootstrap intervals."""
        work = validate_logged_decisions(
            logged_decisions,
            reward_col=reward_col,
            propensity_col=propensity_col,
        ).copy()
        if target_probability_col not in work.columns:
            if self.offline_policy_ is None:
                raise RuntimeError("fit_policy or target_probability_col is required before bootstrap OPE")
            work[target_probability_col] = [self._target_probability(row) for _, row in work.iterrows()]
        return bootstrap_logged_policy(
            work,
            reward_col=reward_col,
            propensity_col=propensity_col,
            target_probability_col=target_probability_col,
            max_weight=max_weight,
            n_bootstrap=n_bootstrap,
            random_state=self.config.random_state,
            cluster_col=cluster_col,
        )

    def shadow_report(
        self,
        *,
        cluster_bootstrap_samples: int = 300,
        max_weight: Optional[float] = None,
    ) -> ShadowDeploymentReport:
        """Summarize shadow traffic, linked outcomes, drift, calibration, and OPE."""
        if cluster_bootstrap_samples < 0:
            raise ValueError("cluster_bootstrap_samples must be non-negative")
        decisions = self.decision_log_frame(completed_only=False)
        if decisions.empty:
            raise ValueError("shadow_report requires at least one recommend_and_log decision")
        completed = self.decision_log_frame(completed_only=True)
        if not completed.empty:
            completed = completed.sort_values(["outcome_timestamp", "decision_id"], kind="mergesort")
        candidate_counts = decisions["candidate_item_ids"].map(lambda value: len(parse_candidate_list(value)))
        propensities = decisions["propensity"].astype(float)
        score_regrets = [_decision_score_regret(row) for _, row in completed.iterrows()]
        predicted, observed = _chosen_predictions_and_outcomes(completed)
        calibration_errors = predicted - observed if predicted.size else np.asarray([], dtype=float)
        reward_drift = _half_window_shift(completed, "reward")
        calibration_drift = _half_window_calibration_shift(predicted, observed)

        bootstrap: Optional[BootstrapPolicyComparisonReport] = None
        gate: Optional[OPERolloutGateReport] = None
        if self.offline_policy_ is not None and not completed.empty and cluster_bootstrap_samples > 0:
            evidence = self._evaluate_candidate_policy(
                self.offline_policy_,
                completed,
                max_weight=max_weight,
                reward_col="reward",
                cluster_bootstrap_samples=cluster_bootstrap_samples,
                cluster_col="user_id",
            )
            if not isinstance(evidence, BootstrapPolicyComparisonReport):
                raise RuntimeError("cluster bootstrap policy evaluation did not return bootstrap evidence")
            bootstrap = evidence
            gate = evaluate_rollout_gate(
                bootstrap,
                min_effect=self.config.offline_policy_min_effect,
                min_ess_fraction=self.config.offline_policy_min_ess_fraction,
                min_coverage=self.config.offline_policy_min_coverage,
                max_clipped_fraction=self.config.offline_policy_max_clipped_fraction,
            )
        rewards = completed["reward"].astype(float) if not completed.empty else pd.Series(dtype=float)
        outcomes = completed["outcome"].dropna().astype(float) if "outcome" in completed.columns else pd.Series(dtype=float)
        return ShadowDeploymentReport(
            n_decisions=int(len(decisions)),
            n_outcomes=int(len(completed)),
            outcome_coverage=float(len(completed) / len(decisions)),
            unique_users=int(decisions["user_id"].nunique()),
            unique_items_chosen=int(decisions["chosen_item_id"].nunique()),
            candidate_count_mean=float(candidate_counts.mean()),
            exploration_fraction=float(decisions["was_exploration"].astype(bool).mean()),
            propensity_mean=float(propensities.mean()),
            propensity_min=float(propensities.min()),
            reward_mean=float(rewards.mean()) if not rewards.empty else None,
            outcome_mean=float(outcomes.mean()) if not outcomes.empty else None,
            score_regret_mean=float(np.mean(score_regrets)) if score_regrets else None,
            calibration_brier=float(np.mean(calibration_errors**2)) if calibration_errors.size else None,
            calibration_bias=float(np.mean(calibration_errors)) if calibration_errors.size else None,
            reward_drift=reward_drift,
            calibration_drift=calibration_drift,
            policy_versions=tuple(sorted(decisions["policy_version"].astype(str).unique().tolist())),
            bootstrap_ope=bootstrap,
            rollout_gate=gate,
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
            "offline_policy_gate": None if self.offline_policy_gate_ is None else self.offline_policy_gate_.to_dict(),
            "offline_policy_bootstrap": (
                None if self.offline_policy_bootstrap_ is None else self.offline_policy_bootstrap_.to_dict()
            ),
            "last_policy_gate": None if self.last_policy_gate_ is None else self.last_policy_gate_.to_dict(),
            "last_policy_evidence": (
                None if self.last_policy_evidence_ is None else self.last_policy_evidence_.to_dict()
            ),
            "rolling_policy_update": (
                None if self.rolling_policy_report_ is None else self.rolling_policy_report_.to_dict()
            ),
            "logged_decisions": len(self.decision_store.decisions()),
            "linked_outcomes": len(self.decision_store.outcomes()),
            "has_sketch_generator": self.sketch_generator_ is not None,
            "semantic_encoder": None if self.semantic_encoder_ is None else self.semantic_encoder_.diagnostics(),
        }
        return data

    def learning_readiness(self) -> dict[str, Any]:
        """Return the data-readiness assessment for the fitted learning policy.

        Small pilots automatically use Orchid's empirical adaptive baseline by
        default. The report explains what support is still needed before a
        knowledge-tracing model is selected.
        """
        self._require_fitted()
        assert self.recommender_ is not None
        return self.recommender_.readiness_report().to_dict()

    def _recommendations_from_logged_decision(
        self,
        decision: LoggedDecision,
        *,
        top_k: int,
    ) -> list[AdaptiveLearningRecommendation]:
        """Reconstruct a stable serving response for an idempotent retry.

        Scores and correctness predictions come from the immutable decision,
        not from the mutable current learner state.  Current catalog metadata
        fills the descriptive fields because it does not alter the action that
        was already logged.
        """
        assert self.recommender_ is not None
        candidates = list(decision.candidate_item_ids)
        predictions = list(decision.predicted_outcomes or ())
        if len(predictions) != len(candidates):
            predictions = [0.5] * len(candidates)
        metadata = decision.policy_metadata or {}
        external_feedback_required = bool(metadata.get("external_feedback_required", False))
        recommendation_by_item: dict[Any, AdaptiveLearningRecommendation] = {}
        for item_id, score, outcome_probability in zip(candidates, decision.scores, predictions):
            category_id = self.recommender_.concept_by_item_.get(item_id)
            recommendation_by_item[item_id] = AdaptiveLearningRecommendation(
                item_id=item_id,
                score=float(score),
                outcome_probability=float(outcome_probability),
                policy=decision.policy_name,
                difficulty=self.recommender_.difficulty_by_item_.get(item_id),
                category_id=category_id,
                expected_reward=float(score),
                model_prediction=float(outcome_probability),
                item_support=float(self.recommender_.item_support_.get(item_id, 0.0)),
                concept_support=float(self.recommender_.concept_support_.get(category_id, 0.0)),
                feedback_supported=not external_feedback_required
                and self.recommender_.feedback_supported(item_id),
            )
        ordered = [
            recommendation_by_item[decision.chosen_item_id],
            *(recommendation_by_item[item_id] for item_id in candidates if item_id != decision.chosen_item_id),
        ]
        return ordered[: min(int(top_k), len(ordered))]

    @staticmethod
    def _is_safe_exploration_candidate(
        rec: AdaptiveLearningRecommendation,
        *,
        min_item_support: float,
        min_outcome_probability: float,
        max_outcome_probability: float,
        min_difficulty: Optional[float],
        max_difficulty: Optional[float],
        require_prerequisites: bool,
    ) -> bool:
        if require_prerequisites and not rec.prerequisites_met:
            return False
        if float(rec.item_support) < float(min_item_support):
            return False
        if not (
            float(min_outcome_probability)
            <= float(rec.outcome_probability)
            <= float(max_outcome_probability)
        ):
            return False
        if min_difficulty is not None:
            if rec.difficulty is None or float(rec.difficulty) < float(min_difficulty):
                return False
        if max_difficulty is not None:
            if rec.difficulty is None or float(rec.difficulty) > float(max_difficulty):
                return False
        return True

    def _semantic_cold_start_recommendations(
        self,
        user_id: Any,
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
            concept = _first_metadata_value(metadata, ("category_id",))
            difficulty = _optional_float(_first_metadata_value(metadata, ("difficulty", "item_difficulty", "difficulty_score")))
            prerequisites_met = self._cold_start_prerequisites_met(user_id, concept)
            if (
                not prerequisites_met
                and self.recommender_.config.enforce_prerequisites
                and not self.recommender_.config.allow_prerequisite_fallback
            ):
                continue
            competence = self.recommender_.competence_for(user_id, concept) if concept is not None else None
            outcome_probability = _cold_start_outcome_prior(competence=competence, difficulty=difficulty)
            normalizer = max(self.config.target_outcome, 1.0 - self.config.target_outcome, 1e-6)
            stretch_fit = max(0.0, 1.0 - abs(outcome_probability - self.config.target_outcome) / normalizer)
            uncertainty = max(0.0, 1.0 - 2.0 * abs(outcome_probability - 0.5))
            structure_score = 0.7 * stretch_fit + 0.3 * uncertainty
            score = semantic_weight * _clamp01(semantic_scores[item_id]) + structure_weight * structure_score
            recs.append(
                AdaptiveLearningRecommendation(
                    item_id=item_id,
                    score=float(score),
                    outcome_probability=float(outcome_probability),
                    policy="semantic_cold_start",
                    difficulty=difficulty,
                    category_id=concept,
                    competence=competence,
                    expected_reward=float(score),
                    stretch_fit=float(stretch_fit),
                    uncertainty=float(uncertainty),
                    support_penalty=1.0,
                    item_support=0.0,
                    concept_support=0.0,
                    prerequisites_met=prerequisites_met,
                    feedback_supported=self.recommender_.feedback_supported(item_id),
                )
            )
        recs.sort(key=lambda rec: (rec.score, str(rec.item_id)), reverse=True)
        return recs[: min(int(top_k), len(recs))]

    def _cold_start_prerequisites_met(self, user_id: Any, concept: Any) -> bool:
        if self.recommender_ is None or concept is None:
            return True
        requirements = self.recommender_.prerequisite_by_concept_.get(concept, set())
        if not requirements:
            return True
        return set(requirements).issubset(self.recommender_.mastered_concepts(user_id))

    def _adaptive_config(self, *, policy: str) -> AdaptiveLearningConfig:
        return AdaptiveLearningConfig(
            tracer_model=self.config.kt_backbone,
            policy=policy,
            target_correct=self.config.target_outcome,
            max_seq_len=self.config.max_seq_len,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            correct_threshold=self.config.outcome_threshold,
            delayed_gain_window=self.config.delayed_gain_window,
            delayed_gain_shrinkage=self.config.delayed_gain_shrinkage,
            reward_model_max_examples=self.config.reward_model_max_examples,
            reward_model_example_weighting=self.config.reward_model_example_weighting,
            reward_model_cross_fit_folds=self.config.reward_model_cross_fit_folds,
            reward_model_max_sample_weight=self.config.reward_model_max_sample_weight,
            progression_competence_blend=self.config.progression_competence_blend,
            hybrid_progression_weight=self.config.hybrid_progression_weight,
            hybrid_item_prior_weight=self.config.hybrid_item_prior_weight,
            hybrid_concept_prior_weight=self.config.hybrid_concept_prior_weight,
            hybrid_kt_weight=self.config.hybrid_kt_weight,
            hybrid_support_weight=self.config.hybrid_support_weight,
            hybrid_unsupported_penalty_weight=self.config.hybrid_unsupported_penalty_weight,
            hybrid_prior_smoothing=self.config.hybrid_prior_smoothing,
            hybrid_concept_smoothing=self.config.hybrid_concept_smoothing,
            hybrid_min_item_support=self.config.hybrid_min_item_support,
            hybrid_min_concept_support=self.config.hybrid_min_concept_support,
            mastery_threshold=self.config.mastery_threshold,
            enforce_prerequisites=self.config.enforce_prerequisites,
            allow_prerequisite_fallback=self.config.allow_prerequisite_fallback,
            fallback_to_empirical=self.config.fallback_to_empirical,
            min_kt_events=self.config.min_kt_events,
            min_kt_users=self.config.min_kt_users,
            min_kt_items=self.config.min_kt_items,
            min_kt_median_events_per_user=self.config.min_kt_median_events_per_user,
            device=self.config.device,
            random_state=self.config.random_state,
        )

    def _offline_policy_allowed(self) -> bool:
        return bool(self.offline_policy_gate_ is not None and self.offline_policy_gate_.allowed)

    def _offline_policy_serving_weight(self) -> float:
        weight = max(0.0, float(self.config.offline_policy_weight))
        cap = max(0.0, float(self.config.offline_policy_max_weight))
        return min(weight, cap)

    def _serving_offline_scores(self, scores: dict[Any, float]) -> dict[Any, float]:
        if not self.config.offline_policy_normalize or not scores:
            return {item_id: float(value) for item_id, value in scores.items()}
        values = [float(value) for value in scores.values()]
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            return {item_id: 0.0 for item_id in scores}
        return {item_id: (float(value) - lo) / (hi - lo) for item_id, value in scores.items()}

    def _blended_scores(
        self,
        policy: CQLDiscretePolicy,
        *,
        context_hash: Any,
        candidate_item_ids: Sequence[Any],
        base_scores: Sequence[float],
    ) -> dict[Any, float]:
        """Score the exact adaptive-base plus CQL action rule used in serving."""
        candidates = list(candidate_item_ids)
        if len(base_scores) != len(candidates):
            raise ValueError("base_scores length must match candidate_item_ids")
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate_item_ids must not contain duplicates")
        q_scores = self._serving_offline_scores(policy.score(context_hash, candidates))
        weight = self._offline_policy_serving_weight()
        return {
            item_id: float(float(base_score) + weight * q_scores.get(item_id, 0.0))
            for item_id, base_score in zip(candidates, base_scores)
        }

    def _apply_offline_overlay(
        self,
        base_recommendations: Sequence[AdaptiveLearningRecommendation],
        *,
        context_hash: Any,
    ) -> list[AdaptiveLearningRecommendation]:
        """Apply the promoted CQL overlay through the same function used by OPE."""
        recs = list(base_recommendations)
        if self.offline_policy_ is None or not self._offline_policy_allowed() or not recs:
            return sorted(recs, key=lambda rec: (rec.score, str(rec.item_id)), reverse=True)
        scores = self._blended_scores(
            self.offline_policy_,
            context_hash=context_hash,
            candidate_item_ids=[rec.item_id for rec in recs],
            base_scores=[rec.score for rec in recs],
        )
        blended = [replace(rec, score=scores[rec.item_id]) for rec in recs]
        return sorted(blended, key=lambda rec: (rec.score, str(rec.item_id)), reverse=True)

    def _composite_recommend(
        self,
        policy: CQLDiscretePolicy,
        *,
        context_hash: Any,
        candidate_item_ids: Sequence[Any],
        base_scores: Sequence[float],
    ) -> list[Any]:
        """Return the deterministic action order of a prospective hybrid+CQL deployment."""
        scores = self._blended_scores(
            policy,
            context_hash=context_hash,
            candidate_item_ids=candidate_item_ids,
            base_scores=base_scores,
        )
        return sorted(scores, key=lambda item_id: (scores[item_id], str(item_id)), reverse=True)

    def _target_probability(self, row: pd.Series) -> float:
        assert self.offline_policy_ is not None
        candidates = parse_candidate_list(row["candidate_item_ids"])
        chosen = row["chosen_item_id"]
        selected = self._composite_recommend(
            self.offline_policy_,
            context_hash=row["context_hash"],
            candidate_item_ids=candidates,
            base_scores=_base_scores_for_logged_row(row),
        )[:1]
        return float(bool(selected and selected[0] == chosen))

    def _evaluate_candidate_policy(
        self,
        candidate_policy: CQLDiscretePolicy,
        evaluation: pd.DataFrame,
        *,
        reward_col: str,
        cluster_bootstrap_samples: int,
        cluster_col: str,
        max_weight: Optional[float] = None,
    ) -> PolicyComparisonReport | BootstrapPolicyComparisonReport:
        """Compare a local candidate with the explicit logging-policy baseline."""
        work = evaluation.copy()
        candidate_probability_col = "__orchid_candidate_probability__"
        baseline_probability_col = "__orchid_logging_probability__"
        def candidate_probability(row: pd.Series) -> float:
            candidates = parse_candidate_list(row["candidate_item_ids"])
            selected = self._composite_recommend(
                candidate_policy,
                context_hash=row["context_hash"],
                candidate_item_ids=candidates,
                base_scores=_base_scores_for_logged_row(row),
            )[:1]
            return float(bool(selected and selected[0] == row["chosen_item_id"]))

        work[candidate_probability_col] = [candidate_probability(row) for _, row in work.iterrows()]
        # The logging-policy probability of the action actually taken is the
        # recorded propensity. This supplies an explicit, evaluable incumbent
        # baseline with value equal to the logged reward mean.
        work[baseline_probability_col] = work["propensity"].astype(float)
        if cluster_bootstrap_samples > 0:
            return bootstrap_compare_logged_policies(
                work,
                reward_col=reward_col,
                propensity_col="propensity",
                target_probability_col=candidate_probability_col,
                baseline_probability_col=baseline_probability_col,
                max_weight=max_weight,
                n_bootstrap=cluster_bootstrap_samples,
                random_state=self.config.random_state,
                cluster_col=cluster_col,
            )
        return compare_logged_policies(
            work,
            reward_col=reward_col,
            propensity_col="propensity",
            target_probability_col=candidate_probability_col,
            baseline_probability_col=baseline_probability_col,
            max_weight=max_weight,
        )

    def _resolved_policy_name(self) -> str:
        base = (
            str(self.recommender_.policy_name_)
            if self.recommender_ is not None and self.recommender_.policy_name_ is not None
            else str(self.config.policy)
        )
        return f"{base}+cql" if self.offline_policy_ is not None and self._offline_policy_allowed() else base

    def _derive_deployment_version(self) -> str:
        """Create a deployment fingerprint from learned state, not input tables."""
        digest = hashlib.sha256()
        digest.update(repr(self.config).encode("utf-8"))
        digest.update(self._resolved_policy_name().encode("utf-8"))
        if self.recommender_ is not None:
            digest.update(repr(self.recommender_.config).encode("utf-8"))
            digest.update(str(self.recommender_.policy_name_).encode("utf-8"))
            _update_digest_with_mapping(
                digest,
                {
                    "difficulty_by_item": self.recommender_.difficulty_by_item_,
                    "concept_by_item": self.recommender_.concept_by_item_,
                    "item_support": self.recommender_.item_support_,
                    "item_correct": self.recommender_.item_correct_,
                    "concept_support": self.recommender_.concept_support_,
                    "concept_correct": self.recommender_.concept_correct_,
                    "global_correct": self.recommender_.global_correct_,
                },
            )
            model = getattr(self.recommender_.tracer_, "model", None)
            if model is not None:
                for name, tensor in sorted(model.state_dict().items()):
                    values = tensor.detach().cpu().contiguous().numpy()
                    digest.update(name.encode("utf-8"))
                    digest.update(str(values.dtype).encode("utf-8"))
                    digest.update(repr(values.shape).encode("utf-8"))
                    digest.update(values.tobytes())
        if self.offline_policy_ is not None:
            digest.update(self.offline_policy_.state_fingerprint().encode("utf-8"))
        return f"orchid-{self._resolved_policy_name()}-{digest.hexdigest()[:12]}"

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("AdaptiveRanker.fit must be called before serving")


def _require_decision_store(store: Any) -> None:
    """Fail early when a custom decision store misses the persistence contract."""
    required = (
        "get_decision",
        "get_outcome",
        "get_outcome_by_event_id",
        "create_decision",
        "attach_outcome",
        "is_outcome_applied",
        "mark_outcome_applied",
        "pending_outcomes",
        "decisions",
        "outcomes",
    )
    missing = [name for name in required if not callable(getattr(store, name, None))]
    if missing:
        raise TypeError(f"decision_store is missing required methods: {missing}")


def _observe_recommender_locally(
    recommender: AdaptiveLearningRecommender,
    *,
    user_id: Any,
    item_id: Any,
    outcome: int,
    timestamp: float,
) -> Any:
    """Advance learner-specific state while restoring aggregate counters.

    Tracer-based policies keep a learner history in their tracer, whereas the
    recommender also maintains aggregate item/concept counters used for future
    fitting and diagnostics.  Frozen pilots need the former but must retain the
    latter exactly.  ``EmpiricalTracer`` additionally owns aggregate
    global/item counts, so those are restored while its user and user-item
    counts remain advanced.
    """
    aggregate_mappings = (
        "item_support_",
        "item_correct_",
        "concept_support_",
        "concept_correct_",
    )
    mapping_snapshot = {
        name: dict(getattr(recommender, name))
        for name in aggregate_mappings
        if isinstance(getattr(recommender, name, None), dict)
    }
    scalar_names = ("_global_correct_total", "_global_outcome_count", "global_correct_")
    scalar_snapshot = {
        name: getattr(recommender, name)
        for name in scalar_names
        if hasattr(recommender, name)
    }
    tracer = recommender.tracer_
    empirical_mapping_names = ("_item_successes", "_item_count")
    empirical_mapping_snapshot = {
        name: dict(getattr(tracer, name))
        for name in empirical_mapping_names
        if isinstance(getattr(tracer, name, None), dict)
    }
    empirical_scalar_names = ("_global_successes", "_global_count")
    empirical_scalar_snapshot = {
        name: getattr(tracer, name)
        for name in empirical_scalar_names
        if hasattr(tracer, name)
    }
    try:
        return recommender.observe(user_id, item_id, outcome, timestamp=timestamp)
    finally:
        for name, snapshot in mapping_snapshot.items():
            target = getattr(recommender, name)
            target.clear()
            target.update(snapshot)
        for name, snapshot in scalar_snapshot.items():
            setattr(recommender, name, snapshot)
        for name, snapshot in empirical_mapping_snapshot.items():
            target = getattr(tracer, name)
            target.clear()
            target.update(snapshot)
        for name, snapshot in empirical_scalar_snapshot.items():
            setattr(tracer, name, snapshot)


def _decision_request_fingerprint(
    *,
    user_id: Any,
    timestamp: float,
    candidate_item_ids: Sequence[Any],
    exploration: float,
    min_item_support: float,
    min_outcome_probability: float,
    max_outcome_probability: float,
    min_difficulty: Optional[float],
    max_difficulty: Optional[float],
    require_prerequisites: bool,
    allow_unsupported_feedback: bool,
    policy_version: Optional[str],
    context_hash: str,
    concept_goal: Optional[Any],
    decision_metadata: Optional[Mapping[str, Any]],
) -> str:
    """Hash all inputs that determine a logged serving decision."""
    payload = _fingerprint_value(
        {
            "user_id": user_id,
            "timestamp": timestamp,
            "candidate_item_ids": list(candidate_item_ids),
            "exploration": float(exploration),
            "min_item_support": float(min_item_support),
            "min_outcome_probability": float(min_outcome_probability),
            "max_outcome_probability": float(max_outcome_probability),
            "min_difficulty": min_difficulty,
            "max_difficulty": max_difficulty,
            "require_prerequisites": bool(require_prerequisites),
            "allow_unsupported_feedback": bool(allow_unsupported_feedback),
            "policy_version": policy_version,
            "context_hash": context_hash,
            "concept_goal": concept_goal,
            "decision_metadata": decision_metadata,
        }
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_decision_metadata(value: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """Return a deeply JSON-compatible immutable-decision metadata payload."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("decision_metadata must be a mapping")
    return _json_metadata_mapping(value)


def _json_metadata_value(value: Any) -> Any:
    """Validate the portable metadata shape used by durable decision stores."""
    if isinstance(value, np.generic):
        return _json_metadata_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("decision_metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return _json_metadata_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_metadata_value(item) for item in value]
    raise TypeError("decision_metadata values must be JSON-compatible")


def _json_metadata_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("decision_metadata mapping keys must be strings")
        result[key] = _json_metadata_value(item)
    return result


def _require_matching_decision_request(decision: LoggedDecision, request_fingerprint: str) -> None:
    """Reject accidental reuse of an idempotency key for a different request."""
    metadata = decision.policy_metadata or {}
    stored_fingerprint = metadata.get("_orchid_request_fingerprint")
    if stored_fingerprint != request_fingerprint:
        raise ValueError(f"decision_id already exists for a different serving request: {decision.decision_id}")


def _prepare_learning_catalog(
    events: pd.DataFrame,
    *,
    item_col: str,
    category_col: Optional[str],
    difficulty_col: Optional[str],
    catalog: Optional[pd.DataFrame],
    catalog_item_col: str,
    catalog_category_col: str,
    catalog_difficulty_col: str,
) -> tuple[pd.DataFrame, Optional[str], Optional[str], Optional[pd.DataFrame]]:
    """Attach authoritative catalog metadata to historical learning attempts.

    Event-level category/difficulty columns remain supported for compatibility.
    When absent, a catalog becomes the canonical source for those values and
    also registers exercises that have not yet received learner feedback.
    """
    if catalog is None:
        return events, category_col, difficulty_col, None
    if catalog_item_col not in catalog.columns:
        raise ValueError(f"catalog must include item column {catalog_item_col!r}")
    prepared_catalog = catalog.copy()
    if prepared_catalog[catalog_item_col].isna().any():
        raise ValueError("catalog item identifiers must not be missing")
    if prepared_catalog[catalog_item_col].duplicated().any():
        raise ValueError("catalog must contain one canonical row per item")
    catalog_ids = set(prepared_catalog[catalog_item_col].tolist())
    missing_items = [item_id for item_id in events[item_col].drop_duplicates().tolist() if item_id not in catalog_ids]
    if missing_items:
        preview = ", ".join(repr(item_id) for item_id in missing_items[:5])
        raise ValueError(f"catalog is missing historical item IDs: {preview}")

    prepared_events = events.copy()
    lookup = prepared_catalog.set_index(catalog_item_col)
    resolved_category_col = category_col
    if resolved_category_col is None and catalog_category_col in prepared_catalog.columns:
        generated_category_col = _generated_catalog_column(prepared_events, "__orchid_catalog_category__")
        categories = prepared_events[item_col].map(lookup[catalog_category_col])
        if categories.isna().any():
            missing = prepared_events.loc[categories.isna(), item_col].drop_duplicates().tolist()
            preview = ", ".join(repr(item_id) for item_id in missing[:5])
            raise ValueError(f"catalog category metadata is missing for historical item IDs: {preview}")
        prepared_events[generated_category_col] = categories
        resolved_category_col = generated_category_col

    resolved_difficulty_col = difficulty_col
    if resolved_difficulty_col is None and catalog_difficulty_col in prepared_catalog.columns:
        generated_difficulty_col = _generated_catalog_column(prepared_events, "__orchid_catalog_difficulty__")
        difficulties = prepared_events[item_col].map(lookup[catalog_difficulty_col])
        if difficulties.isna().any():
            missing = prepared_events.loc[difficulties.isna(), item_col].drop_duplicates().tolist()
            preview = ", ".join(repr(item_id) for item_id in missing[:5])
            raise ValueError(f"catalog difficulty metadata is missing for historical item IDs: {preview}")
        prepared_events[generated_difficulty_col] = difficulties
        resolved_difficulty_col = generated_difficulty_col
    return prepared_events, resolved_category_col, resolved_difficulty_col, prepared_catalog


def _generated_catalog_column(events: pd.DataFrame, base: str) -> str:
    """Choose an internal metadata column that cannot overwrite caller data."""
    candidate = base
    suffix = 1
    while candidate in events.columns:
        candidate = f"{base}{suffix}"
        suffix += 1
    return candidate


def _first_metadata_value(metadata: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _binary_outcome(value: Any) -> int:
    """Validate one live binary outcome without integer truncation."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome must be exactly 0 or 1") from exc
    if not np.isfinite(numeric) or numeric not in {0.0, 1.0}:
        raise ValueError("outcome must be exactly 0 or 1")
    return int(numeric)


def _require_disjoint_policy_logs(training: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Reject repeated held-out events by ID and by their immutable event signature."""
    if "decision_id" in training.columns and "decision_id" in evaluation.columns:
        training_ids = set(training["decision_id"].astype(str))
        evaluation_ids = set(evaluation["decision_id"].astype(str))
        if training_ids.intersection(evaluation_ids):
            raise ValueError("evaluation_decisions must be disjoint from policy training")
    training_signatures = {_policy_event_signature(row) for _, row in training.iterrows()}
    evaluation_signatures = {_policy_event_signature(row) for _, row in evaluation.iterrows()}
    if training_signatures.intersection(evaluation_signatures):
        raise ValueError("evaluation_decisions must be disjoint from policy training")


def _policy_event_signature(row: pd.Series) -> str:
    """Return a duplicate-resistant signature independent of mutable decision IDs."""
    payload = {
        "user_id": row["user_id"],
        "timestamp": float(row["timestamp"]),
        "context_hash": row["context_hash"],
        "candidate_item_ids": parse_candidate_list(row["candidate_item_ids"]),
        "chosen_item_id": row["chosen_item_id"],
        "reward": float(row["reward"]),
    }
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _base_scores_for_logged_row(row: pd.Series) -> list[float]:
    """Read the base adaptive scores required to replay a hybrid+CQL action.

    Current decision records store them in immutable policy metadata.  Older
    pre-overlay records have only ``scores``; those are accepted because they
    were necessarily the base action scores.  A record made by an older CQL
    overlay without this field cannot be evaluated exactly and is rejected.
    """
    candidates = parse_candidate_list(row["candidate_item_ids"])
    metadata_raw = row.get("policy_metadata")
    metadata: Mapping[str, Any] = {}
    if isinstance(metadata_raw, Mapping):
        metadata = metadata_raw
    elif isinstance(metadata_raw, str) and metadata_raw.strip():
        try:
            decoded = json.loads(metadata_raw)
        except json.JSONDecodeError as exc:
            raise ValueError("policy_metadata must be a JSON object when serialized") from exc
        if not isinstance(decoded, dict):
            raise ValueError("policy_metadata must decode to an object")
        metadata = decoded
    if "base_scores" in metadata:
        values = [float(value) for value in parse_candidate_list(metadata["base_scores"])]
        if len(values) != len(candidates) or not np.all(np.isfinite(values)):
            raise ValueError("policy_metadata.base_scores must be finite and align with candidate_item_ids")
        return values
    if "+cql" in str(row.get("policy_name", "")):
        raise ValueError(
            "hybrid+CQL logs require policy_metadata.base_scores for exact offline-policy evaluation"
        )
    values = [float(value) for value in parse_candidate_list(row["scores"])]
    if len(values) != len(candidates) or not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite and align with candidate_item_ids")
    return values


def _update_digest_with_mapping(digest: Any, value: Any) -> None:
    """Hash structured learned state with stable ordering across mapping types."""
    payload = json.dumps(_fingerprint_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest.update(payload.encode("utf-8"))


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            "mapping": [
                (repr(key), _fingerprint_value(item))
                for key, item in sorted(value.items(), key=lambda entry: repr(entry[0]))
            ]
        }
    if isinstance(value, np.ndarray):
        return {"array": value.tolist()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted((_fingerprint_value(item) for item in value), key=repr)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"repr": repr(value)}


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


def _cold_start_outcome_prior(*, competence: Optional[float], difficulty: Optional[float]) -> float:
    if competence is None and difficulty is None:
        return 0.5
    if competence is None:
        # difficulty is non-None here (both-None handled above).
        assert difficulty is not None
        return _clamp01(1.0 - difficulty)
    if difficulty is None:
        return _clamp01(competence)
    return _clamp01(0.5 + 0.5 * (competence - difficulty))


def _decision_score_regret(row: pd.Series) -> float:
    candidates = parse_candidate_list(row["candidate_item_ids"])
    scores = [float(value) for value in parse_candidate_list(row["scores"])]
    chosen_index = candidates.index(row["chosen_item_id"])
    return float(max(scores) - scores[chosen_index])


def _chosen_predictions_and_outcomes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[float] = []
    outcomes: list[float] = []
    if frame.empty or "predicted_outcomes" not in frame.columns or "outcome" not in frame.columns:
        return np.asarray(predictions, dtype=float), np.asarray(outcomes, dtype=float)
    for _, row in frame.dropna(subset=["predicted_outcomes", "outcome"]).iterrows():
        candidates = parse_candidate_list(row["candidate_item_ids"])
        values = [float(value) for value in parse_candidate_list(row["predicted_outcomes"])]
        predictions.append(values[candidates.index(row["chosen_item_id"])])
        outcomes.append(float(row["outcome"]))
    return np.asarray(predictions, dtype=float), np.asarray(outcomes, dtype=float)


def _half_window_shift(frame: pd.DataFrame, value_col: str) -> Optional[float]:
    if frame.empty or value_col not in frame.columns:
        return None
    ordered = frame.dropna(subset=[value_col]).sort_values(["outcome_timestamp", "decision_id"], kind="mergesort")
    if len(ordered) < 2:
        return None
    midpoint = len(ordered) // 2
    early = ordered.iloc[:midpoint][value_col].astype(float)
    late = ordered.iloc[midpoint:][value_col].astype(float)
    return float(late.mean() - early.mean())


def _half_window_calibration_shift(predicted: np.ndarray, observed: np.ndarray) -> Optional[float]:
    if predicted.size < 2:
        return None
    midpoint = predicted.size // 2
    early_bias = float(np.mean(predicted[:midpoint] - observed[:midpoint]))
    late_bias = float(np.mean(predicted[midpoint:] - observed[midpoint:]))
    return late_bias - early_bias
