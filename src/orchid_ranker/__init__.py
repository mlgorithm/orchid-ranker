"""Orchid Ranker -- adaptive-learning and knowledge-tracing recommender.

Build systems that choose the next exercise, lesson, task, or review item from
learner state, catalog structure, progression reward, and live outcome updates.

The public API is intentionally adaptive-learning first. Orchid no longer
exports a generic recommender model zoo from the package root.

**Stable adaptive surface**:
    AdaptiveRanker, AdaptiveLearningEngine, AdaptiveLearningRecommender,
    knowledge-tracing tracers, BKT/EDM/IRT helpers, progression policies,
    prerequisite graphs, semantic exercise retrieval, logged-policy learning,
    offline policy evaluation, calibration, retention scheduling, and rollout
    safety helpers.

**Advanced surface**:
    AdaptiveAgent, AdaptiveAgentFactory, MultiUserOrchestrator, MultiConfig,
    UserCtx, optional observability, security, connector, and visualization
    helpers.

Installation extras
-------------------
``pip install orchid-ranker``            -- torch-free adaptive-learning utilities
``pip install orchid-ranker[adaptive]``  -- PyTorch-backed KT and AdaptiveRanker
``pip install orchid-ranker[all]``       -- adaptive stack plus optional operations integrations
"""

__version__ = "0.5.0"

# ── Tier 1: Torch-free core toolkit (always available) ────────────────────

from .adaptive_schema import (
    LearnerEvent,
    LoggedDecision,
    hash_identifier,
    learner_events_to_frame,
    logged_decisions_to_frame,
    stable_context_hash,
    validate_learner_events,
    validate_logged_decisions,
)
from .bandits import BanditScore, PersonalizedLinUCB
from .bkt_em import BKTFitReport, fit_bkt_em
from .calibration import (
    CalibrationReport,
    IsotonicProbabilityCalibrator,
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
)
from .curriculum import (
    DependencyGraph,
    ProgressionRecommender,
)
from .delayed_gain import (
    DelayedGainRewardModel,
    build_delayed_gain_training_frame,
    diagnose_delayed_gain_predictions,
    fit_delayed_gain_reward_model,
    fit_delayed_gain_reward_model_from_frame,
)
from .edm import AFMTracer, EDMTrainingReport, PFATracer
from .evaluation import (
    ProgressionReport,
    category_coverage,
    engagement_score,
    progression_gain,
    sequence_adherence,
    stretch_fit,
)
from .fqe import FQEReport, TabularFQE
from .irt import IRTAdaptiveSelector, IRTItem, IRTRecommendation
from .knowledge_tracing import (
    BayesianKnowledgeTracing,
    ForgettingCurve,
    ProficiencyTracker,
)
from .learning_policy import (
    DelayedGainValuePolicy,
    ProgressionValuePolicy,
    SupportConstrainedDelayedGainPolicy,
)
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
    deterministic_policy_probabilities,
    evaluate_logged_policy,
    evaluate_rollout_gate,
)
from .progression_reward import ProgressionRewardConfig
from .pykt_bridge import (
    PyKTPredictionAdapter,
    PyKTSequence,
    export_pykt_sequences,
    load_pykt_sequences,
    pykt_sequences_to_interactions,
)
from .scenarios import (
    ScenarioFit,
    ScenarioRecipe,
    available_scenarios,
    recommend_scenarios,
)
from .semantic import DenseSemanticItemEncoder, SemanticExerciseRanker, SemanticItemEncoder, SemanticRecommendation
from .spaced_repetition import FSRSReviewState, FSRSScheduler, ReviewRecommendation

# ── Lazy imports for torch-dependent and optional-dependency modules ──────
#
# Everything below is loaded on first access via __getattr__.
# This lets `import orchid_ranker` succeed without torch installed,
# while still providing the full API when torch is available.

_TORCH_LAZY = {
    # Tier 1 -- requires torch
    "AdaptiveRanker": (".adaptive_ranker", "AdaptiveRanker"),
    "AdaptiveRankerConfig": (".adaptive_ranker", "AdaptiveRankerConfig"),
    "AdaptiveLearningConfig": (".adaptive_learning", "AdaptiveLearningConfig"),
    "AdaptiveLearningEngine": (".adaptive_learning", "AdaptiveLearningRecommender"),
    "AdaptiveLearningRecommendation": (".adaptive_learning", "AdaptiveLearningRecommendation"),
    "AdaptiveLearningRecommender": (".adaptive_learning", "AdaptiveLearningRecommender"),
    "SAKTTracer": (".kt", "SAKTTracer"),
    "DKTTracer": (".kt", "DKTTracer"),
    "DKVMNTracer": (".kt", "DKVMNTracer"),
    "AKTTracer": (".kt", "AKTTracer"),
    "SAINTTracer": (".kt", "SAINTTracer"),
    "SAINTPlusTracer": (".kt", "SAINTPlusTracer"),
    # Tier 2 -- requires torch
    "AdaptiveAgent": (".agents.student_agent", "AdaptiveAgent"),
    "AdaptiveAgentFactory": (".agents.student_agent", "AdaptiveAgentFactory"),
    "MultiUserOrchestrator": (".agents.orchestrator", "MultiUserOrchestrator"),
    "MultiConfig": (".agents.config", "MultiConfig"),
    "UserCtx": (".agents.config", "UserCtx"),
}

# Tier 3 -- optional deps (torch, prometheus, matplotlib, etc.)
_OPTIONAL_LAZY = {
    "LinUCBPolicy": (".agents.policies", "LinUCBPolicy"),
    "BootTS": (".agents.policies", "BootTS"),
    "JSONLLogger": (".agents.logging_util", "JSONLLogger"),
    "plot_user_activity": (".visualization", "plot_user_activity"),
    "plot_item_difficulty": (".visualization", "plot_item_difficulty"),
    "plot_learning_curve": (".visualization", "plot_learning_curve"),
    "plot_acceptance_heatmap": (".visualization", "plot_acceptance_heatmap"),
    "plot_round_comparison": (".visualization", "plot_round_comparison"),
    "plot_knowledge_trajectory": (".visualization", "plot_knowledge_trajectory"),
    "get_dp_config": (".dp", "get_dp_config"),
    "configure_logging": (".logging", "configure_logging"),
    "AccessControl": (".security", "AccessControl"),
    "DEFAULT_POLICY": (".security", "DEFAULT_POLICY"),
    "AuditLogger": (".security", "AuditLogger"),
    "SnowflakeConnector": (".connectors", "SnowflakeConnector"),
    "BigQueryConnector": (".connectors", "BigQueryConnector"),
    "S3StreamConnector": (".connectors", "S3StreamConnector"),
    "MLflowTracker": (".connectors", "MLflowTracker"),
    "metrics_registry": (".observability", "metrics_registry"),
    "start_metrics_server": (".observability", "start_metrics_server"),
    "record_training": (".observability", "record_training"),
    "export_metrics": (".observability", "export_metrics"),
    "metrics_content_type": (".observability", "metrics_content_type"),
    "BloomFilter": (".sketch", "BloomFilter"),
    "CountMinSketch": (".sketch", "CountMinSketch"),
    "ExactEmbeddingIndex": (".sketch", "ExactEmbeddingIndex"),
    "ReservoirSampler": (".sketch", "ReservoirSampler"),
    "SketchCandidateGenerator": (".sketch", "SketchCandidateGenerator"),
}

_ALL_LAZY = {**_TORCH_LAZY, **_OPTIONAL_LAZY}

# Deprecated aliases: old_name -> (module_path, new_name)
_DEPRECATED_ALIASES = {
    "MasteryTracker": (".knowledge_tracing", "ProficiencyTracker"),
    "PrerequisiteGraph": (".curriculum", "DependencyGraph"),
    "CurriculumRecommender": (".curriculum", "ProgressionRecommender"),
    "learning_gain": (".evaluation", "progression_gain"),
    "knowledge_coverage": (".evaluation", "category_coverage"),
    "proficiency_coverage": (".evaluation", "category_coverage"),
    "curriculum_adherence": (".evaluation", "sequence_adherence"),
    "difficulty_appropriateness": (".evaluation", "stretch_fit"),
    "EducationalReport": (".evaluation", "ProgressionReport"),
    "StudentAgent": (".agents.student_agent", "AdaptiveAgent"),
    "StudentAgentFactory": (".agents.student_agent", "AdaptiveAgentFactory"),
}


def __getattr__(name: str):
    # Deprecated aliases — emit warning, then return the new name
    if name in _DEPRECATED_ALIASES:
        import importlib
        import warnings
        module_path, attr_name = _DEPRECATED_ALIASES[name]

        # torch-dependent deprecated names need torch check
        if name in _TORCH_LAZY:
            from ._compat import require_torch
            require_torch(f"orchid_ranker.{name}")

        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        warnings.warn(
            f"orchid_ranker.{name} is deprecated, use {attr_name} instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        globals()[name] = value
        return value

    # Lazy imports (torch-dependent and optional)
    if name in _ALL_LAZY:
        module_path, attr_name = _ALL_LAZY[name]

        # For torch-dependent modules, provide a clear error
        if name in _TORCH_LAZY:
            from ._compat import require_torch
            require_torch(f"orchid_ranker.{name}")

        import importlib
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        # Cache in module globals so __getattr__ is only called once
        globals()[name] = value
        return value

    raise AttributeError(f"module 'orchid_ranker' has no attribute {name!r}")


# ── __all__: Only Tier 1 + Tier 2 symbols are public ────────────────────

__all__ = [
    # Tier 1 -- Stable (generic names)
    "__version__",
    "AdaptiveRanker",
    "AdaptiveRankerConfig",
    "AdaptiveLearningConfig",
    "AdaptiveLearningEngine",
    "AdaptiveLearningRecommendation",
    "AdaptiveLearningRecommender",
    "SAKTTracer",
    "DKTTracer",
    "DKVMNTracer",
    "AKTTracer",
    "SAINTTracer",
    "SAINTPlusTracer",
    "BayesianKnowledgeTracing",
    "ProficiencyTracker",
    "ForgettingCurve",
    "DependencyGraph",
    "ProgressionRecommender",
    "progression_gain",
    "category_coverage",
    "sequence_adherence",
    "stretch_fit",
    "engagement_score",
    "ProgressionReport",
    "LoggedPolicyReport",
    "OPERolloutGateReport",
    "PolicyComparisonReport",
    "BootstrapLoggedPolicyReport",
    "BootstrapPolicyComparisonReport",
    "evaluate_logged_policy",
    "compare_logged_policies",
    "bootstrap_logged_policy",
    "bootstrap_compare_logged_policies",
    "deterministic_policy_probabilities",
    "evaluate_rollout_gate",
    "LearnerEvent",
    "LoggedDecision",
    "hash_identifier",
    "learner_events_to_frame",
    "logged_decisions_to_frame",
    "stable_context_hash",
    "validate_learner_events",
    "validate_logged_decisions",
    "CQLDiscretePolicy",
    "CQLTrainingReport",
    "TabularFQE",
    "FQEReport",
    "PersonalizedLinUCB",
    "BanditScore",
    "PFATracer",
    "AFMTracer",
    "EDMTrainingReport",
    "fit_bkt_em",
    "BKTFitReport",
    "FSRSScheduler",
    "FSRSReviewState",
    "ReviewRecommendation",
    "TemperatureScaler",
    "IsotonicProbabilityCalibrator",
    "CalibrationReport",
    "expected_calibration_error",
    "brier_score",
    "BloomFilter",
    "CountMinSketch",
    "ExactEmbeddingIndex",
    "ReservoirSampler",
    "SketchCandidateGenerator",
    "DenseSemanticItemEncoder",
    "SemanticItemEncoder",
    "SemanticRecommendation",
    "SemanticExerciseRanker",
    "IRTAdaptiveSelector",
    "IRTItem",
    "IRTRecommendation",
    "PyKTSequence",
    "PyKTPredictionAdapter",
    "export_pykt_sequences",
    "load_pykt_sequences",
    "pykt_sequences_to_interactions",
    "DelayedGainRewardModel",
    "build_delayed_gain_training_frame",
    "diagnose_delayed_gain_predictions",
    "fit_delayed_gain_reward_model",
    "fit_delayed_gain_reward_model_from_frame",
    "ScenarioFit",
    "ScenarioRecipe",
    "available_scenarios",
    "recommend_scenarios",
    "DelayedGainValuePolicy",
    "ProgressionValuePolicy",
    "ProgressionRewardConfig",
    "SupportConstrainedDelayedGainPolicy",
    # Tier 1 -- Backward-compatible aliases
    "MasteryTracker",
    "PrerequisiteGraph",
    "CurriculumRecommender",
    "learning_gain",
    "knowledge_coverage",
    "proficiency_coverage",
    "curriculum_adherence",
    "difficulty_appropriateness",
    "EducationalReport",
    # Tier 2 -- Advanced (generic names)
    "AdaptiveAgent",
    "AdaptiveAgentFactory",
    "MultiUserOrchestrator",
    "MultiConfig",
    "UserCtx",
    # Tier 2 -- Backward-compatible aliases
    "StudentAgent",
    "StudentAgentFactory",
]
