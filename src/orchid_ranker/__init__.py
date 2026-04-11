"""Orchid Ranker -- adaptive progression & recommender toolkit.

Build adaptive systems for any domain: education, corporate training,
rehabilitation, fitness, gaming, onboarding, and more.

Public API is organized into three stability tiers:

**Tier 1 -- Stable** (semver-guaranteed, safe for production):
    OrchidRecommender, Recommendation, BayesianKnowledgeTracing,
    ProficiencyTracker, ForgettingCurve, DependencyGraph,
    ProgressionRecommender, save_model, load_model, cross_validate,
    compare_models, train_test_split, evaluate_on_holdout,
    progression_gain, proficiency_coverage, sequence_adherence,
    difficulty_appropriateness, engagement_score, ProgressionReport

**Tier 2 -- Advanced** (stable but may evolve between minor versions):
    AdaptiveAgent, AdaptiveAgentFactory, MultiUserOrchestrator,
    TwoTowerRecommender, DualRecommender, MultiConfig, UserCtx

**Tier 3 -- Internal / Experimental** (not in ``__all__``; import from submodule):
    LinUCBPolicy, BootTS, JSONLLogger, PolicyState, OnlineState,
    GridSearchCV, RandomSearchCV, RankingExperiment, get_dp_config,
    configure_logging, AccessControl, AuditLogger, connectors, etc.

Installation extras
-------------------
``pip install orchid-ranker``          -- educational toolkit (BKT, curriculum, evaluation)
``pip install orchid-ranker[ml]``      -- adds PyTorch for ML recommender strategies
``pip install orchid-ranker[all]``     -- everything (ML, viz, agentic, observability, connectors)
"""

__version__ = "0.3.1"

# ── Tier 1: Torch-free educational toolkit (always available) ─────────────

from .curriculum import (
    DependencyGraph,
    ProgressionRecommender,
)
from .evaluation import (
    ProgressionReport,
    difficulty_appropriateness,
    engagement_score,
    proficiency_coverage,
    progression_gain,
    sequence_adherence,
)
from .knowledge_tracing import (
    BayesianKnowledgeTracing,
    ForgettingCurve,
    ProficiencyTracker,
)

# ── Lazy imports for torch-dependent and optional-dependency modules ──────
#
# Everything below is loaded on first access via __getattr__.
# This lets `import orchid_ranker` succeed without torch installed,
# while still providing the full API when torch is available.

_TORCH_LAZY = {
    # Tier 1 -- requires torch
    "OrchidRecommender": (".recommender", "OrchidRecommender"),
    "Recommendation": (".recommender", "Recommendation"),
    "SUPPORTED_STRATEGIES": (".recommender", "SUPPORTED_STRATEGIES"),
    "STRATEGY_GUIDE": (".recommender", "STRATEGY_GUIDE"),
    "save_model": (".serialization", "save_model"),
    "load_model": (".serialization", "load_model"),
    "cross_validate": (".model_selection", "cross_validate"),
    "compare_models": (".model_selection", "compare_models"),
    "train_test_split": (".model_selection", "train_test_split"),
    "evaluate_on_holdout": (".model_selection", "evaluate_on_holdout"),
    "EVALUATION_METRICS": (".model_selection", "EVALUATION_METRICS"),
    # Tier 2 -- requires torch
    "AdaptiveAgent": (".agents.student_agent", "AdaptiveAgent"),
    "AdaptiveAgentFactory": (".agents.student_agent", "AdaptiveAgentFactory"),
    "MultiUserOrchestrator": (".agents.orchestrator", "MultiUserOrchestrator"),
    "MultiConfig": (".agents.config", "MultiConfig"),
    "UserCtx": (".agents.config", "UserCtx"),
    "TwoTowerRecommender": (".agents.two_tower", "TwoTowerRecommender"),
    "DualRecommender": (".agents.dual_recommender", "DualRecommender"),
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
    "RankingExperiment": (".experiments", "RankingExperiment"),
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
    "GridSearchCV": (".tuning", "GridSearchCV"),
    "RandomSearchCV": (".tuning", "RandomSearchCV"),
}

_ALL_LAZY = {**_TORCH_LAZY, **_OPTIONAL_LAZY}

# Deprecated aliases: old_name -> (module_path, new_name)
_DEPRECATED_ALIASES = {
    "MasteryTracker": (".knowledge_tracing", "ProficiencyTracker"),
    "PrerequisiteGraph": (".curriculum", "DependencyGraph"),
    "CurriculumRecommender": (".curriculum", "ProgressionRecommender"),
    "learning_gain": (".evaluation", "progression_gain"),
    "knowledge_coverage": (".evaluation", "proficiency_coverage"),
    "curriculum_adherence": (".evaluation", "sequence_adherence"),
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
        new_public_name = attr_name if attr_name != name else attr_name
        warnings.warn(
            f"orchid_ranker.{name} is deprecated, use {new_public_name} instead. "
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
    "OrchidRecommender",
    "Recommendation",
    "SUPPORTED_STRATEGIES",
    "STRATEGY_GUIDE",
    "BayesianKnowledgeTracing",
    "ProficiencyTracker",
    "ForgettingCurve",
    "DependencyGraph",
    "ProgressionRecommender",
    "progression_gain",
    "proficiency_coverage",
    "sequence_adherence",
    "difficulty_appropriateness",
    "engagement_score",
    "ProgressionReport",
    "save_model",
    "load_model",
    "cross_validate",
    "compare_models",
    "train_test_split",
    "evaluate_on_holdout",
    "EVALUATION_METRICS",
    # Tier 1 -- Backward-compatible aliases
    "MasteryTracker",
    "PrerequisiteGraph",
    "CurriculumRecommender",
    "learning_gain",
    "knowledge_coverage",
    "curriculum_adherence",
    "EducationalReport",
    # Tier 2 -- Advanced (generic names)
    "AdaptiveAgent",
    "AdaptiveAgentFactory",
    "MultiUserOrchestrator",
    "MultiConfig",
    "UserCtx",
    "TwoTowerRecommender",
    "DualRecommender",
    # Tier 2 -- Backward-compatible aliases
    "StudentAgent",
    "StudentAgentFactory",
]
