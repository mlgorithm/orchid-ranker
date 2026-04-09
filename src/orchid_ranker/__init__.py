"""Orchid Ranker – adaptive educational recommender toolkit.

Public API is organized into three stability tiers:

**Tier 1 – Stable** (semver-guaranteed, safe for production):
    OrchidRecommender, Recommendation, BayesianKnowledgeTracing,
    MasteryTracker, ForgettingCurve, PrerequisiteGraph,
    CurriculumRecommender, save_model, load_model, cross_validate,
    compare_models, train_test_split, evaluate_on_holdout,
    learning_gain, knowledge_coverage, curriculum_adherence,
    difficulty_appropriateness, engagement_score, EducationalReport

**Tier 2 – Advanced** (stable but may evolve between minor versions):
    StudentAgent, StudentAgentFactory, MultiUserOrchestrator,
    TwoTowerRecommender, DualRecommender, MultiConfig, UserCtx

**Tier 3 – Internal / Experimental** (not in ``__all__``; import from submodule):
    LinUCBPolicy, BootTS, JSONLLogger, PolicyState, OnlineState,
    GridSearchCV, RandomSearchCV, RankingExperiment, get_dp_config,
    configure_logging, AccessControl, AuditLogger, connectors, etc.
"""

__version__ = "0.3.0"

# ── Tier 1: Stable public API (semver-guaranteed) ───────────────────────

from .recommender import OrchidRecommender, Recommendation, SUPPORTED_STRATEGIES, STRATEGY_GUIDE
from .knowledge_tracing import (
    BayesianKnowledgeTracing,
    MasteryTracker,
    ForgettingCurve,
)
from .curriculum import PrerequisiteGraph, CurriculumRecommender
from .evaluation import (
    learning_gain,
    knowledge_coverage,
    curriculum_adherence,
    difficulty_appropriateness,
    engagement_score,
    EducationalReport,
)
from .serialization import save_model, load_model
from .model_selection import (
    cross_validate,
    compare_models,
    train_test_split,
    evaluate_on_holdout,
)

# ── Tier 2: Advanced API (stable, may evolve between minor versions) ────

from .agents.student_agent import StudentAgent, StudentAgentFactory
from .agents.orchestrator import MultiUserOrchestrator
from .agents.config import MultiConfig, UserCtx
from .agents.two_tower import TwoTowerRecommender
from .agents.dual_recommender import DualRecommender

# ── Tier 3: Internal / Experimental (import from submodule directly) ────
# These are importable but NOT in __all__. Use at your own risk across
# minor version bumps.

from .agents.policies import LinUCBPolicy, BootTS
from .agents.logging_util import JSONLLogger
from .visualization import (
    plot_user_activity,
    plot_item_difficulty,
    plot_learning_curve,
    plot_acceptance_heatmap,
    plot_round_comparison,
    plot_knowledge_trajectory,
)
from .experiments import RankingExperiment
from .dp import get_dp_config
from .logging import configure_logging
from .security import AccessControl, DEFAULT_POLICY, AuditLogger
from .connectors import (
    SnowflakeConnector,
    BigQueryConnector,
    S3StreamConnector,
    MLflowTracker,
)
from .observability import (
    metrics_registry,
    start_metrics_server,
    record_training,
    export_metrics,
    metrics_content_type,
)
from .tuning import GridSearchCV, RandomSearchCV

# ── __all__: Only Tier 1 + Tier 2 symbols are public ────────────────────

__all__ = [
    # Tier 1 — Stable
    "__version__",
    "OrchidRecommender",
    "Recommendation",
    "SUPPORTED_STRATEGIES",
    "STRATEGY_GUIDE",
    "BayesianKnowledgeTracing",
    "MasteryTracker",
    "ForgettingCurve",
    "PrerequisiteGraph",
    "CurriculumRecommender",
    "learning_gain",
    "knowledge_coverage",
    "curriculum_adherence",
    "difficulty_appropriateness",
    "engagement_score",
    "EducationalReport",
    "save_model",
    "load_model",
    "cross_validate",
    "compare_models",
    "train_test_split",
    "evaluate_on_holdout",
    # Tier 2 — Advanced
    "StudentAgent",
    "StudentAgentFactory",
    "MultiUserOrchestrator",
    "MultiConfig",
    "UserCtx",
    "TwoTowerRecommender",
    "DualRecommender",
]
