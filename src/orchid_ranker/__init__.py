"""Orchid Ranker – adaptive educational recommender toolkit."""

__version__ = "0.2.1"

from .agents.student_agent import StudentAgent, StudentAgentFactory
from .agents.agentic import MultiUserOrchestrator, MultiConfig, UserCtx
from .agents.recommender_agent import TwoTowerRecommender, DualRecommender, LinUCBPolicy, BootTS, JSONLLogger
from .visualization import (
    plot_user_activity,
    plot_item_difficulty,
    plot_learning_curve,
    plot_acceptance_heatmap,
    plot_round_comparison,
    plot_knowledge_trajectory,
)
from .experiments import RankingExperiment
from .recommender import OrchidRecommender, Recommendation, SUPPORTED_STRATEGIES, STRATEGY_GUIDE
from .serialization import save_model, load_model
from .evaluation import (
    learning_gain,
    knowledge_coverage,
    curriculum_adherence,
    difficulty_appropriateness,
    engagement_score,
    EducationalReport,
)
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
from .model_selection import (
    cross_validate,
    compare_models,
    train_test_split,
    evaluate_on_holdout,
)
from .tuning import GridSearchCV, RandomSearchCV
from .knowledge_tracing import (
    BayesianKnowledgeTracing,
    MasteryTracker,
    ForgettingCurve,
)
from .curriculum import PrerequisiteGraph, CurriculumRecommender

__all__ = [
    "__version__",
    "StudentAgent",
    "StudentAgentFactory",
    "MultiUserOrchestrator",
    "MultiConfig",
    "UserCtx",
    "TwoTowerRecommender",
    "DualRecommender",
    "LinUCBPolicy",
    "BootTS",
    "JSONLLogger",
    "RankingExperiment",
    "get_dp_config",
    "configure_logging",
    "plot_user_activity",
    "plot_item_difficulty",
    "plot_learning_curve",
    "plot_acceptance_heatmap",
    "plot_round_comparison",
    "plot_knowledge_trajectory",
    "OrchidRecommender",
    "Recommendation",
    "SUPPORTED_STRATEGIES",
    "STRATEGY_GUIDE",
    "learning_gain",
    "knowledge_coverage",
    "curriculum_adherence",
    "difficulty_appropriateness",
    "engagement_score",
    "EducationalReport",
    "save_model",
    "load_model",
    "AccessControl",
    "DEFAULT_POLICY",
    "AuditLogger",
    "SnowflakeConnector",
    "BigQueryConnector",
    "S3StreamConnector",
    "MLflowTracker",
    "metrics_registry",
    "start_metrics_server",
    "record_training",
    "export_metrics",
    "metrics_content_type",
    "cross_validate",
    "compare_models",
    "train_test_split",
    "evaluate_on_holdout",
    "GridSearchCV",
    "RandomSearchCV",
    "BayesianKnowledgeTracing",
    "MasteryTracker",
    "ForgettingCurve",
    "PrerequisiteGraph",
    "CurriculumRecommender",
]
