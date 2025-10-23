"""Orchid Ranker – adaptive educational recommender toolkit."""

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
from .recommender import OrchidRecommender, Recommendation
from .dp import get_dp_config

__all__ = [
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
    "plot_user_activity",
    "plot_item_difficulty",
    "plot_learning_curve",
    "plot_acceptance_heatmap",
    "plot_round_comparison",
    "plot_knowledge_trajectory",
    "OrchidRecommender",
    "Recommendation",
]
