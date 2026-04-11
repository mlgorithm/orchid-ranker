"""Agent modules: recommenders, adaptive agents, orchestrators."""

from .config import MultiConfig, UserCtx
from .dual_recommender import DualRecommender
from .logging_util import JSONLLogger
from .orchestrator import MultiUserOrchestrator
from .policies import BootTS, LinUCBPolicy
from .student_agent import AdaptiveAgent, AdaptiveAgentFactory
from .two_tower import TwoTowerRecommender

__all__ = [
    "AdaptiveAgent",
    "AdaptiveAgentFactory",
    "TwoTowerRecommender",
    "DualRecommender",
    "JSONLLogger",
    "LinUCBPolicy",
    "BootTS",
    "MultiUserOrchestrator",
    "MultiConfig",
    "UserCtx",
    # Backward-compatible aliases (deprecated)
    "StudentAgent",
    "StudentAgentFactory",
]


# --- Deprecation handling for renamed symbols (PEP 562) ---
_DEPRECATED_NAMES = {
    "StudentAgent": (".student_agent", "AdaptiveAgent"),
    "StudentAgentFactory": (".student_agent", "AdaptiveAgentFactory"),
}


def __getattr__(name: str):
    if name in _DEPRECATED_NAMES:
        import importlib
        import warnings
        module_path, attr_name = _DEPRECATED_NAMES[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        warnings.warn(
            f"orchid_ranker.agents.{name} is deprecated, use {attr_name} instead. "
            "Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
