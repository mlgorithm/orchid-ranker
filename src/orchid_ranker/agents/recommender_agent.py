"""Backward-compatible shim — all classes moved to focused modules.

Import from the new locations for clarity:
    from orchid_ranker.agents.policies import LinUCBPolicy, BootTS
    from orchid_ranker.agents.logging_util import JSONLLogger
    from orchid_ranker.agents.two_tower import TwoTowerRecommender
    from orchid_ranker.agents.dual_recommender import DualRecommender
"""

from orchid_ranker.agents.policies import LinUCBPolicy, BootTS  # noqa: F401
from orchid_ranker.agents.logging_util import JSONLLogger  # noqa: F401
from orchid_ranker.agents.two_tower import (  # noqa: F401
    TwoTowerRecommender,
    enable_debug_rec_logs,
    _DEBUG_REC,
    _d,
)
from orchid_ranker.agents.dual_recommender import DualRecommender  # noqa: F401

try:
    from orchid_ranker.agents.student_agent import ItemMeta  # noqa: F401
except ImportError:
    pass

__all__ = [
    "JSONLLogger",
    "LinUCBPolicy",
    "BootTS",
    "TwoTowerRecommender",
    "DualRecommender",
    "enable_debug_rec_logs",
]
