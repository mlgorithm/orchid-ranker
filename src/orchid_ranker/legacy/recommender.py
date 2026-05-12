"""Compatibility re-exports for the historical generic recommender API."""
from __future__ import annotations

from ..recommender import STRATEGY_GUIDE, SUPPORTED_STRATEGIES, OrchidRecommender, Recommendation

__all__ = [
    "OrchidRecommender",
    "Recommendation",
    "STRATEGY_GUIDE",
    "SUPPORTED_STRATEGIES",
]
