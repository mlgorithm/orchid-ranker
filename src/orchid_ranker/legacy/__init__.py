"""Backward-compatible generic recommender namespace.

Adaptive learning is Orchid's primary product surface. The historical generic
strategy zoo remains importable here for older experiments and migration work.
"""
from __future__ import annotations

from ..recommender import STRATEGY_GUIDE, SUPPORTED_STRATEGIES, OrchidRecommender, Recommendation

__all__ = [
    "OrchidRecommender",
    "Recommendation",
    "STRATEGY_GUIDE",
    "SUPPORTED_STRATEGIES",
]
