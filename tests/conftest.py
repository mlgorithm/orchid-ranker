"""Shared test fixtures for Orchid Ranker test suite."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if src.exists():
    sys.path.insert(0, str(src))


# ── Knowledge tracing fixtures ───────────────────────────────────────────

@pytest.fixture
def sample_bkt():
    """A fresh BKT instance with default parameters."""
    from orchid_ranker import BayesianKnowledgeTracing
    return BayesianKnowledgeTracing()


@pytest.fixture
def trained_bkt():
    """A BKT instance after 50 correct observations."""
    from orchid_ranker import BayesianKnowledgeTracing
    bkt = BayesianKnowledgeTracing()
    for _ in range(50):
        bkt.update(correct=True)
    return bkt


@pytest.fixture
def sample_mastery_tracker():
    """A ProficiencyTracker with 5 math skills."""
    from orchid_ranker import ProficiencyTracker
    return ProficiencyTracker(skills=["algebra", "geometry", "calculus", "statistics", "linear_algebra"])


# ── Curriculum fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_prerequisite_graph():
    """A prerequisite graph: algebra → calculus → diff_eq, algebra → statistics."""
    from orchid_ranker import DependencyGraph
    g = DependencyGraph()
    g.add_edge("algebra", "calculus")
    g.add_edge("algebra", "statistics")
    g.add_edge("calculus", "differential_equations")
    return g


@pytest.fixture
def sample_curriculum_recommender(sample_prerequisite_graph):
    """A ProgressionRecommender with the sample graph and difficulty map."""
    from orchid_ranker import ProgressionRecommender
    return ProgressionRecommender(
        graph=sample_prerequisite_graph,
        difficulty_map={
            "algebra": 0.3,
            "calculus": 0.6,
            "statistics": 0.5,
            "differential_equations": 0.8,
        },
    )
