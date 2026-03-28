"""Shared test fixtures for Orchid Ranker test suite."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if src.exists():
    sys.path.insert(0, str(src))


# ── Data fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def small_interactions():
    """Minimal interaction DataFrame (10 users, 20 items, 200 rows)."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "user_id": rng.randint(0, 10, size=200),
        "item_id": rng.randint(0, 20, size=200),
        "rating":  rng.uniform(1, 5, size=200).round(1),
    })


@pytest.fixture
def medium_interactions():
    """Medium interaction DataFrame (50 users, 100 items, 5000 rows)."""
    rng = np.random.RandomState(99)
    return pd.DataFrame({
        "user_id": rng.randint(0, 50, size=5000),
        "item_id": rng.randint(0, 100, size=5000),
        "rating":  rng.uniform(1, 5, size=5000).round(1),
    })


@pytest.fixture
def binary_interactions():
    """Interaction DataFrame with binary labels (accept/reject)."""
    rng = np.random.RandomState(7)
    return pd.DataFrame({
        "user_id": rng.randint(0, 20, size=500),
        "item_id": rng.randint(0, 30, size=500),
        "rating":  rng.randint(0, 2, size=500).astype(float),
    })


# ── Model fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def fitted_popularity_model(small_interactions):
    """A fitted popularity recommender."""
    from orchid_ranker import OrchidRecommender
    rec = OrchidRecommender(strategy="popularity")
    rec.fit(small_interactions)
    return rec


@pytest.fixture
def fitted_als_model(small_interactions):
    """A fitted ALS recommender."""
    from orchid_ranker import OrchidRecommender
    rec = OrchidRecommender(strategy="als", epochs=3)
    rec.fit(small_interactions)
    return rec


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
    """A MasteryTracker with 5 math skills."""
    from orchid_ranker import MasteryTracker
    return MasteryTracker(skills=["algebra", "geometry", "calculus", "statistics", "linear_algebra"])


# ── Curriculum fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_prerequisite_graph():
    """A prerequisite graph: algebra → calculus → diff_eq, algebra → statistics."""
    from orchid_ranker import PrerequisiteGraph
    g = PrerequisiteGraph()
    g.add_edge("algebra", "calculus")
    g.add_edge("algebra", "statistics")
    g.add_edge("calculus", "differential_equations")
    return g


@pytest.fixture
def sample_curriculum_recommender(sample_prerequisite_graph):
    """A CurriculumRecommender with the sample graph and difficulty map."""
    from orchid_ranker import CurriculumRecommender
    return CurriculumRecommender(
        graph=sample_prerequisite_graph,
        difficulty_map={
            "algebra": 0.3,
            "calculus": 0.6,
            "statistics": 0.5,
            "differential_equations": 0.8,
        },
    )


# ── Temporary directory fixture ──────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Temporary directory for file I/O tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
