"""Edge case and boundary condition tests for OrchidRecommender and related classes.

Tests production-failure scenarios including:
- Empty/minimal data (empty DataFrame, single user, single item, single interaction)
- Duplicate interactions
- Invalid data (NaN, Inf values)
- Unknown users/items in predict/recommend
- Boundary parameters (top_k=0, top_k > available items)
- Fully-seen items with filter_seen=True
- Mismatched predict_many inputs
- BayesianKnowledgeTracing edge cases (p_init=0, p_init=1, oscillation)
- PrerequisiteGraph cycles and self-loops
- MasteryTracker unknown skills
- GridSearchCV empty param_grid
- Model save/load edge cases
"""

import tempfile
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import pytest

from orchid_ranker import (
    OrchidRecommender,
    BayesianKnowledgeTracing,
    MasteryTracker,
    PrerequisiteGraph,
)


# ────────────────────────────────────────────────────────────────────────────
# OrchidRecommender edge cases
# ────────────────────────────────────────────────────────────────────────────

def test_fit_empty_dataframe_raises_value_error():
    """Verify fit() raises ValueError on empty DataFrame."""
    rec = OrchidRecommender(strategy="als")
    empty_df = pd.DataFrame({"user_id": [], "item_id": [], "rating": []})

    with pytest.raises(ValueError, match="empty"):
        rec.fit(empty_df, rating_col="rating")


def test_fit_single_user_single_item():
    """Verify fit() handles single user, single item."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1],
        "item_id": [10],
        "rating": [5.0],
    })

    # Should not raise
    rec.fit(data, rating_col="rating")
    assert len(rec.all_users()) == 1
    assert len(rec.all_items()) == 1


def test_fit_single_interaction():
    """Verify fit() handles single interaction row."""
    rec = OrchidRecommender(strategy="popularity")
    data = pd.DataFrame({
        "user_id": [1],
        "item_id": [10],
        "rating": [1.0],
    })

    rec.fit(data, rating_col="rating")
    assert len(rec.all_users()) == 1
    assert len(rec.all_items()) == 1


def test_fit_handles_duplicate_interactions_gracefully():
    """Verify fit() handles duplicate user-item pairs without error."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2],
        "item_id": [10, 10, 10, 10, 11],  # user 1 interacted with item 10 three times
        "rating": [1.0, 2.0, 1.5, 3.0, 4.0],
    })

    # Should not raise
    rec.fit(data, rating_col="rating")
    assert len(rec.all_users()) == 2
    assert len(rec.all_items()) == 2


def test_fit_with_nan_in_ratings_raises():
    """Verify fit() raises or handles NaN ratings gracefully."""
    rec = OrchidRecommender(strategy="als")
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, np.nan, 3.0],
    })

    # Should raise or handle gracefully; most strategies should fail
    with pytest.raises((ValueError, RuntimeError)):
        rec.fit(data, rating_col="rating")


def test_fit_with_inf_in_ratings_raises():
    """Verify fit() raises or handles Inf ratings gracefully."""
    rec = OrchidRecommender(strategy="als")
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, np.inf, 3.0],
    })

    with pytest.raises((ValueError, RuntimeError)):
        rec.fit(data, rating_col="rating")


def test_predict_unknown_user_raises_key_error():
    """Verify predict() raises KeyError for unknown user."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    with pytest.raises(KeyError):
        rec.predict(user_id=999, item_id=10)


def test_predict_unknown_item_raises_key_error():
    """Verify predict() raises KeyError for unknown item."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    with pytest.raises(KeyError):
        rec.predict(user_id=1, item_id=999)


def test_predict_unknown_user_and_item_raises_key_error():
    """Verify predict() raises KeyError for both unknown user and item."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    with pytest.raises(KeyError):
        rec.predict(user_id=999, item_id=999)


def test_recommend_unknown_user_raises_key_error():
    """Verify recommend() raises KeyError for unknown user."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    with pytest.raises(KeyError):
        rec.recommend(user_id=999, top_k=5)


def test_recommend_top_k_zero_returns_empty_list():
    """Verify recommend(top_k=0) returns empty list."""
    rec = OrchidRecommender(strategy="popularity")
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    recs = rec.recommend(user_id=1, top_k=0)

    assert isinstance(recs, list)
    assert len(recs) == 0


def test_recommend_top_k_greater_than_available_items():
    """Verify recommend() returns fewer items when top_k > available items."""
    rec = OrchidRecommender(strategy="popularity")
    data = pd.DataFrame({
        "user_id": [1, 1, 1],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    # Request top 100 items when only 3 exist
    recs = rec.recommend(user_id=1, top_k=100, filter_seen=False)

    assert len(recs) <= 3, (
        f"Expected at most 3 items, got {len(recs)}"
    )


def test_recommend_filter_seen_true_all_items_seen_returns_empty():
    """Verify recommend(filter_seen=True) returns empty list if all items seen."""
    rec = OrchidRecommender(strategy="popularity")
    data = pd.DataFrame({
        "user_id": [1, 1, 1],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    # User 1 has seen all 3 items
    recs = rec.recommend(user_id=1, top_k=10, filter_seen=True)

    assert len(recs) == 0, (
        f"Expected empty list when all items seen, got {len(recs)} items"
    )


def test_predict_many_mismatched_lengths_raises_value_error():
    """Verify predict_many() raises ValueError when user_ids and item_ids differ in length."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    with pytest.raises(ValueError, match="same length"):
        rec.predict_many([1, 2], [10, 11, 12])  # 2 users, 3 items


def test_predict_many_empty_inputs_returns_empty_array():
    """Verify predict_many() with empty inputs returns empty array."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    scores = rec.predict_many([], [])

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 0
    assert scores.dtype in (np.float32, np.float64)


def test_predict_many_unknown_user_raises_key_error():
    """Verify predict_many() raises KeyError for unknown user."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    with pytest.raises(KeyError):
        rec.predict_many([999], [10])


def test_predict_many_unknown_item_raises_key_error():
    """Verify predict_many() raises KeyError for unknown item."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1.0, 2.0, 3.0],
    })
    rec.fit(data, rating_col="rating")

    with pytest.raises(KeyError):
        rec.predict_many([1], [999])


def test_save_before_fit_raises_runtime_error(tmp_path):
    """Verify save() before fit() raises RuntimeError."""
    rec = OrchidRecommender(strategy="als")
    path = tmp_path / "model.pt"

    with pytest.raises(RuntimeError):
        rec.save(str(path))


def test_load_nonexistent_path_raises_file_not_found_error():
    """Verify load() raises FileNotFoundError for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        OrchidRecommender.load("/nonexistent/path/model.pt")


def test_save_and_load_roundtrip(tmp_path):
    """Verify save() and load() roundtrip preserves model state."""
    # Create and fit a model
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "item_id": [10, 11, 12, 10, 11],
        "rating": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    rec.fit(data, rating_col="rating")

    # Save
    path = tmp_path / "model.pt"
    rec.save(str(path))
    assert path.exists()

    # Load
    rec_loaded = OrchidRecommender.load(str(path))

    # Verify predictions match
    score_orig = rec.predict(1, 10)
    score_loaded = rec_loaded.predict(1, 10)

    assert np.isclose(score_orig, score_loaded, rtol=1e-5)


def test_recommend_before_fit_raises_runtime_error():
    """Verify recommend() before fit() raises RuntimeError."""
    rec = OrchidRecommender(strategy="als")

    with pytest.raises(RuntimeError):
        rec.recommend(user_id=1, top_k=5)


def test_predict_before_fit_raises_runtime_error():
    """Verify predict() before fit() raises RuntimeError."""
    rec = OrchidRecommender(strategy="als")

    with pytest.raises(RuntimeError):
        rec.predict(user_id=1, item_id=10)


# ────────────────────────────────────────────────────────────────────────────
# BayesianKnowledgeTracing edge cases
# ────────────────────────────────────────────────────────────────────────────

def test_bkt_p_init_zero():
    """Verify BKT with p_init=0 works (knowledge starts at 0)."""
    bkt = BayesianKnowledgeTracing(p_init=0.0)

    assert bkt.p_known() == 0.0
    assert not bkt.is_mastered()

    # Update with correct answer should increase knowledge
    bkt.update(correct=True)
    assert bkt.p_known() > 0.0


def test_bkt_p_init_one():
    """Verify BKT with p_init=1 works (knowledge starts at 1)."""
    bkt = BayesianKnowledgeTracing(p_init=1.0)

    assert bkt.p_known() == 1.0
    assert bkt.is_mastered()

    # Even with incorrect answer, knowledge should stay high
    bkt.update(correct=False)
    assert bkt.p_known() >= 0.8  # Should stay relatively high


def test_bkt_mastery_threshold_zero():
    """Verify BKT with mastery_threshold=0 always mastered."""
    bkt = BayesianKnowledgeTracing(p_init=0.0, mastery_threshold=0.0)

    assert bkt.is_mastered()


def test_bkt_mastery_threshold_one():
    """Verify BKT with mastery_threshold=1.0 rarely mastered."""
    bkt = BayesianKnowledgeTracing(p_init=0.5, mastery_threshold=1.0)

    assert not bkt.is_mastered()

    # Even many correct answers might not reach 1.0 exactly
    for _ in range(100):
        bkt.update(correct=True)

    # Should be very close but might not be exactly 1.0
    assert bkt.p_known() >= 0.99


def test_bkt_alternating_correct_incorrect_oscillates():
    """Verify BKT knowledge oscillates with alternating correct/incorrect."""
    bkt = BayesianKnowledgeTracing(p_init=0.5, p_transit=0.1)

    knowledge_values = [bkt.p_known()]

    for i in range(10):
        correct = i % 2 == 0  # Alternate correct/incorrect
        bkt.update(correct)
        knowledge_values.append(bkt.p_known())

    # Should have variation (not monotonic)
    assert len(set(np.round(knowledge_values, 3))) > 3, (
        "Knowledge should oscillate with alternating feedback"
    )


def test_bkt_reset_returns_to_prior():
    """Verify reset() returns BKT to initial state."""
    bkt = BayesianKnowledgeTracing(p_init=0.2)

    initial_p = bkt.p_known()

    # Make many updates
    for _ in range(50):
        bkt.update(correct=True)

    assert bkt.p_known() > initial_p

    # Reset
    bkt.reset()

    assert bkt.p_known() == 0.2
    assert bkt.p_known() == initial_p


def test_bkt_invalid_p_init_raises():
    """Verify BKT raises ValueError for invalid p_init."""
    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(p_init=-0.1)

    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(p_init=1.5)


def test_bkt_invalid_p_transit_raises():
    """Verify BKT raises ValueError for invalid p_transit."""
    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(p_transit=-0.1)

    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(p_transit=1.5)


def test_bkt_invalid_p_slip_raises():
    """Verify BKT raises ValueError for invalid p_slip."""
    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(p_slip=-0.1)

    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(p_slip=1.5)


def test_bkt_invalid_p_guess_raises():
    """Verify BKT raises ValueError for invalid p_guess."""
    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(p_guess=-0.1)

    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(p_guess=1.5)


def test_bkt_invalid_mastery_threshold_raises():
    """Verify BKT raises ValueError for invalid mastery_threshold."""
    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(mastery_threshold=-0.1)

    with pytest.raises(ValueError):
        BayesianKnowledgeTracing(mastery_threshold=1.5)


# ────────────────────────────────────────────────────────────────────────────
# MasteryTracker edge cases
# ────────────────────────────────────────────────────────────────────────────

def test_mastery_tracker_empty_skills_raises():
    """Verify MasteryTracker raises ValueError for empty skills list."""
    with pytest.raises(ValueError):
        MasteryTracker(skills=[])


def test_mastery_tracker_single_skill():
    """Verify MasteryTracker works with single skill."""
    tracker = MasteryTracker(skills=["algebra"])

    mastery = tracker.get_mastery()
    assert "algebra" in mastery
    assert mastery["algebra"] == 0.1  # Default p_init


def test_mastery_tracker_unknown_skill_update_raises():
    """Verify MasteryTracker.update() raises KeyError for unknown skill."""
    tracker = MasteryTracker(skills=["algebra", "geometry"])

    with pytest.raises(KeyError):
        tracker.update("calculus", correct=True)


def test_mastery_tracker_unknown_skill_ready_for_raises():
    """Verify MasteryTracker.ready_for() raises KeyError for unknown skill."""
    tracker = MasteryTracker(skills=["algebra"])

    with pytest.raises(KeyError):
        tracker.ready_for("calculus")


def test_mastery_tracker_many_skills():
    """Verify MasteryTracker handles many skills."""
    skills = [f"skill_{i}" for i in range(100)]
    tracker = MasteryTracker(skills=skills)

    assert len(tracker.mastered_skills()) == 0  # None mastered initially
    assert len(tracker.unmastered_skills()) == 100


def test_mastery_tracker_recommend_next_empty_when_all_mastered():
    """Verify recommend_next() returns empty when all skills mastered."""
    tracker = MasteryTracker(skills=["algebra", "geometry"])

    # Master all skills
    for _ in range(100):
        for skill in ["algebra", "geometry"]:
            tracker.update(skill, correct=True)

    recommendations = tracker.recommend_next(n=10)

    assert len(recommendations) == 0


# ────────────────────────────────────────────────────────────────────────────
# PrerequisiteGraph edge cases
# ────────────────────────────────────────────────────────────────────────────

def test_prerequisite_graph_self_loop_raises():
    """Verify PrerequisiteGraph raises ValueError for self-loops."""
    graph = PrerequisiteGraph()

    with pytest.raises(ValueError, match="self-loop"):
        graph.add_edge("algebra", "algebra")


def test_prerequisite_graph_cycle_raises():
    """Verify PrerequisiteGraph raises ValueError for cycles."""
    graph = PrerequisiteGraph()
    graph.add_edge("algebra", "calculus")
    graph.add_edge("calculus", "linear_algebra")

    with pytest.raises(ValueError, match="cycle"):
        graph.add_edge("linear_algebra", "algebra")


def test_prerequisite_graph_two_node_cycle_raises():
    """Verify PrerequisiteGraph detects 2-node cycles."""
    graph = PrerequisiteGraph()
    graph.add_edge("algebra", "geometry")

    with pytest.raises(ValueError, match="cycle"):
        graph.add_edge("geometry", "algebra")


def test_prerequisite_graph_empty():
    """Verify PrerequisiteGraph works when empty."""
    graph = PrerequisiteGraph()

    assert len(graph.prerequisites_for("algebra")) == 0
    assert len(graph.all_prerequisites_for("algebra")) == 0


def test_prerequisite_graph_single_node():
    """Verify PrerequisiteGraph works with single node (no edges)."""
    graph = PrerequisiteGraph()
    graph.add_edge("algebra", "calculus")

    assert len(graph.prerequisites_for("algebra")) == 0
    assert len(graph.prerequisites_for("calculus")) == 1
    assert "algebra" in graph.prerequisites_for("calculus")


def test_prerequisite_graph_complex_dag():
    """Verify PrerequisiteGraph handles complex DAG without cycles."""
    graph = PrerequisiteGraph()
    edges = [
        ("algebra", "calculus"),
        ("algebra", "linear_algebra"),
        ("trigonometry", "calculus"),
        ("calculus", "differential_equations"),
    ]
    graph.add_edges(edges)

    # No exception should be raised
    assert "algebra" in graph.all_prerequisites_for("differential_equations")
    assert "trigonometry" in graph.all_prerequisites_for("differential_equations")


def test_prerequisite_graph_add_edges_rollback_on_cycle():
    """Verify add_edges() doesn't modify graph if any edge creates cycle."""
    graph = PrerequisiteGraph()
    graph.add_edge("algebra", "calculus")

    # Try to add edges including one that would create a cycle
    edges_with_cycle = [
        ("calculus", "linear_algebra"),
        ("linear_algebra", "algebra"),  # This creates a cycle
    ]

    with pytest.raises(ValueError, match="cycle"):
        graph.add_edges(edges_with_cycle)

    # Graph should not have been modified
    assert "linear_algebra" not in graph.all_prerequisites_for("calculus")


# ────────────────────────────────────────────────────────────────────────────
# Validation toggle edge cases
# ────────────────────────────────────────────────────────────────────────────

def test_validation_enabled_rejects_non_integer_ids():
    """Verify validate_inputs=True rejects non-integer user/item IDs."""
    rec = OrchidRecommender(strategy="als", validate_inputs=True, epochs=1)
    data = pd.DataFrame({
        "user_id": ["user_1", "user_2"],
        "item_id": [1, 2],
        "rating": [1.0, 2.0],
    })

    with pytest.raises(ValueError):
        rec.fit(data, rating_col="rating")


def test_validation_disabled_coerces_string_ids():
    """Verify validate_inputs=False allows string IDs (coerced to int)."""
    rec = OrchidRecommender(strategy="als", validate_inputs=False, epochs=1)
    data = pd.DataFrame({
        "user_id": ["1", "2"],
        "item_id": ["10", "11"],
        "rating": [1.0, 2.0],
    })

    # Should not raise (IDs coerced to int)
    rec.fit(data, rating_col="rating")
    assert len(rec.all_users()) == 2
    assert len(rec.all_items()) == 2


# ────────────────────────────────────────────────────────────────────────────
# Numerical boundary edge cases
# ────────────────────────────────────────────────────────────────────────────

def test_ratings_very_small_values():
    """Verify recommender handles very small rating values."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1e-6, 1e-8, 1e-10],
    })

    rec.fit(data, rating_col="rating")
    recs = rec.recommend(user_id=1, top_k=2)

    assert len(recs) > 0


def test_ratings_very_large_values():
    """Verify recommender handles very large rating values."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [10, 11, 12],
        "rating": [1e6, 1e8, 1e10],
    })

    rec.fit(data, rating_col="rating")
    recs = rec.recommend(user_id=1, top_k=2)

    assert len(recs) > 0


def test_user_item_ids_large_integers():
    """Verify recommender handles large integer user/item IDs."""
    rec = OrchidRecommender(strategy="als", epochs=1)
    data = pd.DataFrame({
        "user_id": [1000000, 2000000, 3000000],
        "item_id": [10000000, 11000000, 12000000],
        "rating": [1.0, 2.0, 3.0],
    })

    rec.fit(data, rating_col="rating")
    recs = rec.recommend(user_id=1000000, top_k=2)

    assert len(recs) > 0
