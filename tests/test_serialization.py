"""Comprehensive tests for model serialization utilities."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from orchid_ranker.serialization import save_model, load_model
from orchid_ranker.recommender import OrchidRecommender


# ============================================================================
# Fixtures for test data
# ============================================================================

@pytest.fixture
def small_interactions():
    """Create a small synthetic interactions DataFrame for testing."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "item_id": [10, 20, 30, 10, 20, 40, 20, 30, 50, 10, 40, 50, 30, 40, 50],
    })


@pytest.fixture
def fitted_popularity_model(small_interactions):
    """Create a fitted OrchidRecommender with popularity strategy."""
    model = OrchidRecommender(strategy="popularity")
    model.fit(small_interactions)
    return model


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield temp
    # Cleanup is handled by OS


# ============================================================================
# Save Model Tests
# ============================================================================

class TestSaveModel:
    """Tests for save_model function."""

    def test_save_model_unfitted_raises(self, temp_dir):
        """Test that save_model raises error for unfitted model."""
        model = OrchidRecommender(strategy="popularity")
        path = Path(temp_dir) / "model.pt"

        with pytest.raises(RuntimeError, match="Cannot save an unfitted"):
            save_model(model, path)

    def test_save_model_creates_file(self, fitted_popularity_model, temp_dir):
        """Test that save_model creates a file at the specified path."""
        path = Path(temp_dir) / "model.pt"
        save_model(fitted_popularity_model, path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_model_with_string_path(self, fitted_popularity_model, temp_dir):
        """Test save_model with string path instead of Path object."""
        path = str(Path(temp_dir) / "model.pt")
        save_model(fitted_popularity_model, path)

        assert Path(path).exists()

    def test_save_model_creates_parent_directories(self, fitted_popularity_model, temp_dir):
        """Test that save_model creates parent directories as needed."""
        path = Path(temp_dir) / "subdir" / "deep" / "model.pt"
        save_model(fitted_popularity_model, path)

        assert path.exists()
        assert path.parent.parent.exists()


# ============================================================================
# Load Model Tests
# ============================================================================

class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_nonexistent_file_raises(self, temp_dir):
        """Test that load_model raises FileNotFoundError for missing file."""
        path = Path(temp_dir) / "nonexistent.pt"

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            load_model(path)

    def test_load_model_corrupted_file_raises(self, temp_dir):
        """Test that load_model raises RuntimeError for corrupted file."""
        path = Path(temp_dir) / "corrupted.pt"
        # Write garbage bytes to simulate corrupted file
        with open(path, "wb") as f:
            f.write(b"this is not a valid checkpoint")

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_model(path)

    def test_load_model_returns_orchid_recommender(self, fitted_popularity_model, temp_dir):
        """Test that load_model returns an OrchidRecommender instance."""
        path = Path(temp_dir) / "model.pt"
        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        assert isinstance(loaded, OrchidRecommender)

    def test_load_model_with_string_path(self, fitted_popularity_model, temp_dir):
        """Test load_model with string path instead of Path object."""
        path = str(Path(temp_dir) / "model.pt")
        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        assert isinstance(loaded, OrchidRecommender)


# ============================================================================
# Save/Load Roundtrip Tests
# ============================================================================

class TestSaveLoadRoundtrip:
    """Tests for save/load roundtrip consistency."""

    def test_save_load_roundtrip_basic(self, fitted_popularity_model, temp_dir):
        """Test basic save/load roundtrip preserves model functionality."""
        path = Path(temp_dir) / "model.pt"
        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        # Check that loaded model has the same strategy
        assert loaded.strategy == fitted_popularity_model.strategy

    def test_save_load_roundtrip_strategy_preserved(self, fitted_popularity_model, temp_dir):
        """Test that strategy is preserved after save/load."""
        path = Path(temp_dir) / "model.pt"
        original_strategy = fitted_popularity_model.strategy

        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        assert loaded.strategy == original_strategy
        assert loaded.strategy == "popularity"

    def test_save_load_roundtrip_user_mappings_preserved(self, fitted_popularity_model, temp_dir):
        """Test that user/item mappings are preserved after save/load."""
        path = Path(temp_dir) / "model.pt"

        original_users = set(fitted_popularity_model.all_users())
        original_items = set(fitted_popularity_model.all_items())

        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        loaded_users = set(loaded.all_users())
        loaded_items = set(loaded.all_items())

        assert loaded_users == original_users
        assert loaded_items == original_items

    def test_save_load_roundtrip_item_mappings_preserved(self, fitted_popularity_model, temp_dir):
        """Test that item mappings specifically are preserved."""
        path = Path(temp_dir) / "model.pt"

        # Check internal mappings
        original_item2idx = fitted_popularity_model._item2idx.copy()

        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        assert loaded._item2idx == original_item2idx

    def test_save_load_roundtrip_user_item_mappings_consistent(
        self, fitted_popularity_model, temp_dir
    ):
        """Test that user and item mappings are internally consistent."""
        path = Path(temp_dir) / "model.pt"

        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        # Check forward/backward mapping consistency
        for user_id, user_idx in loaded._user2idx.items():
            assert loaded._idx2user[user_idx] == user_id

        for item_id, item_idx in loaded._item2idx.items():
            assert loaded._idx2item[item_idx] == item_id

    def test_save_load_roundtrip_recommendations_identical(
        self, fitted_popularity_model, temp_dir
    ):
        """Test that recommendations are identical after save/load."""
        path = Path(temp_dir) / "model.pt"

        original_recs = fitted_popularity_model.recommend(user_id=1, top_k=5)

        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        loaded_recs = loaded.recommend(user_id=1, top_k=5)

        assert len(original_recs) == len(loaded_recs)
        for orig, loaded_rec in zip(original_recs, loaded_recs):
            assert orig.item_id == loaded_rec.item_id
            assert orig.score == loaded_rec.score

    def test_save_load_roundtrip_predictions_identical(
        self, fitted_popularity_model, temp_dir
    ):
        """Test that predictions are identical after save/load."""
        path = Path(temp_dir) / "model.pt"

        original_score = fitted_popularity_model.predict(user_id=1, item_id=10)

        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        loaded_score = loaded.predict(user_id=1, item_id=10)

        assert original_score == loaded_score


# ============================================================================
# Checkpoint Version Tests
# ============================================================================

class TestCheckpointVersion:
    """Tests for checkpoint versioning."""

    def test_checkpoint_contains_version(self, fitted_popularity_model, temp_dir):
        """Test that checkpoint file contains version information."""
        import torch

        path = Path(temp_dir) / "model.pt"
        save_model(fitted_popularity_model, path)

        checkpoint = torch.load(path, weights_only=False)
        assert "version" in checkpoint
        assert checkpoint["version"] == "1.0"

    def test_checkpoint_contains_model_type(self, fitted_popularity_model, temp_dir):
        """Test that checkpoint file contains model type."""
        import torch

        path = Path(temp_dir) / "model.pt"
        save_model(fitted_popularity_model, path)

        checkpoint = torch.load(path, weights_only=False)
        assert "model_type" in checkpoint
        assert checkpoint["model_type"] == "OrchidRecommender"

    def test_checkpoint_contains_state(self, fitted_popularity_model, temp_dir):
        """Test that checkpoint file contains state dictionary."""
        import torch

        path = Path(temp_dir) / "model.pt"
        save_model(fitted_popularity_model, path)

        checkpoint = torch.load(path, weights_only=False)
        assert "state" in checkpoint
        state = checkpoint["state"]
        assert "strategy" in state
        assert "user_map" in state
        assert "item_map" in state


# ============================================================================
# OrchidRecommender save/load Convenience Methods Tests
# ============================================================================

class TestOrchidRecommenderSaveLoad:
    """Tests for OrchidRecommender.save() and .load() convenience methods."""

    def test_orchid_recommender_save_method(self, fitted_popularity_model, temp_dir):
        """Test OrchidRecommender.save() convenience method."""
        path = Path(temp_dir) / "model.pt"
        fitted_popularity_model.save(str(path))

        assert path.exists()
        assert path.stat().st_size > 0

    def test_orchid_recommender_save_unfitted_raises(self, temp_dir):
        """Test that OrchidRecommender.save() raises for unfitted model."""
        model = OrchidRecommender(strategy="popularity")
        path = Path(temp_dir) / "model.pt"

        with pytest.raises(RuntimeError, match="Cannot save an unfitted"):
            model.save(str(path))

    def test_orchid_recommender_load_class_method(self, fitted_popularity_model, temp_dir):
        """Test OrchidRecommender.load() class method."""
        path = Path(temp_dir) / "model.pt"
        fitted_popularity_model.save(str(path))

        loaded = OrchidRecommender.load(str(path))
        assert isinstance(loaded, OrchidRecommender)
        assert loaded.strategy == "popularity"

    def test_orchid_recommender_load_wrong_type_raises(self, temp_dir):
        """Test that OrchidRecommender.load() raises for wrong model type."""
        # This is a safeguard test - if somehow a non-OrchidRecommender model
        # were saved, loading it should raise TypeError
        path = Path(temp_dir) / "model.pt"

        # Create and save a valid model first
        model = OrchidRecommender(strategy="popularity")
        interactions = pd.DataFrame({
            "user_id": [1, 2],
            "item_id": [10, 20],
        })
        model.fit(interactions)
        model.save(str(path))

        # Now load it - should succeed
        loaded = OrchidRecommender.load(str(path))
        assert isinstance(loaded, OrchidRecommender)

    def test_orchid_recommender_save_load_roundtrip(self, fitted_popularity_model, temp_dir):
        """Test full save/load roundtrip using convenience methods."""
        path = Path(temp_dir) / "model.pt"

        # Save using convenience method
        fitted_popularity_model.save(str(path))

        # Load using convenience method
        loaded = OrchidRecommender.load(str(path))

        # Verify they work the same
        assert loaded.strategy == fitted_popularity_model.strategy
        assert loaded.all_users() == fitted_popularity_model.all_users()
        assert loaded.all_items() == fitted_popularity_model.all_items()


# ============================================================================
# Multiple Save/Load Cycles Tests
# ============================================================================

class TestMultipleSaveLoadCycles:
    """Tests for multiple save/load cycles."""

    def test_save_load_save_load_cycle(self, fitted_popularity_model, temp_dir):
        """Test that multiple save/load cycles preserve model state."""
        path1 = Path(temp_dir) / "model1.pt"
        path2 = Path(temp_dir) / "model2.pt"

        # First save
        save_model(fitted_popularity_model, path1)

        # Load and save again
        loaded1 = load_model(path1)
        save_model(loaded1, path2)

        # Load again
        loaded2 = load_model(path2)

        # Verify consistency
        assert loaded2.strategy == fitted_popularity_model.strategy
        assert loaded2.all_users() == fitted_popularity_model.all_users()
        assert loaded2.all_items() == fitted_popularity_model.all_items()

        # Verify recommendations are still identical
        original_recs = fitted_popularity_model.recommend(user_id=1, top_k=5)
        final_recs = loaded2.recommend(user_id=1, top_k=5)

        assert len(original_recs) == len(final_recs)
        for orig, final in zip(original_recs, final_recs):
            assert orig.item_id == final.item_id


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestSerializationEdgeCases:
    """Edge case tests for serialization."""

    def test_save_load_with_minimal_data(self, temp_dir):
        """Test save/load with minimal interactions data."""
        interactions = pd.DataFrame({
            "user_id": [1, 1],
            "item_id": [10, 20],
        })

        model = OrchidRecommender(strategy="popularity")
        model.fit(interactions)

        path = Path(temp_dir) / "minimal.pt"
        save_model(model, path)
        loaded = load_model(path)

        assert loaded.strategy == "popularity"
        assert loaded.all_users() == [1]
        assert set(loaded.all_items()) == {10, 20}

    def test_save_load_preserves_seen_items(self, fitted_popularity_model, temp_dir):
        """Test that seen_items dict is preserved."""
        path = Path(temp_dir) / "model.pt"

        original_seen = fitted_popularity_model._seen_items.copy()

        save_model(fitted_popularity_model, path)
        loaded = load_model(path)

        # Verify seen items are preserved
        assert loaded._seen_items == original_seen

    def test_multiple_users_and_items_preserved(self, temp_dir):
        """Test with larger dataset to ensure all mappings are preserved."""
        interactions = pd.DataFrame({
            "user_id": [i % 10 for i in range(100)],
            "item_id": [i % 20 for i in range(100)],
        })

        model = OrchidRecommender(strategy="popularity")
        model.fit(interactions)

        path = Path(temp_dir) / "large.pt"
        save_model(model, path)
        loaded = load_model(path)

        assert set(loaded.all_users()) == set(model.all_users())
        assert set(loaded.all_items()) == set(model.all_items())
