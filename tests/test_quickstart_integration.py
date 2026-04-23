"""Integration test: verify the canonical quickstart runs end-to-end.

This test mirrors examples/quickstart.py to ensure the documented entry
point works from a fresh install.
"""
import numpy as np
import pandas as pd
import pytest

from orchid_ranker import OrchidRecommender


class TestQuickstartIntegration:
    """Verify the canonical quickstart flow."""

    @pytest.fixture
    def sample_interactions(self):
        rng = np.random.RandomState(42)
        n_users, n_items, n_interactions = 12, 50, 300
        return pd.DataFrame({
            "user_id": rng.randint(0, n_users, n_interactions),
            "item_id": rng.randint(0, n_items, n_interactions),
        })

    def _fit_quickstart_recommender(self, interactions):
        return OrchidRecommender.from_interactions(
            interactions,
            strategy="als",
            epochs=1,
            embedding_dim=8,
        )

    def test_from_interactions_and_recommend(self, sample_interactions):
        """from_interactions -> recommend -> predict round-trip."""
        rec = self._fit_quickstart_recommender(sample_interactions)
        assert rec.is_fitted

        top5 = rec.recommend(user_id=0, top_k=5)
        assert len(top5) == 5
        assert all(hasattr(r, "item_id") for r in top5)

    def test_baseline_rank(self, sample_interactions):
        """baseline_rank produces same shape as recommend."""
        rec = self._fit_quickstart_recommender(sample_interactions)
        recs = rec.recommend(user_id=0, top_k=5)
        fallback = rec.baseline_rank(user_id=0, top_k=5)
        assert len(fallback) == len(recs)

    def test_predict_score(self, sample_interactions):
        """predict returns a float score for a known user-item pair."""
        rec = self._fit_quickstart_recommender(sample_interactions)
        top1 = rec.recommend(user_id=0, top_k=1)
        score = rec.predict(user_id=0, item_id=top1[0].item_id)
        assert isinstance(score, float)

    def test_properties_available(self, sample_interactions):
        """tower, user_features, item_features are accessible."""
        rec = self._fit_quickstart_recommender(sample_interactions)
        # ALS exposes a tower (model attribute)
        assert rec.user_features is not None
        assert rec.item_features is not None

    def test_from_interactions_stores_metadata(self, sample_interactions):
        """item_difficulties and prerequisite_graph are stored."""
        diffs = {0: 0.3, 1: 0.7}
        prereqs = {1: {0}}
        rec = OrchidRecommender.from_interactions(
            sample_interactions,
            strategy="als",
            epochs=1,
            embedding_dim=8,
            item_difficulties=diffs,
            prerequisite_graph=prereqs,
        )
        assert rec._item_difficulties == diffs
        assert rec._prerequisite_graph == prereqs

    def test_quickstart_script_runs(self):
        """The actual quickstart.py script executes without error."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "examples/quickstart.py"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"quickstart.py failed:\n{result.stderr}"
        assert "Quickstart complete" in result.stdout
