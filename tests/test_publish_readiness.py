"""Publish-readiness tests covering all pre-publish fixes.

Exercises lazy imports, serialization safety, ALS label clamping,
non-finite rating rejection, pre-fit error handling, LinUCB numerical
stability, NeuralMF user broadcast, ECE ValueError, learning_gain
edge cases, and CurriculumRecommender behaviour.
"""
import math

import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Torch-free imports always work
# ---------------------------------------------------------------------------
from orchid_ranker import (
    __version__,
    BayesianKnowledgeTracing,
    MasteryTracker,
    ForgettingCurve,
    PrerequisiteGraph,
    CurriculumRecommender,
    learning_gain,
    knowledge_coverage,
    curriculum_adherence,
    difficulty_appropriateness,
    engagement_score,
    EducationalReport,
)
from orchid_ranker._compat import torch_available, require_torch
from orchid_ranker.evaluation import expected_calibration_error

# Torch-dependent imports -- skip the entire module if torch is absent.
torch = pytest.importorskip("torch")
from orchid_ranker import OrchidRecommender, save_model, load_model  # noqa: E402


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture()
def small_interactions():
    """Minimal interaction DataFrame with 25 users x 8 items.

    25 users ensures that user_knn (default k=20) has enough neighbours.
    """
    rng = np.random.RandomState(42)
    rows = []
    for uid in range(25):
        # each user interacts with 4-6 items
        n_items = rng.randint(4, 7)
        items = rng.choice(8, size=n_items, replace=False)
        for iid in items:
            rows.append({"user_id": uid, "item_id": iid, "rating": rng.rand()})
    return pd.DataFrame(rows)


@pytest.fixture()
def small_interactions_binary():
    """Interaction DataFrame with binary (0/1) labels."""
    rng = np.random.RandomState(7)
    rows = []
    for uid in range(5):
        items = rng.choice(8, size=5, replace=False)
        for iid in items:
            rows.append({"user_id": uid, "item_id": iid, "rating": float(rng.randint(0, 2))})
    return pd.DataFrame(rows)


@pytest.fixture()
def item_features_8():
    """Random feature matrix for 8 items, 4 features."""
    rng = np.random.RandomState(99)
    return rng.rand(8, 4).astype(np.float32)


# ===================================================================
# 1. Lazy imports / torch optional
# ===================================================================

class TestLazyImports:
    """Verify __getattr__-based lazy import infrastructure."""

    def test_import_succeeds(self):
        import orchid_ranker  # noqa: F811
        # Module should have loaded without error
        assert hasattr(orchid_ranker, "__version__")

    def test_torch_free_symbols_available(self):
        """Torch-free symbols should be importable at the top level."""
        assert BayesianKnowledgeTracing is not None
        assert MasteryTracker is not None
        assert ForgettingCurve is not None
        assert PrerequisiteGraph is not None
        assert CurriculumRecommender is not None
        assert callable(learning_gain)
        assert callable(knowledge_coverage)
        assert callable(curriculum_adherence)
        assert callable(difficulty_appropriateness)
        assert callable(engagement_score)
        assert EducationalReport is not None

    def test_torch_dependent_symbols_accessible(self):
        """OrchidRecommender etc. are reachable via lazy import."""
        assert OrchidRecommender is not None
        assert callable(save_model)
        assert callable(load_model)

    def test_version_is_030(self):
        assert __version__ == "0.3.0"

    def test_compat_torch_available(self):
        """torch_available() returns True in dev environment."""
        assert torch_available() is True

    def test_compat_require_torch_no_raise(self):
        """require_torch() should not raise when torch is installed."""
        require_torch("test feature")  # should not raise

    def test_unknown_attr_raises(self):
        import orchid_ranker
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = orchid_ranker.totally_bogus_name_xyz


# ===================================================================
# 2. Serialization security -- safe state dicts instead of pickle
# ===================================================================

class TestSerializationRoundtrip:
    """Save/load roundtrip for every built-in strategy."""

    DETERMINISTIC_STRATEGIES = ["als", "explicit_mf", "popularity", "user_knn", "neural_mf"]

    @pytest.fixture()
    def _fitted_model(self, small_interactions, item_features_8, request):
        """Return a fitted OrchidRecommender for the parameterised strategy."""
        strategy = request.param
        kwargs = {}
        fit_kwargs = {"rating_col": "rating"}
        if strategy == "linucb":
            kwargs["alpha"] = 1.0
            fit_kwargs["item_features"] = item_features_8
        if strategy == "neural_mf":
            kwargs["epochs"] = 2
        if strategy == "als":
            kwargs["epochs"] = 2
        rec = OrchidRecommender(strategy=strategy, **kwargs)
        rec.fit(small_interactions, **fit_kwargs)
        return rec

    @pytest.mark.parametrize(
        "_fitted_model",
        ["als", "explicit_mf", "popularity", "random", "neural_mf", "user_knn", "linucb"],
        indirect=True,
    )
    def test_roundtrip_predictions_match(self, _fitted_model, tmp_path):
        """Save then load; predictions on the first user must match."""
        rec = _fitted_model
        path = tmp_path / "model.pt"
        save_model(rec, path)
        loaded = load_model(path)

        # Structural checks
        assert loaded.strategy == rec.strategy
        assert set(loaded.all_users()) == set(rec.all_users())
        assert set(loaded.all_items()) == set(rec.all_items())

        # Prediction check -- score every item for user 0
        user_id = rec.all_users()[0]
        for item_id in rec.all_items():
            orig = rec.predict(user_id, item_id)
            rest = loaded.predict(user_id, item_id)
            # random strategy is non-deterministic; skip value check
            if rec.strategy != "random":
                assert math.isclose(orig, rest, rel_tol=1e-5, abs_tol=1e-6), (
                    f"strategy={rec.strategy} user={user_id} item={item_id}: "
                    f"{orig} != {rest}"
                )

    @pytest.mark.parametrize(
        "_fitted_model",
        ["als", "explicit_mf", "popularity", "random", "neural_mf", "user_knn", "linucb"],
        indirect=True,
    )
    def test_checkpoint_has_baseline_data_key(self, _fitted_model, tmp_path):
        """Checkpoint state must use 'baseline_data' (not legacy 'baseline_object')."""
        path = tmp_path / "model.pt"
        save_model(_fitted_model, path)
        checkpoint = torch.load(path, weights_only=False)
        state = checkpoint["state"]
        assert "baseline_data" in state, (
            f"strategy={_fitted_model.strategy}: checkpoint missing 'baseline_data'"
        )
        assert "baseline_object" not in state

    def test_save_unfitted_raises(self):
        rec = OrchidRecommender(strategy="als")
        with pytest.raises(RuntimeError, match="unfitted|not been fit"):
            save_model(rec, "/tmp/should_not_exist.pt")


# ===================================================================
# 3. ALS label clamping
# ===================================================================

class TestALSLabelClamping:
    """ALS baseline now clamps labels to [0, 1]."""

    def test_ratings_above_one_do_not_crash(self):
        """Fit with ratings in 1-5 range should succeed (clamped internally)."""
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [0, 1, 0, 2, 1, 2],
            "rating":  [5.0, 3.0, 4.0, 2.0, 1.0, 5.0],
        })
        rec = OrchidRecommender(strategy="als", epochs=2)
        rec.fit(df, rating_col="rating")
        # Should produce a valid prediction
        score = rec.predict(0, 0)
        assert np.isfinite(score)

    def test_ratings_in_zero_one_work(self):
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 1],
            "rating":  [0.0, 1.0, 0.5, 0.8],
        })
        rec = OrchidRecommender(strategy="als", epochs=2)
        rec.fit(df, rating_col="rating")
        score = rec.predict(0, 0)
        assert np.isfinite(score)


# ===================================================================
# 4. Non-finite rating rejection
# ===================================================================

class TestNonFiniteRatingRejection:
    """fit() must reject inf and NaN ratings with ValueError."""

    def test_inf_rating_raises(self):
        df = pd.DataFrame({
            "user_id": [0, 0, 1],
            "item_id": [0, 1, 0],
            "rating":  [1.0, float("inf"), 0.5],
        })
        rec = OrchidRecommender(strategy="als")
        with pytest.raises(ValueError, match="non-finite"):
            rec.fit(df, rating_col="rating")

    def test_neg_inf_rating_raises(self):
        df = pd.DataFrame({
            "user_id": [0, 0, 1],
            "item_id": [0, 1, 0],
            "rating":  [1.0, float("-inf"), 0.5],
        })
        rec = OrchidRecommender(strategy="als")
        with pytest.raises(ValueError, match="non-finite"):
            rec.fit(df, rating_col="rating")

    def test_nan_rating_raises(self):
        df = pd.DataFrame({
            "user_id": [0, 0, 1],
            "item_id": [0, 1, 0],
            "rating":  [1.0, float("nan"), 0.5],
        })
        rec = OrchidRecommender(strategy="als")
        # NaN may be caught by input validation as "null values" or by
        # the non-finite check -- either way, a ValueError is raised.
        with pytest.raises(ValueError, match="non-finite|null"):
            rec.fit(df, rating_col="rating")


# ===================================================================
# 5. Pre-fit error handling
# ===================================================================

class TestPreFitErrors:
    """predict/recommend before fit must raise RuntimeError."""

    def test_predict_before_fit_raises(self):
        rec = OrchidRecommender(strategy="als")
        with pytest.raises(RuntimeError, match="not been fit"):
            rec.predict(0, 0)

    def test_recommend_before_fit_raises(self):
        rec = OrchidRecommender(strategy="als")
        with pytest.raises(RuntimeError, match="not been fit"):
            rec.recommend(0)


# ===================================================================
# 6. LinUCB numerical stability
# ===================================================================

class TestLinUCBStability:
    """Quadratic form in LinUCB.infer is clamped to avoid negative sqrt."""

    def test_random_features_finite_scores(self, small_interactions, item_features_8):
        rec = OrchidRecommender(strategy="linucb", alpha=2.0)
        rec.fit(small_interactions, rating_col="rating", item_features=item_features_8)
        recs = rec.recommend(0, top_k=3)
        for r in recs:
            assert np.isfinite(r.score), f"LinUCB score is not finite: {r.score}"

    def test_high_alpha_finite_scores(self, small_interactions, item_features_8):
        """Even with very large alpha, scores should remain finite."""
        rec = OrchidRecommender(strategy="linucb", alpha=100.0)
        rec.fit(small_interactions, rating_col="rating", item_features=item_features_8)
        score = rec.predict(0, 0)
        assert np.isfinite(score)


# ===================================================================
# 7. NeuralMF user broadcast
# ===================================================================

class TestNeuralMFBroadcast:
    """infer() must broadcast a single user_id across multiple items."""

    def test_single_user_multiple_items(self, small_interactions):
        rec = OrchidRecommender(strategy="neural_mf", epochs=2)
        rec.fit(small_interactions, rating_col="rating")
        # Score several items for user 0
        items = rec.all_items()
        scores = []
        for iid in items:
            scores.append(rec.predict(0, iid))
        assert len(scores) == len(items)
        assert all(np.isfinite(s) for s in scores)

    def test_recommend_returns_correct_count(self, small_interactions):
        rec = OrchidRecommender(strategy="neural_mf", epochs=2)
        rec.fit(small_interactions, rating_col="rating")
        recs = rec.recommend(0, top_k=3, filter_seen=False)
        assert len(recs) == 3
        for r in recs:
            assert np.isfinite(r.score)


# ===================================================================
# 8. ECE assert -> ValueError
# ===================================================================

class TestECEValueError:
    """expected_calibration_error raises ValueError on mismatched shapes."""

    def test_mismatched_shapes_raises_valueerror(self):
        preds = np.array([0.5, 0.6, 0.7])
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="same shape"):
            expected_calibration_error(preds, labels)

    def test_matched_shapes_works(self):
        preds = np.array([0.5, 0.6, 0.7])
        labels = np.array([0, 1, 1])
        result = expected_calibration_error(preds, labels)
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_empty_arrays_return_zero(self):
        result = expected_calibration_error(np.array([]), np.array([]))
        assert result == 0.0


# ===================================================================
# 9. learning_gain edge cases
# ===================================================================

class TestLearningGainEdgeCases:
    """learning_gain with negative gain, perfect pre-score, and normal case."""

    def test_negative_gain(self):
        """post < pre yields a negative value."""
        result = learning_gain(pre_score=0.8, post_score=0.5)
        assert result < 0.0
        # (0.5 - 0.8) / (1.0 - 0.8) = -0.3 / 0.2 = -1.5
        assert math.isclose(result, -1.5, rel_tol=1e-9)

    def test_pre_score_one_returns_zero(self):
        """When pre_score == 1.0, denominator is zero -> returns 0.0."""
        result = learning_gain(pre_score=1.0, post_score=0.9)
        assert result == 0.0

    def test_normal_case(self):
        """Standard normalized gain calculation."""
        result = learning_gain(pre_score=0.4, post_score=0.7)
        expected = (0.7 - 0.4) / (1.0 - 0.4)  # 0.3 / 0.6 = 0.5
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_perfect_improvement(self):
        """Pre 0.0, post 1.0 -> gain = 1.0."""
        result = learning_gain(pre_score=0.0, post_score=1.0)
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_no_change(self):
        """Same pre and post -> gain = 0.0."""
        result = learning_gain(pre_score=0.5, post_score=0.5)
        assert result == 0.0


# ===================================================================
# 10. CurriculumRecommender behaviour
# ===================================================================

class TestCurriculumRecommender:
    """CurriculumRecommender respects prerequisite ordering strictly."""

    def test_linear_chain_mastered_a(self):
        """Graph a->b->c, mastered={'a'} => recommend only ['b']."""
        graph = PrerequisiteGraph([("a", "b"), ("b", "c")])
        rec = CurriculumRecommender(graph)
        result = rec.recommend({"a"}, n=5)
        # 'b' is available (prereq 'a' met), 'c' is NOT (prereq 'b' not met)
        assert result == ["b"]

    def test_linear_chain_mastered_ab(self):
        """Graph a->b->c, mastered={'a','b'} => recommend only ['c']."""
        graph = PrerequisiteGraph([("a", "b"), ("b", "c")])
        rec = CurriculumRecommender(graph)
        result = rec.recommend({"a", "b"}, n=5)
        assert result == ["c"]

    def test_linear_chain_nothing_mastered(self):
        """Graph a->b->c, mastered={} => recommend only ['a']."""
        graph = PrerequisiteGraph([("a", "b"), ("b", "c")])
        rec = CurriculumRecommender(graph)
        result = rec.recommend(set(), n=5)
        assert result == ["a"]

    def test_diamond_graph(self):
        """Diamond: a->{b,c}->d, mastered={'a'} => ['b','c'] (not 'd')."""
        graph = PrerequisiteGraph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        rec = CurriculumRecommender(graph)
        result = rec.recommend({"a"}, n=5)
        assert sorted(result) == ["b", "c"]
        assert "d" not in result

    def test_all_mastered_returns_empty(self):
        graph = PrerequisiteGraph([("a", "b")])
        rec = CurriculumRecommender(graph)
        result = rec.recommend({"a", "b"}, n=5)
        assert result == []

    def test_difficulty_ordering(self):
        """With difficulty map, easier skills come first."""
        graph = PrerequisiteGraph([("a", "b"), ("a", "c")])
        difficulty = {"b": 0.8, "c": 0.3}
        rec = CurriculumRecommender(graph, difficulty_map=difficulty)
        result = rec.recommend({"a"}, n=5)
        # 'c' is easier (0.3) than 'b' (0.8), so 'c' should come first
        assert result == ["c", "b"]


# ===================================================================
# Additional integration: predict_many roundtrip
# ===================================================================

class TestPredictMany:
    """Verify predict_many consistency across strategies."""

    @pytest.mark.parametrize("strategy", ["als", "explicit_mf", "popularity", "user_knn"])
    def test_predict_many_matches_predict(self, small_interactions, strategy):
        kwargs = {}
        if strategy == "als":
            kwargs["epochs"] = 2
        rec = OrchidRecommender(strategy=strategy, **kwargs)
        rec.fit(small_interactions, rating_col="rating")
        users = rec.all_users()[:2]
        items = rec.all_items()[:2]
        u_ids = [users[0], users[1]]
        i_ids = [items[0], items[1]]
        many = rec.predict_many(u_ids, i_ids)
        assert many.shape == (2,)
        for k in range(2):
            single = rec.predict(u_ids[k], i_ids[k])
            assert math.isclose(float(many[k]), single, rel_tol=1e-4, abs_tol=1e-5)
