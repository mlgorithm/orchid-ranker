import logging

import pandas as pd
import pytest

from orchid_ranker import SUPPORTED_STRATEGIES, OrchidRecommender, configure_logging


def _dataset():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "item_id": [10, 11, 12, 10, 13, 12],
            "label": [1, 0, 1, 1, 0, 1],
        }
    )


def test_popularity_recommender_round_trip():
    data = _dataset()
    rec = OrchidRecommender(strategy="popularity").fit(data, rating_col="label")

    output = rec.recommend(user_id=1, top_k=3)
    assert output, "expected at least one recommendation"
    assert all(hasattr(item, "item_id") for item in output)


def test_als_predicts_known_pair():
    data = _dataset()
    rec = OrchidRecommender(strategy="als", epochs=1)
    rec.fit(data, rating_col="label")
    score = rec.predict(user_id=1, item_id=10)
    assert 0.0 <= score <= 1.0


def test_linucb_requires_item_features():
    data = _dataset()
    rec = OrchidRecommender(strategy="linucb")
    try:
        rec.fit(data, rating_col="label")
    except ValueError as exc:  # expected path
        assert "item_features" in str(exc).lower()
    else:
        raise AssertionError("linucb strategy should require item_features")


def test_linucb_with_features_scores_items():
    data = _dataset()
    # provide simple per-item features (identity matrix)
    item_ids = sorted(data.item_id.unique())
    features = pd.DataFrame({
        "item_id": item_ids,
        "f1": [1.0] * len(item_ids),
    })
    matrix = features[["f1"]].to_numpy(dtype="float32")

    rec = OrchidRecommender(strategy="linucb", alpha=0.5)
    rec.fit(data, rating_col="label", item_features=matrix)
    suggestions = rec.recommend(user_id=1, top_k=2)
    assert suggestions, "expected non-empty slate"
    assert all(isinstance(item.item_id, int) for item in suggestions)


def test_cold_start_user_raises_key_error():
    data = _dataset()
    rec = OrchidRecommender(strategy="popularity").fit(data, rating_col="label")
    try:
        rec.recommend(user_id=99, top_k=3)
    except KeyError:
        pass
    else:
        raise AssertionError("Unknown users should raise KeyError")


def test_neural_mf_recommender_outputs_scores():
    data = _dataset()
    rec = OrchidRecommender(strategy="neural_mf", epochs=2, emb_dim=8, hidden=(16,))
    rec.fit(data, rating_col="label")
    slate = rec.recommend(user_id=1, top_k=2)
    assert slate
    assert all(0.0 <= r.score <= 1.0 for r in slate)


def test_from_interactions_default_auto_fits():
    data = _dataset()
    rec = OrchidRecommender.from_interactions(data, rating_col="label", epochs=1)
    assert rec.is_fitted
    assert rec.recommend(user_id=1, top_k=1)


def test_recommend_respects_candidate_item_ids():
    data = _dataset()
    rec = OrchidRecommender(strategy="popularity").fit(data, rating_col="label")

    slate = rec.recommend(
        user_id=1,
        top_k=5,
        filter_seen=False,
        candidate_item_ids=[13, 999, 10, 13],
    )
    assert [r.item_id for r in slate] == [10, 13]

    unseen_slate = rec.recommend(
        user_id=1,
        top_k=5,
        filter_seen=True,
        candidate_item_ids=[10, 13],
    )
    assert [r.item_id for r in unseen_slate] == [13]


def test_baseline_rank_respects_candidate_item_ids():
    data = _dataset()
    rec = OrchidRecommender(strategy="popularity").fit(data, rating_col="label")

    slate = rec.baseline_rank(user_id=1, top_k=5, candidate_item_ids=[10, 13])
    assert [r.item_id for r in slate] == [13]


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        OrchidRecommender(strategy="does_not_exist")


def test_supported_strategies_contains_user_knn():
    assert "user_knn" in SUPPORTED_STRATEGIES


def test_user_knn_recommender_returns_items():
    data = _dataset()
    rec = OrchidRecommender(strategy="user_knn", k=2)
    rec.fit(data, rating_col="label")
    slate = rec.recommend(user_id=1, top_k=2)
    assert slate, "expected user_knn to produce a slate"
    assert all(isinstance(item.item_id, int) for item in slate)


def test_implicit_als_comparison():
    pytest.importorskip("implicit")
    data = _dataset()
    rec = OrchidRecommender(strategy="implicit_als", factors=8, iterations=5)
    rec.fit(data, rating_col="label")
    slate = rec.recommend(user_id=1, top_k=2)
    assert slate


def test_implicit_bpr_handles_binary_feedback():
    pytest.importorskip("implicit")
    data = _dataset()
    rec = OrchidRecommender(strategy="implicit_bpr", factors=8, iterations=5)
    rec.fit(data, rating_col="label")
    slate = rec.recommend(user_id=1, top_k=2)
    assert slate


def test_validation_rejects_non_integer_ids():
    data = _dataset().copy()
    data["user_id"] = data["user_id"].astype(str)
    rec = OrchidRecommender(strategy="als", validate_inputs=True, epochs=1)
    with pytest.raises((ValueError, TypeError)):
        rec.fit(data, rating_col="label")


def test_validation_toggle_allows_best_effort():
    data = _dataset().copy()
    data["user_id"] = data["user_id"].astype(str)
    rec = OrchidRecommender(strategy="als", validate_inputs=False, epochs=1)
    rec.fit(data, rating_col="label")
    assert rec.recommend(user_id=1, top_k=1)


def test_fit_emits_info_log(caplog):
    data = _dataset()
    rec = OrchidRecommender(strategy="popularity")
    with caplog.at_level("INFO", logger="orchid_ranker.recommender"):
        rec.fit(data, rating_col="label")
    assert any("fitted strategy" in message for message in caplog.messages)


def test_configure_logging_returns_logger():
    logger = configure_logging(level="DEBUG", logger_name="orchid_ranker.enterprise")
    assert logger.name == "orchid_ranker.enterprise"
    assert logger.level == logging.DEBUG
    assert logger.handlers, "configure_logging should attach at least one handler"


# ---------------------------------------------------------------------------
# as_streaming() error conditions
# ---------------------------------------------------------------------------
def test_as_streaming_unfitted_raises():
    """as_streaming on an unfitted recommender raises RuntimeError."""
    rec = OrchidRecommender(strategy="neural_mf", epochs=1, emb_dim=8, hidden=(16,))
    with pytest.raises(RuntimeError, match="fitted"):
        rec.as_streaming()


def test_as_streaming_non_neural_strategy_raises():
    """as_streaming on a strategy without a neural tower raises RuntimeError."""
    data = _dataset()
    rec = OrchidRecommender(strategy="popularity")
    rec.fit(data, rating_col="label")
    with pytest.raises(RuntimeError, match="tower"):
        rec.as_streaming()


def test_as_streaming_als_strategy_raises_clear_error():
    """ALS has a torch model, but not the streaming tower protocol."""
    data = _dataset()
    rec = OrchidRecommender(strategy="als", epochs=1)
    rec.fit(data, rating_col="label")
    with pytest.raises(RuntimeError, match="neural_mf"):
        rec.as_streaming()


def test_as_streaming_neural_mf_uses_external_ids():
    """The OrchidRecommender bridge accepts original user/item IDs, not row indexes."""
    data = _dataset()
    rec = OrchidRecommender.from_interactions(
        data,
        strategy="neural_mf",
        rating_col="label",
        epochs=1,
        emb_dim=8,
        hidden=(16,),
    )

    streamer = rec.as_streaming(lr=0.05)
    before = streamer.rank(user_id=1, candidate_item_ids=[10, 11, 13, 999], top_k=2)
    assert before
    assert {item_id for item_id, _score in before}.issubset({10, 11, 13})

    update = streamer.observe(user_id=1, item_id=10, correct=True)
    after = streamer.rank(user_id=1, candidate_item_ids=[10, 11, 13], top_k=2)

    assert update["p_known"] >= 0.0
    assert after
    assert streamer.updates_for(1) == 1
