import logging

import pandas as pd
import pytest

from orchid_ranker import OrchidRecommender, configure_logging, SUPPORTED_STRATEGIES


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
    with pytest.raises(ValueError):
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
