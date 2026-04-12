import numpy as np
import pytest

from orchid_ranker.evaluation import (
    RankingReport,
    average_precision,
    expected_calibration_error,
    evaluate_recommendations,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_ranking_metrics_basic():
    slate = [1, 2, 3]
    relevant = [2, 3, 4]
    assert precision_at_k(slate, relevant, 2) == 0.5
    assert recall_at_k(slate, relevant, 3) == pytest.approx(2 / 3)


def test_average_precision():
    slate = [1, 2, 3, 4]
    relevant = [2, 4]
    assert average_precision(slate, relevant, 4) == pytest.approx((1/2 + 2/4) / 2)


def test_ndcg():
    slate = [1, 2, 3]
    rel = {2: 1.0, 3: 0.5}
    score = ndcg_at_k(slate, rel, 3)
    assert 0 <= score <= 1


def test_expected_calibration_error():
    preds = np.array([0.1, 0.4, 0.9])
    labels = np.array([0, 1, 1])
    ece = expected_calibration_error(preds, labels, bins=3)
    assert 0 <= ece <= 1


def test_evaluate_recommendations():
    recs = {1: [1, 2, 3], 2: [2, 3, 4]}
    rel = {1: [2, 3], 2: [4]}
    report = evaluate_recommendations(recs, rel)
    assert isinstance(report, RankingReport)
    assert 0 <= report.precision_at_5 <= 1
    assert 0 <= report.ndcg_at_10 <= 1


def test_evaluate_recommendations_counts_missing_users_as_zero():
    recs = {1: [1]}
    rel = {1: [1], 2: [2]}
    report = evaluate_recommendations(recs, rel, k_prec=1, k_rec=1, k_map=1, k_ndcg=1)
    assert report.precision == pytest.approx(0.5)
    assert report.recall == pytest.approx(0.5)
    assert report.map == pytest.approx(0.5)
    assert report.ndcg == pytest.approx(0.5)
