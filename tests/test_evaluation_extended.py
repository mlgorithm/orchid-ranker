"""Extended tests for evaluation metrics."""
import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from orchid_ranker.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision,
    expected_calibration_error,
    evaluate_recommendations,
    RankingReport,
)


class TestPrecisionAtK:
    """Test precision_at_k metric."""

    def test_perfect_ranking(self):
        """Test precision with perfect ranking."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3, 4, 5]
        p = precision_at_k(recommended, relevant, k=5)
        assert p == 1.0

    def test_no_relevant_items(self):
        """Test precision with no relevant items."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [10, 11, 12]
        p = precision_at_k(recommended, relevant, k=5)
        assert p == 0.0

    def test_partial_relevance(self):
        """Test precision with partially relevant items."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5]
        p = precision_at_k(recommended, relevant, k=5)
        assert p == 0.6  # 3/5

    def test_k_larger_than_list(self):
        """Test when k is larger than recommendation list."""
        recommended = [1, 2, 3]
        relevant = [1, 2, 3, 4, 5]
        p = precision_at_k(recommended, relevant, k=10)
        # Precision@k = hits / k, so 3/10 = 0.3
        assert p == pytest.approx(0.3)

    def test_k_zero_returns_zero(self):
        """Test that k=0 returns 0."""
        recommended = [1, 2, 3]
        relevant = [1, 2, 3]
        p = precision_at_k(recommended, relevant, k=0)
        assert p == 0.0

    def test_negative_k_returns_zero(self):
        """Test that negative k returns 0."""
        recommended = [1, 2, 3]
        relevant = [1, 2, 3]
        p = precision_at_k(recommended, relevant, k=-1)
        assert p == 0.0


class TestRecallAtK:
    """Test recall_at_k metric."""

    def test_all_relevant_returned(self):
        """Test recall when all relevant items returned."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3]
        r = recall_at_k(recommended, relevant, k=5)
        assert r == 1.0

    def test_partial_recall(self):
        """Test recall with partial coverage."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3, 4, 5, 6, 7]
        r = recall_at_k(recommended, relevant, k=5)
        assert r == pytest.approx(5 / 7)

    def test_empty_relevant_returns_zero(self):
        """Test recall with empty relevant set."""
        recommended = [1, 2, 3]
        relevant = []
        r = recall_at_k(recommended, relevant, k=3)
        assert r == 0.0

    def test_no_overlap(self):
        """Test recall when there's no overlap."""
        recommended = [1, 2, 3]
        relevant = [4, 5, 6]
        r = recall_at_k(recommended, relevant, k=3)
        assert r == 0.0


class TestNDCGAtK:
    """Test ndcg_at_k metric."""

    def test_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        recommended = [1, 2, 3]
        relevant = {1: 1.0, 2: 1.0, 3: 1.0}
        ndcg = ndcg_at_k(recommended, relevant, k=3)
        assert ndcg == pytest.approx(1.0)

    def test_with_graded_relevance(self):
        """Test NDCG with graded relevance scores."""
        recommended = [1, 2, 3, 4]
        relevant = {1: 2.0, 2: 1.0, 3: 0.0, 4: 1.0}
        ndcg = ndcg_at_k(recommended, relevant, k=4)
        assert 0.0 <= ndcg <= 1.0

    def test_k_zero_returns_zero(self):
        """Test that k=0 returns 0."""
        recommended = [1, 2, 3]
        relevant = {1: 1.0, 2: 1.0}
        ndcg = ndcg_at_k(recommended, relevant, k=0)
        assert ndcg == 0.0

    def test_empty_relevant_returns_zero(self):
        """Test NDCG with empty relevant dict."""
        recommended = [1, 2, 3]
        relevant = {}
        ndcg = ndcg_at_k(recommended, relevant, k=3)
        assert ndcg == 0.0

    def test_worst_ranking(self):
        """Test NDCG with worst possible ranking."""
        recommended = [4, 5, 6, 1, 2, 3]
        relevant = {1: 1.0, 2: 1.0, 3: 1.0}
        ndcg_worst = ndcg_at_k(recommended, relevant, k=6)

        # Should be less than perfect
        ndcg_best = 1.0
        assert ndcg_worst < ndcg_best


class TestAveragePrecision:
    """Test average_precision metric."""

    def test_perfect_ranking(self):
        """Test MAP with perfect ranking."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3]
        ap = average_precision(recommended, relevant, k=5)
        assert ap == pytest.approx(1.0)

    def test_no_relevant_items(self):
        """Test MAP with no relevant items."""
        recommended = [1, 2, 3]
        relevant = [4, 5, 6]
        ap = average_precision(recommended, relevant, k=3)
        assert ap == 0.0

    def test_one_relevant_at_end(self):
        """Test MAP with one relevant item at the end."""
        recommended = [4, 5, 6, 1]
        relevant = [1, 2, 3]
        ap = average_precision(recommended, relevant, k=4)
        # Only hit at position 4, so precision at that point is 1/4
        # Average precision = (1/4) / 3 relevant items = 1/12
        assert ap == pytest.approx(1 / 12)

    def test_multiple_relevant_items(self):
        """Test MAP with multiple relevant items."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5]
        ap = average_precision(recommended, relevant, k=5)
        # Hits at positions 1, 3, 5
        # Precisions: 1/1, 2/3, 3/5
        expected = (1.0 + 2/3 + 3/5) / 3
        assert ap == pytest.approx(expected)


class TestExpectedCalibrationError:
    """Test expected_calibration_error metric."""

    def test_perfectly_calibrated(self):
        """Test ECE with perfectly calibrated predictions."""
        # Predictions equal to empirical accuracy
        preds = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 1, 0, 0])
        ece = expected_calibration_error(preds, labels, bins=2)
        # Should be small or 0
        assert ece < 0.1

    def test_badly_calibrated(self):
        """Test ECE with badly calibrated predictions."""
        # Predictions are always high but accuracy is low
        preds = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ece = expected_calibration_error(preds, labels, bins=1)
        # Should be large (predicted 0.9, actual 0.1)
        assert ece > 0.5

    def test_empty_input_returns_zero(self):
        """Test ECE with empty input."""
        preds = np.array([])
        labels = np.array([])
        ece = expected_calibration_error(preds, labels)
        assert ece == 0.0

    def test_all_correct_predictions(self):
        """Test ECE when all predictions are correct."""
        preds = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        ece = expected_calibration_error(preds, labels, bins=2)
        # Perfect predictions should give 0 error
        assert ece < 1e-6


class TestEvaluateRecommendations:
    """Test evaluate_recommendations end-to-end."""

    def test_basic_evaluation(self):
        """Test basic evaluation with synthetic data."""
        recommendations = {
            0: [1, 2, 3, 4, 5],
            1: [5, 4, 3, 2, 1],
        }
        relevant = {
            0: [1, 2, 6, 7],
            1: [1, 3, 5],
        }

        report = evaluate_recommendations(recommendations, relevant)

        assert isinstance(report, RankingReport)
        assert 0.0 <= report.precision_at_5 <= 1.0
        assert 0.0 <= report.recall_at_5 <= 1.0
        assert 0.0 <= report.map_at_10 <= 1.0
        assert 0.0 <= report.ndcg_at_10 <= 1.0

    def test_perfect_recommendations(self):
        """Test evaluation with perfect recommendations."""
        recommendations = {
            0: [1, 2, 3, 4, 5],
            1: [1, 2, 3, 4, 5],
        }
        relevant = {
            0: [1, 2, 3],
            1: [1, 2, 3],
        }

        report = evaluate_recommendations(
            recommendations, relevant, k_prec=5, k_rec=5
        )

        # With 3 relevant items at top-5: precision = 3/5 = 0.6, recall = 3/3 = 1.0
        assert report.precision_at_5 == pytest.approx(0.6)
        assert report.recall_at_5 == pytest.approx(1.0)

    def test_custom_k_values(self):
        """Test evaluation with custom k values."""
        recommendations = {0: list(range(1, 21))}
        relevant = {0: [1, 2, 3, 4, 5]}

        report = evaluate_recommendations(
            recommendations,
            relevant,
            k_prec=3,
            k_rec=5,
            k_map=10,
            k_ndcg=10,
        )

        # Should use custom k values
        assert isinstance(report.precision_at_5, float)

    def test_empty_relevant(self):
        """Test evaluation with empty relevant items."""
        recommendations = {0: [1, 2, 3]}
        relevant = {0: []}

        report = evaluate_recommendations(recommendations, relevant)

        assert report.precision_at_5 == 0.0
        assert report.recall_at_5 == 0.0

    def test_missing_user(self):
        """Test evaluation when user is missing from relevant."""
        recommendations = {0: [1, 2, 3], 1: [4, 5, 6]}
        relevant = {0: [1, 2]}  # user 1 not in relevant

        # Should handle gracefully
        report = evaluate_recommendations(recommendations, relevant)
        assert isinstance(report, RankingReport)


class TestRankingReport:
    """Test RankingReport dataclass."""

    def test_construction(self):
        """Test RankingReport construction."""
        report = RankingReport(
            precision=0.8,
            recall=0.7,
            map=0.75,
            ndcg=0.85,
        )

        assert report.precision == 0.8
        assert report.recall == 0.7
        assert report.map == 0.75
        assert report.ndcg == 0.85
        # Backward-compatible aliases
        assert report.precision_at_5 == 0.8
        assert report.recall_at_5 == 0.7
        assert report.map_at_10 == 0.75
        assert report.ndcg_at_10 == 0.85
