"""Tests for model selection utilities: train_test_split and evaluate_on_holdout."""

import numpy as np
import pandas as pd

from orchid_ranker.model_selection import train_test_split, evaluate_on_holdout
from orchid_ranker.recommender import OrchidRecommender


class TestTrainTestSplit:
    """Test train_test_split for data partitioning."""

    def test_split_basic(self):
        """Test basic train-test split."""
        df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'item_id': [10, 20, 30, 10, 20, 40, 50, 60, 70],
        })
        train, test = train_test_split(df, test_size=0.2, by_user=True)

        assert len(train) + len(test) == len(df)
        assert len(train) > 0
        assert len(test) > 0

    def test_split_by_user(self):
        """Test per-user stratified split."""
        df = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 2, 2, 2, 2],
            'item_id': [10, 20, 30, 40, 50, 60, 70, 80],
        })
        train, test = train_test_split(df, test_size=0.25, by_user=True, random_state=42)

        # Each user should have ~1 test item (25% of 4)
        user_1_train = len(train[train['user_id'] == 1])
        user_1_test = len(test[test['user_id'] == 1])
        assert user_1_train >= 3
        assert user_1_test >= 1

    def test_split_global(self):
        """Test global random split."""
        df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [10, 20, 30, 40, 50, 60],
        })
        train, test = train_test_split(df, test_size=0.33, by_user=False, random_state=42)

        assert len(train) + len(test) == len(df)
        # Approximately 33% test (with tolerance for small dataset rounding)
        assert abs(len(test) / len(df) - 0.33) < 0.2

    def test_split_invalid_test_size_zero(self):
        """Test that test_size=0 raises ValueError."""
        df = pd.DataFrame({'user_id': [1, 2], 'item_id': [10, 20]})
        try:
            train_test_split(df, test_size=0.0)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "test_size" in str(e).lower()

    def test_split_invalid_test_size_one(self):
        """Test that test_size=1 raises ValueError."""
        df = pd.DataFrame({'user_id': [1, 2], 'item_id': [10, 20]})
        try:
            train_test_split(df, test_size=1.0)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "test_size" in str(e).lower()

    def test_split_no_overlap(self):
        """Test that train and test have no overlap in rows."""
        df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2],
            'item_id': [10, 20, 30, 40, 50, 60],
        })
        train, test = train_test_split(df, test_size=0.33, by_user=True, random_state=42)

        # Verify that no row appears in both splits by checking data values
        # Convert rows to tuples for comparison
        train_rows = set(map(tuple, train.values))
        test_rows = set(map(tuple, test.values))
        assert len(train_rows & test_rows) == 0

    def test_split_reproducible_with_seed(self):
        """Test that split is reproducible with same random_state."""
        df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'item_id': [10, 20, 30, 40, 50, 60, 70, 80, 90],
        })
        train1, test1 = train_test_split(df, test_size=0.3, random_state=42)
        train2, test2 = train_test_split(df, test_size=0.3, random_state=42)

        assert train1.equals(train2)
        assert test1.equals(test2)

    def test_split_respects_columns(self):
        """Test that split respects column names."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 2, 2],
            'product_id': [10, 20, 30, 40],
            'rating': [5, 4, 3, 5],
        })
        train, test = train_test_split(
            df, test_size=0.25, by_user=True,
            user_col='customer_id', item_col='product_id'
        )

        assert 'rating' in train.columns
        assert 'rating' in test.columns
        assert all(train['customer_id'].isin([1, 2]))

    def test_split_single_user_per_group(self):
        """Test split when each user has minimal interactions."""
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [10, 20, 30],
        })
        train, test = train_test_split(df, test_size=0.33, by_user=True, random_state=42)

        # Each user has 1 item, so test should get at least 1
        assert len(test) >= 1


class TestEvaluateOnHoldout:
    """Test evaluate_on_holdout for model evaluation on test data."""

    def test_evaluate_basic(self):
        """Test basic evaluation on holdout data."""
        # Create simple train and test data
        train_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 1, 3, 2, 3],
            'rating': [5, 4, 4, 5, 3, 5],
        })
        test_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [3, 2, 1],
            'rating': [4, 3, 4],
        })

        # Train a simple model
        model = OrchidRecommender(strategy='random')
        model.fit(train_df, user_col='user_id', item_col='item_id')

        # Evaluate
        scores = evaluate_on_holdout(
            model, test_df,
            metrics=['precision@5', 'recall@5'],
            k=5,
            user_col='user_id',
            item_col='item_id'
        )

        assert 'precision@5' in scores
        assert 'recall@5' in scores
        assert 0 <= scores['precision@5'] <= 1
        assert 0 <= scores['recall@5'] <= 1

    def test_evaluate_empty_test_set(self):
        """Test evaluation with empty test set."""
        train_df = pd.DataFrame({
            'user_id': [1, 1],
            'item_id': [1, 2],
        })
        test_df = pd.DataFrame({
            'user_id': pd.Series([], dtype=int),
            'item_id': pd.Series([], dtype=int),
        })

        model = OrchidRecommender(strategy='random')
        model.fit(train_df)

        scores = evaluate_on_holdout(model, test_df, metrics=['precision@5'])
        assert scores['precision@5'] == 0.0

    def test_evaluate_default_metrics(self):
        """Test that default metrics are used when none specified."""
        train_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'item_id': [1, 2, 2, 3],
        })
        test_df = pd.DataFrame({
            'user_id': [1, 2],
            'item_id': [3, 1],
        })

        model = OrchidRecommender(strategy='random')
        model.fit(train_df)

        scores = evaluate_on_holdout(model, test_df)

        # Should have default metrics
        assert 'precision@5' in scores
        assert 'recall@5' in scores
        assert 'ndcg@10' in scores
        assert 'map@10' in scores

    def test_evaluate_invalid_metric_raises(self):
        """Test that invalid metric name raises ValueError."""
        train_df = pd.DataFrame({
            'user_id': [1, 1],
            'item_id': [1, 2],
        })
        test_df = pd.DataFrame({
            'user_id': [1],
            'item_id': [1],
        })

        model = OrchidRecommender(strategy='random')
        model.fit(train_df)

        try:
            evaluate_on_holdout(model, test_df, metrics=['invalid_metric'])
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "metric" in str(e).lower()

    def test_evaluate_custom_k(self):
        """Test evaluation with custom k value."""
        train_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 2, 3, 3, 4],
        })
        test_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [3, 1, 2],
        })

        model = OrchidRecommender(strategy='random')
        model.fit(train_df)

        scores = evaluate_on_holdout(
            model, test_df,
            metrics=['precision@5'],
            k=20  # Request 20 recommendations
        )

        assert 'precision@5' in scores

    def test_evaluate_respects_column_names(self):
        """Test evaluation with custom column names."""
        train_df = pd.DataFrame({
            'customer': [1, 1, 2, 2],
            'product': [1, 2, 2, 3],
        })
        test_df = pd.DataFrame({
            'customer': [1, 2],
            'product': [3, 1],
        })

        model = OrchidRecommender(strategy='random')
        model.fit(train_df, user_col='customer', item_col='product')

        scores = evaluate_on_holdout(
            model, test_df,
            user_col='customer',
            item_col='product',
            metrics=['precision@5']
        )

        assert 'precision@5' in scores

    def test_evaluate_all_metrics(self):
        """Test evaluation with all available metrics."""
        train_df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'item_id': [1, 2, 3, 2, 3, 4, 3, 4, 5],
        })
        test_df = pd.DataFrame({
            'user_id': [1, 2, 3, 1],
            'item_id': [4, 1, 2, 5],
        })

        model = OrchidRecommender(strategy='random')
        model.fit(train_df)

        scores = evaluate_on_holdout(
            model, test_df,
            metrics=['precision@5', 'recall@5', 'ndcg@10', 'map@10'],
            k=10
        )

        assert len(scores) == 4
        for metric in ['precision@5', 'recall@5', 'ndcg@10', 'map@10']:
            assert metric in scores
            assert 0 <= scores[metric] <= 1

    def test_evaluate_handles_missing_users(self):
        """Test evaluation when model can't generate recs for some users."""
        train_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'item_id': [1, 2, 2, 3],
        })
        # Test user doesn't exist in training
        test_df = pd.DataFrame({
            'user_id': [1, 99],
            'item_id': [3, 1],
        })

        model = OrchidRecommender(strategy='random')
        model.fit(train_df)

        # Should not crash
        scores = evaluate_on_holdout(model, test_df, metrics=['precision@5'])
        assert 'precision@5' in scores

    def test_evaluate_multiple_items_per_user(self):
        """Test evaluation with multiple test items per user."""
        train_df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2],
            'item_id': [1, 2, 3, 2, 3, 4],
        })
        test_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'item_id': [4, 5, 1, 5],
        })

        model = OrchidRecommender(strategy='random')
        model.fit(train_df)

        scores = evaluate_on_holdout(
            model, test_df,
            metrics=['recall@5', 'precision@5'],
            k=10
        )

        # Recall should be computable even with multiple items per user
        assert 0 <= scores['recall@5'] <= 1

    def test_evaluate_score_bounds(self):
        """Test that all scores are in valid ranges."""
        train_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 2, 3, 3, 4],
        })
        test_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [3, 1, 2],
        })

        model = OrchidRecommender(strategy='random')
        model.fit(train_df)

        scores = evaluate_on_holdout(model, test_df)

        for metric, score in scores.items():
            assert 0 <= score <= 1, f"{metric} = {score} is out of bounds"
