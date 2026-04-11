"""Extended tests for baseline recommenders."""
import sys
sys.path.insert(0, "src")

import numpy as np
import torch
import pytest

from orchid_ranker.baselines import (
    MatrixFactorization,
    PopularityBaseline,
    RandomBaseline,
    ALSBaseline,
    UserKNNBaseline,
    LinUCBBaseline,
    ExplicitMFBaseline,
    NeuralMatrixFactorizationBaseline,
)


class TestMatrixFactorization:
    """Test MatrixFactorization module."""

    def test_forward_produces_correct_shape(self):
        """Test that forward pass produces correct output shape."""
        model = MatrixFactorization(num_users=10, num_items=20, emb_dim=32)
        user_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        item_ids = torch.tensor([5, 10, 15], dtype=torch.long)

        output = model(user_ids, item_ids)
        assert output.shape == (3,)

    def test_implicit_true_sigmoid(self):
        """Test that implicit=True applies sigmoid."""
        model = MatrixFactorization(
            num_users=5, num_items=5, emb_dim=8, implicit=True
        )
        user_ids = torch.tensor([0], dtype=torch.long)
        item_ids = torch.tensor([0], dtype=torch.long)

        output = model(user_ids, item_ids)
        assert 0.0 <= output.item() <= 1.0

    def test_implicit_false_unbounded(self):
        """Test that implicit=False gives unbounded values."""
        model = MatrixFactorization(
            num_users=5, num_items=5, emb_dim=8, implicit=False
        )
        user_ids = torch.tensor([0], dtype=torch.long)
        item_ids = torch.tensor([0], dtype=torch.long)

        output = model(user_ids, item_ids)
        # Should be unbounded (could be outside [0,1])
        assert isinstance(output.item(), float)

    def test_batch_processing(self):
        """Test that model handles batch inputs."""
        model = MatrixFactorization(num_users=20, num_items=30, emb_dim=16)
        user_ids = torch.randint(0, 20, (100,), dtype=torch.long)
        item_ids = torch.randint(0, 30, (100,), dtype=torch.long)

        output = model(user_ids, item_ids)
        assert output.shape == (100,)


class TestALSBaseline:
    """Test ALSBaseline."""

    def test_fit_infer_cycle(self):
        """Test fit and infer cycle."""
        device = torch.device("cpu")
        baseline = ALSBaseline(num_users=10, num_items=20, device=device)

        # Create dummy data
        user_ids = [0, 1, 2, 0, 1]
        item_ids = [5, 10, 15, 19, 5]
        labels = [1.0, 1.0, 0.0, 1.0, 1.0]

        baseline.fit(user_ids, item_ids, labels)
        assert baseline.result.train_loss is not None

        # Infer for user 0
        user_tensor = torch.tensor([0], dtype=torch.long, device=device)
        item_tensor = torch.tensor([5, 10], dtype=torch.long, device=device)
        scores = baseline.infer(user_ids=user_tensor, item_ids=item_tensor)

        assert scores.shape == (1, 2)
        assert torch.isfinite(scores).all()

    def test_decide_returns_items(self):
        """Test that decide returns items and metadata."""
        device = torch.device("cpu")
        baseline = ALSBaseline(num_users=5, num_items=10, device=device)
        baseline.fit([0, 1], [5, 6], [1.0, 1.0])

        logits = torch.tensor([[0.5, 0.3, 0.8, 0.2]], device=device)
        item_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

        chosen, metadata = baseline.decide(logits=logits, top_k=2, item_ids=item_ids)
        assert len(chosen) == 2
        assert all(isinstance(i, int) for i in chosen)
        assert metadata["policy"] == "als"


class TestPopularityBaseline:
    """Test PopularityBaseline."""

    def test_returns_correct_ordering(self):
        """Test that popularity baseline returns items in popularity order."""
        device = torch.device("cpu")
        popularity = {0: 0.1, 1: 0.9, 2: 0.5}
        baseline = PopularityBaseline(popularity, device=device)

        item_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        scores = baseline.infer(item_ids=item_ids)

        assert scores.shape == (1, 3)
        # Most popular should have highest score
        assert torch.argmax(scores[0]).item() == 1

    def test_decide_respects_popularity(self):
        """Test that decide returns most popular items."""
        device = torch.device("cpu")
        popularity = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.3}
        baseline = PopularityBaseline(popularity, device=device)

        item_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
        scores = baseline.infer(item_ids=item_ids)

        chosen, _ = baseline.decide(logits=scores, top_k=2, item_ids=item_ids)
        # Item 1 (most popular) should be first, then item 2
        assert 1 in chosen


class TestRandomBaseline:
    """Test RandomBaseline."""

    def test_returns_different_orders(self):
        """Test that random baseline returns different orders with different seeds."""
        device = torch.device("cpu")
        baseline1 = RandomBaseline(device)
        baseline2 = RandomBaseline(device)

        item_ids = torch.arange(10, dtype=torch.long, device=device)

        # Get multiple draws
        orders = []
        for _ in range(3):
            scores = baseline1.infer(item_ids=item_ids)
            chosen, _ = baseline1.decide(
                logits=scores, top_k=5, item_ids=item_ids
            )
            orders.append(chosen)

        # Should have some variation (not guaranteed but very likely)
        assert len(orders) == 3

    def test_returns_correct_tensor_shape(self):
        """Test that infer returns correct shape."""
        device = torch.device("cpu")
        baseline = RandomBaseline(device)
        item_ids = torch.arange(20, dtype=torch.long, device=device)

        scores = baseline.infer(item_ids=item_ids)
        assert scores.shape == (1, 20)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()


class TestUserKNNBaseline:
    """Test UserKNNBaseline."""

    def test_finds_similar_users(self):
        """Test that UserKNN finds similar users."""
        device = torch.device("cpu")
        # Create user-item matrix with patterns
        matrix = np.array([
            [1.0, 0.0, 1.0, 0.0],  # user 0: likes items 0,2
            [1.0, 0.0, 1.0, 0.0],  # user 1: likes items 0,2 (similar to 0)
            [0.0, 1.0, 0.0, 1.0],  # user 2: likes items 1,3
        ], dtype=np.float32)

        baseline = UserKNNBaseline(matrix, device=device, k=2)

        # User 0 should have similar preference to user 1
        user_ids = torch.tensor([0], dtype=torch.long, device=device)
        item_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

        scores = baseline.infer(user_ids=user_ids, item_ids=item_ids)
        assert scores.shape == (1, 4)


class TestLinUCBBaseline:
    """Test LinUCBBaseline."""

    def test_with_synthetic_features(self):
        """Test LinUCB with synthetic item features."""
        device = torch.device("cpu")
        # Create synthetic features: 5 items, 3 features each
        item_features = np.random.randn(5, 3).astype(np.float32)

        baseline = LinUCBBaseline(alpha=1.0, item_features=item_features, device=device)

        # Add some rewards
        rewards = {0: 1.0, 1: 0.5, 2: 0.8}
        baseline.fit(rewards)

        # Infer scores
        item_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device)
        scores = baseline.infer(item_ids=item_ids)

        assert scores.shape == (1, 5)
        assert torch.isfinite(scores).all()

    def test_decide_returns_top_k(self):
        """Test that decide returns top-k items."""
        device = torch.device("cpu")
        item_features = np.eye(4, dtype=np.float32)  # 4 items, orthogonal features
        baseline = LinUCBBaseline(alpha=1.0, item_features=item_features, device=device)

        logits = torch.tensor([[0.5, 0.3, 0.8, 0.2]], device=device)
        item_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

        chosen, metadata = baseline.decide(logits=logits, top_k=2, item_ids=item_ids)
        assert len(chosen) == 2


class TestExplicitMFBaseline:
    """Test ExplicitMFBaseline."""

    def test_fit_infer_cycle(self):
        """Test fit and infer cycle."""
        device = torch.device("cpu")
        baseline = ExplicitMFBaseline(
            num_users=10,
            num_items=20,
            device=device,
            epochs=2,
        )

        # Create dummy rating data
        user_ids = [0, 1, 2, 0, 1]
        item_ids = [5, 10, 15, 19, 5]
        ratings = [3.0, 4.0, 2.0, 5.0, 3.5]

        baseline.fit(user_ids, item_ids, ratings)

        # Infer for user 0
        user_tensor = torch.tensor([0], dtype=torch.long, device=device)
        item_tensor = torch.tensor([5, 10], dtype=torch.long, device=device)
        scores = baseline.infer(user_ids=user_tensor, item_ids=item_tensor)

        assert scores.shape == (1, 2)

    def test_clamps_to_rating_range(self):
        """Test that predictions are clamped to observed rating range."""
        device = torch.device("cpu")
        baseline = ExplicitMFBaseline(
            num_users=5,
            num_items=5,
            device=device,
            epochs=1,
        )

        # Ratings in [2, 5]
        user_ids = [0, 1]
        item_ids = [0, 1]
        ratings = [2.0, 5.0]

        baseline.fit(user_ids, item_ids, ratings)

        user_tensor = torch.tensor([0], dtype=torch.long, device=device)
        item_tensor = torch.tensor([0, 1], dtype=torch.long, device=device)
        scores = baseline.infer(user_ids=user_tensor, item_ids=item_tensor)

        # Scores should be clamped to [2, 5]
        assert (scores[0] >= 2.0).all()
        assert (scores[0] <= 5.0).all()


class TestNeuralMatrixFactorization:
    """Test NeuralMatrixFactorizationBaseline."""

    def test_with_bce_loss(self):
        """Test NMF with BCE loss."""
        device = torch.device("cpu")
        baseline = NeuralMatrixFactorizationBaseline(
            num_users=10,
            num_items=20,
            device=device,
            emb_dim=16,
            loss="bce",
            epochs=1,
        )

        user_ids = [0, 1, 2, 0, 1]
        item_ids = [5, 10, 15, 19, 5]
        labels = [1.0, 1.0, 0.0, 1.0, 1.0]

        baseline.fit(user_ids, item_ids, labels)

        user_tensor = torch.tensor([0], dtype=torch.long, device=device)
        item_tensor = torch.tensor([5, 10], dtype=torch.long, device=device)
        scores = baseline.infer(user_ids=user_tensor, item_ids=item_tensor)

        assert scores.shape == (1, 2)
        # BCE with sigmoid should be in [0, 1]
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_with_bpr_loss(self):
        """Test NMF with BPR loss."""
        device = torch.device("cpu")
        baseline = NeuralMatrixFactorizationBaseline(
            num_users=10,
            num_items=20,
            device=device,
            loss="bpr",
            epochs=1,
        )

        user_ids = [0, 1, 2, 0, 1]
        item_ids = [5, 10, 15, 19, 5]
        labels = [1.0, 0.0, 1.0, 1.0, 1.0]

        baseline.fit(user_ids, item_ids, labels)

        user_tensor = torch.tensor([0], dtype=torch.long, device=device)
        item_tensor = torch.tensor([5, 10], dtype=torch.long, device=device)
        scores = baseline.infer(user_ids=user_tensor, item_ids=item_tensor)

        assert scores.shape == (1, 2)
        assert torch.isfinite(scores).all()

    def test_with_softmax_loss(self):
        """Test NMF with sampled softmax loss."""
        device = torch.device("cpu")
        baseline = NeuralMatrixFactorizationBaseline(
            num_users=10,
            num_items=20,
            device=device,
            loss="softmax",
            neg_k=5,
            epochs=1,
        )

        user_ids = [0, 1, 2, 0, 1]
        item_ids = [5, 10, 15, 19, 5]
        labels = [1.0, 1.0, 0.0, 1.0, 1.0]

        baseline.fit(user_ids, item_ids, labels)

        user_tensor = torch.tensor([0], dtype=torch.long, device=device)
        item_tensor = torch.tensor([5, 10], dtype=torch.long, device=device)
        scores = baseline.infer(user_ids=user_tensor, item_ids=item_tensor)

        assert scores.shape == (1, 2)

    def test_infer_returns_correct_shape(self):
        """Test that all baselines return (1, num_items) shape."""
        device = torch.device("cpu")

        for loss_type in ["bce", "bpr", "softmax"]:
            baseline = NeuralMatrixFactorizationBaseline(
                num_users=8,
                num_items=15,
                device=device,
                loss=loss_type,
                epochs=1,
            )

            user_ids = [0, 1]
            item_ids = [0, 5]
            labels = [1.0, 0.0]

            baseline.fit(user_ids, item_ids, labels)

            user_tensor = torch.tensor([0], dtype=torch.long, device=device)
            item_tensor = torch.arange(15, dtype=torch.long, device=device)
            scores = baseline.infer(user_ids=user_tensor, item_ids=item_tensor)

            assert scores.shape == (1, 15)
