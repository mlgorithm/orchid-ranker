"""Baseline recommenders (non-adaptive) used in experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:  # optional dependency for implicit baselines
    import implicit
except ImportError:  # pragma: no cover - optional path
    implicit = None


class MatrixFactorization(nn.Module):
    """Neural matrix factorization model with embeddings and biases.

    Implements a basic matrix factorization recommender using embeddings for users
    and items, plus learned bias terms. Supports both implicit (sigmoid-activated)
    and explicit (raw score) output modes.

    Parameters
    ----------
    num_users : int
        Number of unique users.
    num_items : int
        Number of unique items.
    emb_dim : int, optional
        Embedding dimension (default: 32).
    implicit : bool, optional
        If True, apply sigmoid activation to outputs. If False, return raw scores
        (default: True).
    """

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 32, implicit: bool = True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.implicit = bool(implicit)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding weights with scaled normal, biases to zero."""
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute preference scores for user-item pairs.

        Parameters
        ----------
        user_ids : torch.Tensor
            User indices of shape (batch_size,).
        item_ids : torch.Tensor
            Item indices of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Predicted scores of shape (batch_size,). If implicit=True, values are
            in [0, 1] (sigmoid-activated). If implicit=False, raw unbounded scores.
        """
        u_emb = self.user_emb(user_ids)
        i_emb = self.item_emb(item_ids)
        dot = (u_emb * i_emb).sum(dim=1, keepdim=True)
        output = dot + self.user_bias(user_ids) + self.item_bias(item_ids)
        # NOTE: Using squeeze(-1) instead of squeeze() preserves the batch dimension
        # even when the batch size is 1. This avoids returning a scalar tensor,
        # which was causing downstream loss computations to receive mismatched
        # shapes (scalar vs. 1-element tensor) and throw an error.
        if self.implicit:
            return torch.sigmoid(output).squeeze(-1)
        return output.squeeze(-1)


@dataclass
class BaselineResult:
    """Result container for baseline training.

    Attributes
    ----------
    train_loss : float, optional
        Final training loss after fitting, if applicable.
    """

    train_loss: Optional[float] = None


class BaseBaseline:
    """Abstract base class for all baseline recommenders.

    Provides a common interface for fitting, inference, and decision-making
    across different baseline strategies.

    Parameters
    ----------
    device : torch.device
        Torch device (CPU or CUDA) for computation.

    Attributes
    ----------
    device : torch.device
        The compute device.
    user_matrix : array-like, optional
        Optional user interaction matrix for some baselines.
    result : BaselineResult
        Container for training results.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.user_matrix = None
        self.result = BaselineResult()

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - default no-op
        """Train the baseline on interaction data.

        This is a no-op in the base class; subclasses should override.

        Parameters
        ----------
        *args
            Positional arguments (e.g., user_ids, item_ids, labels).
        **kwargs
            Keyword arguments (e.g., num_users, num_items).
        """
        return None

    def infer(self, **kwargs):  # pragma: no cover - override in subclasses
        """Compute item scores for a given context.

        Must be overridden by subclasses.

        Returns
        -------
        torch.Tensor
            Score tensor of shape (1, num_items).
        """
        raise NotImplementedError

    def decide(self, **kwargs):  # pragma: no cover
        """Select top-k items from scored candidates.

        Must be overridden by subclasses.

        Returns
        -------
        tuple
            (selected_item_ids, metadata_dict)
        """
        raise NotImplementedError


class PopularityBaseline(BaseBaseline):
    """Recommends items based on historical popularity (item frequency/rating average).

    Parameters
    ----------
    popularity : dict
        Mapping from item ID to popularity score.
    device : torch.device
        Torch device for computation.
    """

    def __init__(self, popularity: Dict[int, float], device: torch.device):
        super().__init__(device)
        self.popularity = popularity
        # Pre-build lookup tensor for vectorized scoring
        if popularity:
            max_id = max(popularity.keys()) + 1
        else:
            max_id = 1
        # Pre-allocate with buffer to avoid runtime expansion on OOV items
        buffer_size = max(max_id, 1) + max(int(max_id * 0.2), 100)
        self._pop_tensor = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        for item_id, score in popularity.items():
            self._pop_tensor[item_id] = score

    def infer(self, *, item_ids: torch.Tensor, **_):
        """Score items by their popularity.

        Parameters
        ----------
        item_ids : torch.Tensor
            Item indices to score.

        Returns
        -------
        torch.Tensor
            Popularity scores of shape (1, num_items).
        """
        ids = item_ids.long()
        # Expand lookup tensor if needed for out-of-range item IDs
        max_id = int(ids.max().item()) + 1 if ids.numel() > 0 else 0
        if max_id > self._pop_tensor.size(0):
            expanded = torch.zeros(max_id, dtype=torch.float32, device=self.device)
            expanded[:self._pop_tensor.size(0)] = self._pop_tensor
            self._pop_tensor = expanded
        scores = self._pop_tensor[ids]
        return scores.unsqueeze(0)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        """Select top-k most popular items.

        Parameters
        ----------
        logits : torch.Tensor
            Score tensor from infer().
        top_k : int
            Number of items to select.
        item_ids : torch.Tensor
            Candidate item IDs.

        Returns
        -------
        tuple
            (list of selected item IDs, metadata dict)
        """
        order = torch.argsort(logits[0], descending=True)
        chosen = order[:top_k]
        return [int(item_ids[i].item()) for i in chosen], {"policy": "popularity"}


class RandomBaseline(BaseBaseline):
    """Recommends items uniformly at random.

    Useful as a baseline for comparison and hypothesis testing.
    """

    def infer(self, *, item_ids: torch.Tensor, **_):
        """Generate random scores for all items.

        Parameters
        ----------
        item_ids : torch.Tensor
            Item indices to score.

        Returns
        -------
        torch.Tensor
            Random scores of shape (1, num_items), uniform in [0, 1).
        """
        return torch.rand((1, item_ids.numel()), device=self.device)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        """Select top-k items uniformly at random.

        Parameters
        ----------
        logits : torch.Tensor
            Score tensor (ignored, used only for interface compatibility).
        top_k : int
            Number of items to select.
        item_ids : torch.Tensor
            Candidate item IDs.

        Returns
        -------
        tuple
            (list of randomly selected item IDs, metadata dict)
        """
        chosen = torch.randperm(item_ids.numel(), device=self.device)[:top_k]
        return [int(item_ids[i].item()) for i in chosen], {"policy": "random"}


class ALSBaseline(BaseBaseline):
    """Alternating Least Squares matrix factorization for binary feedback.

    Trains embeddings to predict user-item interaction probabilities using
    binary cross-entropy loss.

    Parameters
    ----------
    num_users : int
        Number of unique users.
    num_items : int
        Number of unique items.
    device : torch.device
        Torch device for computation.
    emb_dim : int, optional
        Embedding dimension (default: 32).
    lr : float, optional
        Adam learning rate (default: 0.01).
    epochs : int, optional
        Number of training epochs (default: 5).
    """

    def __init__(self, num_users: int, num_items: int, device: torch.device,
                 embedding_dim: int = 32, learning_rate: float = 1e-2, epochs: int = 5,
                 *, emb_dim: Optional[int] = None, lr: Optional[float] = None):
        super().__init__(device)
        # Support abbreviated aliases for backward compatibility
        embedding_dim = emb_dim if emb_dim is not None else embedding_dim
        learning_rate = lr if lr is not None else learning_rate
        self.model = MatrixFactorization(num_users, num_items, emb_dim=embedding_dim, implicit=True).to(device)
        self.learning_rate = learning_rate
        self.lr = learning_rate  # alias
        self.epochs = epochs

    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], labels: Iterable[int]) -> None:
        """Train the matrix factorization model.

        Parameters
        ----------
        user_ids : Iterable[int]
            User indices.
        item_ids : Iterable[int]
            Item indices.
        labels : Iterable[int]
            Binary labels (0 or 1) indicating interaction.
        """
        user_tensor = torch.tensor(list(user_ids), dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(list(item_ids), dtype=torch.long, device=self.device)
        label_tensor = torch.tensor(list(labels), dtype=torch.float32, device=self.device)
        # ALS uses BCE loss which requires labels in [0, 1]; clamp to handle explicit ratings
        label_tensor = torch.clamp(label_tensor, 0.0, 1.0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(user_tensor, item_tensor)
            loss = loss_fn(preds, label_tensor)
            loss.backward()
            optimizer.step()
        self.result.train_loss = float(loss.detach().cpu().item())

    def infer(self, *, user_ids: torch.Tensor, item_ids: torch.Tensor, **_):
        """Score all items for a given user.

        Parameters
        ----------
        user_ids : torch.Tensor
            User index (will be broadcast to all items).
        item_ids : torch.Tensor
            Candidate item indices.

        Returns
        -------
        torch.Tensor
            Predicted scores of shape (1, num_items), in [0, 1].
        """
        users = user_ids.expand(item_ids.numel())
        items = item_ids
        with torch.no_grad():
            scores = self.model(users, items)
        return scores.view(1, -1)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        """Select top-k items by predicted score.

        Parameters
        ----------
        logits : torch.Tensor
            Score tensor from infer().
        top_k : int
            Number of items to select.
        item_ids : torch.Tensor
            Candidate item IDs.

        Returns
        -------
        tuple
            (list of selected item IDs, metadata dict)
        """
        top = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in top], {"policy": "als"}


class UserKNNBaseline(BaseBaseline):
    """User-based collaborative filtering using k-nearest neighbors.

    Recommends items liked by similar users. Similarity is computed as
    cosine distance in user-item interaction space.

    Parameters
    ----------
    user_item_matrix : np.ndarray
        User-item interaction matrix of shape (num_users, num_items).
    device : torch.device
        Torch device for computation.
    k : int, optional
        Number of nearest neighbors to consider (default: 20).
    """

    def __init__(self, user_item_matrix: np.ndarray, device: torch.device, k: int = 20):
        super().__init__(device)
        self.matrix = user_item_matrix.astype(np.float32)
        num_users = self.matrix.shape[0]
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = min(int(k), max(1, num_users - 1))
        self._matrix_t = torch.from_numpy(self.matrix).to(device)
        self._norms = torch.norm(self._matrix_t, dim=1, keepdim=True).clamp(min=1e-8)
        self._normed = self._matrix_t / self._norms

    def infer(self, *, user_ids: torch.Tensor, item_ids: torch.Tensor, **_):
        """Score items based on similar users' preferences.

        Parameters
        ----------
        user_ids : torch.Tensor
            User index.
        item_ids : torch.Tensor
            Candidate item indices.

        Returns
        -------
        torch.Tensor
            Item scores of shape (1, num_items).
        """
        uid = int(user_ids.item())
        if not (0 <= uid < self._normed.shape[0]):
            raise ValueError(
                f"User ID {uid} out of bounds [0, {self._normed.shape[0]}). "
                "Pass a valid user index from the interaction matrix."
            )
        user_vec = self._normed[uid]  # (num_items,)
        # Cosine similarity via dot product of normalized vectors
        sims = self._normed @ user_vec  # (num_users,)
        # Get top-k neighbors (exclude self)
        sims[uid] = -1.0
        _, top_idx = torch.topk(sims, self.k)
        # Average neighbor preferences
        neighbor_pref = self._matrix_t[top_idx].mean(dim=0)  # (num_items,)
        scores = neighbor_pref[item_ids.long()]
        return scores.unsqueeze(0)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        """Select top-k items by neighbor preference score.

        Parameters
        ----------
        logits : torch.Tensor
            Score tensor from infer().
        top_k : int
            Number of items to select.
        item_ids : torch.Tensor
            Candidate item IDs.

        Returns
        -------
        tuple
            (list of selected item IDs, metadata dict)
        """
        order = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in order], {"policy": "user_knn"}


class LinUCBBaseline(BaseBaseline):
    """Linear Upper Confidence Bound contextual bandit algorithm.

    Uses item features to maintain a linear model with confidence bounds for
    exploration-exploitation tradeoff.

    Parameters
    ----------
    alpha : float
        Exploration bonus multiplier for the UCB term.
    item_features : np.ndarray
        Item feature matrix of shape (num_items, feature_dim).
    device : torch.device
        Torch device for computation.
    """

    def __init__(self, alpha: float, item_features: np.ndarray, device: torch.device):
        super().__init__(device)
        self.alpha = alpha
        self.item_features = torch.tensor(item_features, dtype=torch.float32, device=device)
        d = self.item_features.shape[1] if self.item_features.shape[1] > 0 else 1
        self.A = torch.eye(d, device=device)
        self.b = torch.zeros(d, device=device)
        self._A_inv = None  # cached inverse, invalidated on fit()

    def fit(self, rewards: Dict[int, float]) -> None:
        """Update linear model parameters with observed rewards.

        Parameters
        ----------
        rewards : dict
            Mapping from item ID to reward value.
        """
        if self.item_features.shape[1] == 0:
            return
        n_items = self.item_features.shape[0]
        for iid, r in rewards.items():
            if not (0 <= int(iid) < n_items):
                raise ValueError(
                    f"Item ID {iid} out of bounds [0, {n_items}). "
                    "Pass valid item indices matching the feature matrix."
                )
            feature = self.item_features[iid]
            self.A += feature.unsqueeze(1) @ feature.unsqueeze(0)
            self.b += float(r) * feature
        self._A_inv = None  # invalidate cached inverse

    def infer(self, *, item_ids: torch.Tensor, **_):
        """Compute UCB scores for candidate items.

        Parameters
        ----------
        item_ids : torch.Tensor
            Candidate item indices.

        Returns
        -------
        torch.Tensor
            UCB scores of shape (1, num_items).
        """
        if self.item_features.shape[1] == 0:
            return torch.zeros((1, item_ids.numel()), device=self.device)
        if self._A_inv is None:
            self._A_inv = torch.inverse(self.A + 1e-6 * torch.eye(self.A.shape[0], device=self.device))
        A_inv = self._A_inv
        theta = A_inv @ self.b
        feats = self.item_features[item_ids]
        means = feats @ theta
        # Quadratic form x'A^{-1}x; clamp to avoid negative values from numerical noise
        quadratic = ((feats @ A_inv) * feats).sum(dim=1).clamp(min=0.0)
        ucb = self.alpha * torch.sqrt(quadratic)
        scores = means + ucb
        return scores.unsqueeze(0)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **kwargs):
        """Select top-k items by UCB score.

        Parameters
        ----------
        logits : torch.Tensor
            Score tensor from infer().
        top_k : int
            Number of items to select.
        item_ids : torch.Tensor
            Candidate item IDs.

        Returns
        -------
        tuple
            (list of selected item IDs, metadata dict)
        """
        order = torch.argsort(logits[0], descending=True)[:top_k]
        chosen = [int(item_ids[i].item()) for i in order]
        return chosen, {"policy": "linucb"}


class _ImplicitBase(BaseBaseline):
    """Base class for implicit feedback baselines using the `implicit` library.

    Provides common infrastructure for training with implicit library models.

    Parameters
    ----------
    factors : int, optional
        Embedding dimension (default: 64).
    iterations : int, optional
        Number of training iterations (default: 20).
    regularization : float, optional
        L2 regularization strength (default: 0.01).
    **kwargs
        Additional arguments passed to implicit model constructors.
    """

    def __init__(self, *, factors: int = 64, iterations: int = 20, regularization: float = 0.01, **kwargs):
        if implicit is None:
            raise ImportError("The 'implicit' package is required for this strategy. Install via `pip install implicit`.")
        super().__init__(torch.device("cpu"))
        self.factors = int(factors)
        self.iterations = int(iterations)
        self.regularization = float(regularization)
        self.kwargs = kwargs
        self.model = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    @staticmethod
    def _coo_matrix(user_ids: Iterable[int], item_ids: Iterable[int], labels: Iterable[float], num_users: int, num_items: int):
        """Convert interaction lists to COO sparse matrix format.

        Parameters
        ----------
        user_ids : Iterable[int]
            User indices.
        item_ids : Iterable[int]
            Item indices.
        labels : Iterable[float]
            Interaction weights.
        num_users : int
            Total number of users.
        num_items : int
            Total number of items.

        Returns
        -------
        scipy.sparse.coo_matrix
            Sparse COO matrix of shape (num_users, num_items).
        """
        import scipy.sparse

        rows = np.asarray(user_ids, dtype=np.int32) if not isinstance(user_ids, np.ndarray) else user_ids.astype(np.int32, copy=False)
        cols = np.asarray(item_ids, dtype=np.int32) if not isinstance(item_ids, np.ndarray) else item_ids.astype(np.int32, copy=False)
        data = np.asarray(labels, dtype=np.float32) if not isinstance(labels, np.ndarray) else labels.astype(np.float32, copy=False)
        return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_users, num_items))

    def infer(self, *, user_ids: torch.Tensor, item_ids: torch.Tensor, **_) -> torch.Tensor:
        """Score items using learned factor embeddings.

        Parameters
        ----------
        user_ids : torch.Tensor
            User index.
        item_ids : torch.Tensor
            Candidate item indices.

        Returns
        -------
        torch.Tensor
            Predicted scores of shape (1, num_items).

        Raises
        ------
        RuntimeError
            If model has not been trained yet.
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Implicit model has not been trained")
        uid_np = user_ids.numpy() if user_ids.device.type == 'cpu' else user_ids.cpu().numpy()
        iid_np = item_ids.numpy() if item_ids.device.type == 'cpu' else item_ids.cpu().numpy()
        u = self.user_factors[uid_np]
        i = self.item_factors[iid_np]
        scores = np.dot(u, i.T)
        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        """Select top-k items by score.

        Parameters
        ----------
        logits : torch.Tensor
            Score tensor from infer().
        top_k : int
            Number of items to select.
        item_ids : torch.Tensor
            Candidate item IDs.

        Returns
        -------
        tuple
            (list of selected item IDs, metadata dict)
        """
        top = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in top], {"policy": type(self).__name__.lower()}


class ImplicitALSBaseline(_ImplicitBase):
    """Implicit ALS (Alternating Least Squares) baseline using the `implicit` library.

    Optimized for binary or weighted implicit feedback (e.g., play counts).
    """

    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], labels: Iterable[float], *, num_users: int, num_items: int) -> None:
        """Train the implicit ALS model.

        Parameters
        ----------
        user_ids : Iterable[int]
            User indices.
        item_ids : Iterable[int]
            Item indices.
        labels : Iterable[float]
            Interaction weights (typically counts or 0/1).
        num_users : int
            Total number of users.
        num_items : int
            Total number of items.
        """
        if implicit is None:
            raise ImportError("implicit is required for ImplicitALSBaseline")
        coo = self._coo_matrix(user_ids, item_ids, labels, num_users, num_items)
        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            **self.kwargs,
        )
        model.fit(coo.tocsr())
        self.model = model
        self.user_factors = np.asarray(model.user_factors)
        self.item_factors = np.asarray(model.item_factors)


class ImplicitBPRBaseline(_ImplicitBase):
    """Implicit BPR (Bayesian Personalized Ranking) baseline using the `implicit` library.

    Optimized for ranking with pairwise loss, good for implicit feedback scenarios.

    Parameters
    ----------
    factors : int, optional
        Embedding dimension (default: 64).
    iterations : int, optional
        Number of training iterations (default: 50).
    learning_rate : float, optional
        Learning rate (default: 0.01).
    regularization : float, optional
        L2 regularization strength (default: 0.01).
    **kwargs
        Additional arguments passed to implicit BPR model.
    """

    def __init__(self, *, factors: int = 64, iterations: int = 50, learning_rate: float = 0.01, regularization: float = 0.01, **kwargs):
        super().__init__(factors=factors, iterations=iterations, regularization=regularization, **kwargs)
        self.learning_rate = float(learning_rate)

    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], labels: Iterable[float], *, num_users: int, num_items: int) -> None:
        """Train the implicit BPR model.

        Parameters
        ----------
        user_ids : Iterable[int]
            User indices.
        item_ids : Iterable[int]
            Item indices.
        labels : Iterable[float]
            Interaction weights (thresholded to binary for BPR).
        num_users : int
            Total number of users.
        num_items : int
            Total number of items.
        """
        if implicit is None:
            raise ImportError("implicit is required for ImplicitBPRBaseline")
        user_arr = np.asarray(list(user_ids), dtype=np.int32)
        item_arr = np.asarray(list(item_ids), dtype=np.int32)
        label_arr = np.asarray(list(labels), dtype=np.float32)
        mask = label_arr > 0
        if not np.any(mask):
            mask = np.ones_like(label_arr, dtype=bool)
        coo = self._coo_matrix(user_arr[mask], item_arr[mask], np.ones(mask.sum(), dtype=np.float32), num_users, num_items)
        counts = coo.tocsr()
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            **self.kwargs,
        )
        model.fit(counts)
        self.model = model
        self.user_factors = np.asarray(model.user_factors)
        self.item_factors = np.asarray(model.item_factors)


class NeuralMatrixFactorizationBaseline(BaseBaseline):
    """Neural matrix factorization with configurable loss functions (BCE, BPR, or Softmax).

    Combines embeddings with an MLP to learn user-item preferences. Supports
    multiple training objectives: binary cross-entropy (implicit feedback),
    Bayesian Personalized Ranking (ranking), or sampled softmax (next-item prediction).

    Parameters
    ----------
    num_users : int
        Number of unique users.
    num_items : int
        Number of unique items.
    device : torch.device
        Torch device for computation.
    emb_dim : int, optional
        Embedding dimension (default: 32).
    hidden : tuple of int, optional
        Hidden layer sizes for MLP (default: (64, 32)).
    epochs : int, optional
        Number of training epochs (default: 5).
    lr : float, optional
        Adam learning rate (default: 0.001).
    loss : str, optional
        Loss type: "bce" (binary cross-entropy), "bpr" (Bayesian Personalized Ranking),
        or "softmax" (sampled softmax). Default: "bce".
    neg_k : int, optional
        Number of negative samples for BPR and softmax losses (default: 10).
    batch_size : int, optional
        Training batch size (default: 256).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        device: torch.device,
        embedding_dim: int = 32,
        hidden: Tuple[int, ...] = (64, 32),
        epochs: int = 5,
        learning_rate: float = 1e-3,
        loss: str = "bce",
        num_negative_samples: int = 10,
        batch_size: int = 256,
        *,
        # Backward-compatible abbreviated aliases
        emb_dim: Optional[int] = None,
        lr: Optional[float] = None,
        neg_k: Optional[int] = None,
    ) -> None:
        super().__init__(device)
        # Support abbreviated aliases
        embedding_dim = emb_dim if emb_dim is not None else embedding_dim
        learning_rate = lr if lr is not None else learning_rate
        num_negative_samples = neg_k if neg_k is not None else num_negative_samples

        valid_losses = {"bce", "bpr", "softmax"}
        if str(loss).lower() not in valid_losses:
            raise ValueError(f"Unknown loss='{loss}'. Must be one of: {sorted(valid_losses)}")

        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = embedding_dim
        self.hidden = tuple(hidden)
        self.epochs = int(epochs)
        self.lr = float(learning_rate)
        self.loss_type = str(loss).lower()
        self.neg_k = int(num_negative_samples)
        self.batch_size = int(batch_size)

        layers: list[nn.Module] = []
        in_dim = self.emb_dim * 2
        for h in self.hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers).to(self.device)

        self.user_emb = nn.Embedding(num_users, self.emb_dim).to(self.device)
        self.item_emb = nn.Embedding(num_items, self.emb_dim).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()

    def parameters(self):
        """Return trainable model parameters.

        Returns
        -------
        list
            List of all model parameters (MLP + user embeddings + item embeddings).
        """
        return list(self.mlp.parameters()) + list(self.user_emb.parameters()) + list(self.item_emb.parameters())

    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], labels: Iterable[float]) -> None:
        """Train the neural matrix factorization model.

        Parameters
        ----------
        user_ids : Iterable[int]
            User indices.
        item_ids : Iterable[int]
            Item indices.
        labels : Iterable[float]
            Target labels (0/1 for binary feedback, or real-valued ratings).
        """
        users = torch.tensor(list(user_ids), dtype=torch.long, device=self.device)
        items = torch.tensor(list(item_ids), dtype=torch.long, device=self.device)
        y = torch.tensor(list(labels), dtype=torch.float32, device=self.device)
        if self.loss_type == "bpr":
            # BPR: Bayesian Personalized Ranking
            bsz = max(256, self.batch_size)
            for _ in range(self.epochs):
                perm = torch.randperm(users.size(0), device=self.device)
                for start in range(0, users.size(0), bsz):
                    idx = perm[start:start+bsz]
                    u_pos = users[idx]
                    i_pos = items[idx]
                    # Vectorized negative sampling — sample batch at once on device.
                    # For large item counts, collision probability with positives is negligible,
                    # so uniform sampling is a good approximation and avoids rejection sampling overhead.
                    j_neg = torch.randint(0, self.num_items, (u_pos.size(0),), device=self.device)
                    self.optimizer.zero_grad()
                    # predicted preference scores (logits before sigmoid)
                    u_emb = self.user_emb(u_pos)
                    i_emb = self.item_emb(i_pos)
                    j_emb = self.item_emb(j_neg)
                    # Use dot products as scores
                    s_ui = (u_emb * i_emb).sum(dim=1)
                    s_uj = (u_emb * j_emb).sum(dim=1)
                    # BPR loss: -log sigma(s_ui - s_uj)
                    loss = -torch.nn.functional.logsigmoid(s_ui - s_uj).mean()
                    loss.backward()
                    self.optimizer.step()
            self.result.train_loss = float(loss.detach().cpu().item()) if 'loss' in locals() else 0.0
        elif self.loss_type == "softmax":
            # Sampled softmax over (1 positive + K negatives) per positive pair
            pos_mask = (y > 0)
            up = users[pos_mask]
            ip = items[pos_mask]
            bsz = max(256, self.batch_size)
            neg_k = max(1, self.neg_k)
            ce = nn.CrossEntropyLoss()
            for _ in range(self.epochs):
                perm = torch.randperm(up.size(0), device=self.device)
                for start in range(0, up.size(0), bsz):
                    idx = perm[start:start+bsz]
                    u_pos = up[idx]
                    i_pos = ip[idx]
                    # Vectorized: sample [batch_size, neg_k] negatives at once.
                    # For large item catalogs, collision probability with positives is negligible.
                    j_neg = torch.randint(0, self.num_items, (u_pos.size(0), neg_k), device=self.device)  # [B, K]

                    self.optimizer.zero_grad()
                    u_emb = self.user_emb(u_pos)                      # [B, D]
                    i_pos_emb = self.item_emb(i_pos)                 # [B, D]
                    i_neg_emb = self.item_emb(j_neg)                 # [B, K, D]
                    s_pos = (u_emb * i_pos_emb).sum(dim=1, keepdim=True)      # [B, 1]
                    s_neg = (u_emb.unsqueeze(1) * i_neg_emb).sum(dim=2)       # [B, K]
                    logits = torch.cat([s_pos, s_neg], dim=1)                 # [B, 1+K]
                    target = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)  # pos at index 0
                    loss = ce(logits, target)
                    loss.backward()
                    self.optimizer.step()
            self.result.train_loss = float(loss.detach().cpu().item()) if 'loss' in locals() else 0.0
        else:
            dataset = torch.utils.data.TensorDataset(users, items, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
            for _ in range(self.epochs):
                for batch_users, batch_items, batch_y in loader:
                    self.optimizer.zero_grad()
                    preds = self._forward(batch_users, batch_items)
                    loss = self.loss_fn(preds, torch.clamp(batch_y, 0.0, 1.0).unsqueeze(1))
                    loss.backward()
                    self.optimizer.step()
            self.result.train_loss = float(loss.detach().cpu().item()) if 'loss' in locals() else 0.0

    def _forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute model predictions for user-item pairs.

        Parameters
        ----------
        user_ids : torch.Tensor
            User indices.
        item_ids : torch.Tensor
            Item indices.

        Returns
        -------
        torch.Tensor
            Sigmoid-activated predictions in [0, 1].
        """
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=1)
        logits = self.mlp(x)
        return self.sigmoid(logits)

    def infer(self, *, user_ids: torch.Tensor, item_ids: torch.Tensor, **_) -> torch.Tensor:
        """Score all candidate items for a given user.

        Parameters
        ----------
        user_ids : torch.Tensor
            User index.
        item_ids : torch.Tensor
            Candidate item indices.

        Returns
        -------
        torch.Tensor
            Predicted scores of shape (1, num_items), in [0, 1].
        """
        with torch.no_grad():
            users = user_ids.to(self.device).expand(item_ids.numel())
            preds = self._forward(users, item_ids.to(self.device))
        return preds.view(1, -1)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        """Select top-k items by predicted score.

        Parameters
        ----------
        logits : torch.Tensor
            Score tensor from infer().
        top_k : int
            Number of items to select.
        item_ids : torch.Tensor
            Candidate item IDs.

        Returns
        -------
        tuple
            (list of selected item IDs, metadata dict)
        """
        top = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in top], {"policy": "neural_mf"}


class ExplicitMFBaseline(BaseBaseline):
    """Matrix factorization for explicit ratings/feedback.

    Trains embeddings to predict real-valued user-item ratings using MSE loss.
    Predictions are centered around the global mean rating. Uses mini-batch
    training with learning rate scheduling for better convergence.

    Parameters
    ----------
    num_users : int
        Number of unique users.
    num_items : int
        Number of unique items.
    device : torch.device
        Torch device for computation.
    emb_dim : int, optional
        Embedding dimension (default: 64).
    lr : float, optional
        Adam learning rate (default: 0.005).
    epochs : int, optional
        Number of training epochs (default: 20).
    weight_decay : float, optional
        L2 regularization coefficient (default: 1e-5).
    batch_size : int, optional
        Mini-batch size for training (default: 512).
    """

    def __init__(self, num_users: int, num_items: int, device: torch.device,
                 embedding_dim: int = 100, learning_rate: float = 3e-3, epochs: int = 30,
                 weight_decay: float = 1e-4, batch_size: int = 512,
                 *, emb_dim: Optional[int] = None, lr: Optional[float] = None):
        super().__init__(device)
        # Support abbreviated aliases
        embedding_dim = emb_dim if emb_dim is not None else embedding_dim
        learning_rate = lr if lr is not None else learning_rate
        # Use the shared MF with implicit=False so it outputs raw scores (no sigmoid)
        self.model = MatrixFactorization(num_users, num_items, emb_dim=embedding_dim, implicit=False).to(device)
        self.lr = float(learning_rate)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self._loss_fn = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self.epochs)
        self._global_mean: float = 0.0
        self._min_rating: float = 0.0
        self._max_rating: float = 1.0

    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], ratings: Iterable[float]) -> None:
        """Train the explicit rating prediction model.

        Parameters
        ----------
        user_ids : Iterable[int]
            User indices.
        item_ids : Iterable[int]
            Item indices.
        ratings : Iterable[float]
            Real-valued rating targets.
        """
        users = torch.tensor(list(user_ids), dtype=torch.long, device=self.device)
        items = torch.tensor(list(item_ids), dtype=torch.long, device=self.device)
        y = torch.tensor(list(ratings), dtype=torch.float32, device=self.device)
        # track rating scale + mean for calibration
        self._global_mean = float(y.mean().item())
        self._min_rating = float(y.min().item())
        self._max_rating = float(y.max().item())
        y_centered = y - self._global_mean
        dataset = torch.utils.data.TensorDataset(users, items, y_centered)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            for batch_users, batch_items, batch_y in loader:
                self._optimizer.zero_grad()
                preds = self.model(batch_users, batch_items)
                loss = self._loss_fn(preds, batch_y)
                loss.backward()
                self._optimizer.step()
            self._scheduler.step()
        self.result.train_loss = float(loss.detach().item())

    def infer(self, *, user_ids: torch.Tensor, item_ids: torch.Tensor, **_) -> torch.Tensor:
        """Predict ratings for candidate items.

        Parameters
        ----------
        user_ids : torch.Tensor
            User index.
        item_ids : torch.Tensor
            Candidate item indices.

        Returns
        -------
        torch.Tensor
            Predicted ratings of shape (1, num_items), clamped to observed range.
        """
        # Score a batch of items for a given user (or vector of users broadcast)
        users = user_ids.expand(item_ids.numel())
        items = item_ids
        with torch.no_grad():
            scores = self.model(users, items) + self._global_mean
            # clamp to observed rating range for stability in RMSE
            scores = torch.clamp(scores, min=self._min_rating, max=self._max_rating)
        return scores.view(1, -1)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        """Select top-k items by predicted rating.

        Parameters
        ----------
        logits : torch.Tensor
            Score tensor from infer().
        top_k : int
            Number of items to select.
        item_ids : torch.Tensor
            Candidate item IDs.

        Returns
        -------
        tuple
            (list of selected item IDs, metadata dict)
        """
        top = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in top], {"policy": "explicit_mf"}


__all__ = [
    "BaseBaseline",
    "BaselineResult",
    "MatrixFactorization",
    "PopularityBaseline",
    "RandomBaseline",
    "ALSBaseline",
    "ExplicitMFBaseline",
    "UserKNNBaseline",
    "LinUCBBaseline",
    "ImplicitALSBaseline",
    "ImplicitBPRBaseline",
    "NeuralMatrixFactorizationBaseline",
]
