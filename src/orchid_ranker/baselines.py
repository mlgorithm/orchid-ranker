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
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 32, implicit: bool = True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.implicit = bool(implicit)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
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
    train_loss: Optional[float] = None


class BaseBaseline:
    def __init__(self, device: torch.device):
        self.device = device
        self.user_matrix = None
        self.result = BaselineResult()

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - default no-op
        return None

    def infer(self, **kwargs):  # pragma: no cover - override in subclasses
        raise NotImplementedError

    def decide(self, **kwargs):  # pragma: no cover
        raise NotImplementedError


class PopularityBaseline(BaseBaseline):
    def __init__(self, popularity: Dict[int, float], device: torch.device):
        super().__init__(device)
        self.popularity = popularity

    def infer(self, *, item_ids: torch.Tensor, **_):
        scores = [self.popularity.get(int(i.item()), 0.0) for i in item_ids]
        return torch.tensor(scores, dtype=torch.float32, device=self.device).unsqueeze(0)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        order = torch.argsort(logits[0], descending=True)
        chosen = order[:top_k]
        return [int(item_ids[i].item()) for i in chosen], {"policy": "popularity"}


class RandomBaseline(BaseBaseline):
    def infer(self, *, item_ids: torch.Tensor, **_):
        return torch.rand((1, item_ids.numel()), device=self.device)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        chosen = torch.randperm(item_ids.numel(), device=self.device)[:top_k]
        return [int(item_ids[i].item()) for i in chosen], {"policy": "random"}


class ALSBaseline(BaseBaseline):
    def __init__(self, num_users: int, num_items: int, device: torch.device, emb_dim: int = 32, lr: float = 1e-2, epochs: int = 5):
        super().__init__(device)
        self.model = MatrixFactorization(num_users, num_items, emb_dim=emb_dim, implicit=True).to(device)
        self.lr = lr
        self.epochs = epochs

    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], labels: Iterable[int]) -> None:
        user_tensor = torch.tensor(list(user_ids), dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(list(item_ids), dtype=torch.long, device=self.device)
        label_tensor = torch.tensor(list(labels), dtype=torch.float32, device=self.device)
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
        users = user_ids.expand(item_ids.numel())
        items = item_ids
        with torch.no_grad():
            scores = self.model(users, items)
        return scores.view(1, -1)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        top = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in top], {"policy": "als"}


class UserKNNBaseline(BaseBaseline):
    def __init__(self, user_item_matrix: np.ndarray, device: torch.device, k: int = 20):
        super().__init__(device)
        self.matrix = user_item_matrix.astype(np.float32)
        self.k = k

    def infer(self, *, user_ids: torch.Tensor, item_ids: torch.Tensor, **_):
        uid = int(user_ids.item())
        user_vector = self.matrix[uid]
        norms = np.linalg.norm(self.matrix, axis=1) + 1e-8
        sims = (self.matrix @ user_vector) / (norms * np.linalg.norm(user_vector) + 1e-8)
        top_neighbors = np.argsort(sims)[-self.k :]
        neighbor_pref = self.matrix[top_neighbors].mean(axis=0)
        scores = neighbor_pref[[int(i.item()) for i in item_ids]]
        return torch.tensor(scores, dtype=torch.float32, device=self.device).unsqueeze(0)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        order = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in order], {"policy": "user_knn"}


class LinUCBBaseline(BaseBaseline):
    def __init__(self, alpha: float, item_features: np.ndarray, device: torch.device):
        super().__init__(device)
        self.alpha = alpha
        self.item_features = torch.tensor(item_features, dtype=torch.float32, device=device)
        d = self.item_features.shape[1] if self.item_features.shape[1] > 0 else 1
        self.A = torch.eye(d, device=device)
        self.b = torch.zeros(d, device=device)

    def fit(self, rewards: Dict[int, float]) -> None:
        if self.item_features.shape[1] == 0:
            return
        for iid, r in rewards.items():
            feature = self.item_features[iid]
            self.A += feature.unsqueeze(1) @ feature.unsqueeze(0)
            self.b += float(r) * feature

    def infer(self, *, item_ids: torch.Tensor, **_):
        if self.item_features.shape[1] == 0:
            return torch.zeros((1, item_ids.numel()), device=self.device)
        A_inv = torch.inverse(self.A + 1e-6 * torch.eye(self.A.shape[0], device=self.device))
        theta = A_inv @ self.b
        feats = self.item_features[item_ids]
        means = feats @ theta
        ucb = self.alpha * torch.sqrt((feats @ A_inv) * feats).sum(dim=1)
        scores = means + ucb
        return scores.unsqueeze(0)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **kwargs):
        order = torch.argsort(logits[0], descending=True)[:top_k]
        chosen = [int(item_ids[i].item()) for i in order]
        return chosen, {"policy": "linucb"}


class _ImplicitBase(BaseBaseline):
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
        import scipy.sparse

        rows = np.asarray(list(user_ids), dtype=np.int32)
        cols = np.asarray(list(item_ids), dtype=np.int32)
        data = np.asarray(list(labels), dtype=np.float32)
        return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_users, num_items))

    def infer(self, *, user_ids: torch.Tensor, item_ids: torch.Tensor, **_) -> torch.Tensor:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Implicit model has not been trained")
        u = self.user_factors[user_ids.cpu().numpy()]
        i = self.item_factors[item_ids.cpu().numpy()]
        scores = np.dot(u, i.T)
        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        top = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in top], {"policy": type(self).__name__.lower()}


class ImplicitALSBaseline(_ImplicitBase):
    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], labels: Iterable[float], *, num_users: int, num_items: int) -> None:
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
    def __init__(self, *, factors: int = 64, iterations: int = 50, learning_rate: float = 0.01, regularization: float = 0.01, **kwargs):
        super().__init__(factors=factors, iterations=iterations, regularization=regularization, **kwargs)
        self.learning_rate = float(learning_rate)

    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], labels: Iterable[float], *, num_users: int, num_items: int) -> None:
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
    def __init__(
        self,
        num_users: int,
        num_items: int,
        device: torch.device,
        emb_dim: int = 32,
        hidden: Tuple[int, ...] = (64, 32),
        epochs: int = 5,
        lr: float = 1e-3,
        loss: str = "bce",  # "bce" (default), "bpr", or "softmax"
        neg_k: int = 10,
        batch_size: int = 256,
    ) -> None:
        super().__init__(device)
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.hidden = tuple(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.loss_type = str(loss).lower()
        self.neg_k = int(neg_k)
        self.batch_size = int(batch_size)

        layers: list[nn.Module] = []
        in_dim = emb_dim * 2
        for h in self.hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers).to(self.device)

        self.user_emb = nn.Embedding(num_users, emb_dim).to(self.device)
        self.item_emb = nn.Embedding(num_items, emb_dim).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()

    def parameters(self):
        return list(self.mlp.parameters()) + list(self.user_emb.parameters()) + list(self.item_emb.parameters())

    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], labels: Iterable[float]) -> None:
        users = torch.tensor(list(user_ids), dtype=torch.long, device=self.device)
        items = torch.tensor(list(item_ids), dtype=torch.long, device=self.device)
        y = torch.tensor(list(labels), dtype=torch.float32, device=self.device)
        if self.loss_type == "bpr":
            # Build per-user positive sets for negative sampling
            import numpy as np
            pos_df = torch.stack([users, items, (y > 0).float()], dim=1).detach().cpu().numpy()
            by_user: dict[int, set[int]] = {}
            for u, i, lbl in pos_df:
                if lbl >= 0.5:
                    by_user.setdefault(int(u), set()).add(int(i))
            all_items = np.arange(self.num_items, dtype=np.int64)
            bsz = max(256, self.batch_size)
            for _ in range(self.epochs):
                perm = torch.randperm(users.size(0), device=self.device)
                for start in range(0, users.size(0), bsz):
                    idx = perm[start:start+bsz]
                    u_pos = users[idx]
                    i_pos = items[idx]
                    # Sample one negative per positive
                    j_neg = []
                    for u in u_pos.detach().cpu().numpy():
                        pos_set = by_user.get(int(u), set())
                        # ensure at least one negative exists
                        cand = all_items
                        # sample until not in positives (cap tries)
                        for _tries in range(10):
                            j = int(np.random.randint(0, self.num_items))
                            if j not in pos_set:
                                j_neg.append(j)
                                break
                        else:
                            j_neg.append(int(np.random.randint(0, self.num_items)))
                    j_neg = torch.tensor(j_neg, dtype=torch.long, device=self.device)
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
            import numpy as np
            pos_mask = (y > 0)
            up = users[pos_mask]
            ip = items[pos_mask]
            # build positives per user for negative sampling
            by_user: dict[int, set[int]] = {}
            for uu, ii in zip(up.detach().cpu().numpy(), ip.detach().cpu().numpy()):
                by_user.setdefault(int(uu), set()).add(int(ii))
            bsz = max(256, self.batch_size)
            neg_k = max(1, self.neg_k)
            ce = nn.CrossEntropyLoss()
            for _ in range(self.epochs):
                perm = torch.randperm(up.size(0), device=self.device)
                for start in range(0, up.size(0), bsz):
                    idx = perm[start:start+bsz]
                    u_pos = up[idx]
                    i_pos = ip[idx]
                    # sample K negatives per positive
                    neg_list = []
                    for u in u_pos.detach().cpu().numpy():
                        pos_set = by_user.get(int(u), set())
                        # sample without replacement if possible
                        candidates = []
                        trials = 0
                        while len(candidates) < neg_k and trials < neg_k * 5:
                            j = int(np.random.randint(0, self.num_items))
                            if j not in pos_set:
                                candidates.append(j)
                            trials += 1
                        # pad if needed
                        while len(candidates) < neg_k:
                            candidates.append(int(np.random.randint(0, self.num_items)))
                        neg_list.append(candidates)
                    j_neg = torch.tensor(neg_list, dtype=torch.long, device=self.device)  # [B, K]

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
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=1)
        logits = self.mlp(x)
        return self.sigmoid(logits)

    def infer(self, *, user_ids: torch.Tensor, item_ids: torch.Tensor, **_) -> torch.Tensor:
        with torch.no_grad():
            preds = self._forward(user_ids.to(self.device), item_ids.to(self.device))
        return preds.view(1, -1)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        top = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in top], {"policy": "neural_mf"}


__all__ = [
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

class ExplicitMFBaseline(BaseBaseline):
    def __init__(self, num_users: int, num_items: int, device: torch.device, emb_dim: int = 64, lr: float = 1e-3, epochs: int = 10, weight_decay: float = 1e-4):
        super().__init__(device)
        # Use the shared MF with implicit=False so it outputs raw scores (no sigmoid)
        self.model = MatrixFactorization(num_users, num_items, emb_dim=emb_dim, implicit=False).to(device)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self._loss_fn = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._global_mean: float = 0.0
        self._min_rating: float = 0.0
        self._max_rating: float = 1.0

    def fit(self, user_ids: Iterable[int], item_ids: Iterable[int], ratings: Iterable[float]) -> None:
        users = torch.tensor(list(user_ids), dtype=torch.long, device=self.device)
        items = torch.tensor(list(item_ids), dtype=torch.long, device=self.device)
        y = torch.tensor(list(ratings), dtype=torch.float32, device=self.device)
        # track rating scale + mean for calibration
        self._global_mean = float(y.mean().detach().cpu().item())
        self._min_rating = float(y.min().detach().cpu().item())
        self._max_rating = float(y.max().detach().cpu().item())
        y_centered = y - self._global_mean
        for _ in range(self.epochs):
            self._optimizer.zero_grad()
            preds = self.model(users, items)
            loss = self._loss_fn(preds, y_centered)
            loss.backward()
            self._optimizer.step()
        self.result.train_loss = float(loss.detach().cpu().item())

    def infer(self, *, user_ids: torch.Tensor, item_ids: torch.Tensor, **_) -> torch.Tensor:
        # Score a batch of items for a given user (or vector of users broadcast)
        users = user_ids.expand(item_ids.numel())
        items = item_ids
        with torch.no_grad():
            scores = self.model(users, items) + self._global_mean
            # clamp to observed rating range for stability in RMSE
            scores = torch.clamp(scores, min=self._min_rating, max=self._max_rating)
        return scores.view(1, -1)

    def decide(self, *, logits: torch.Tensor, top_k: int, item_ids: torch.Tensor, **_):
        top = torch.argsort(logits[0], descending=True)[:top_k]
        return [int(item_ids[i].item()) for i in top], {"policy": "explicit_mf"}
