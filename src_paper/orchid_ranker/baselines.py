"""Baseline recommenders (non-adaptive) used in experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


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


__all__ = [
    "MatrixFactorization",
    "PopularityBaseline",
    "RandomBaseline",
    "ALSBaseline",
    "UserKNNBaseline",
    "LinUCBBaseline",
]