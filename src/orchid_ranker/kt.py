"""Experimental knowledge tracing models for adaptive learning.

This module starts Orchid's modern adaptive-learning algorithm layer with a
compact SAKT-style tracer. The implementation is intentionally small and
production-oriented: it predicts correctness for candidate items from a
learner's recent interaction sequence and exposes a simple practice-ranking
helper for stretch-zone recommendation.

The API is experimental. Import from this submodule directly:

    from orchid_ranker.kt import SAKTTracer
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._compat import require_torch

require_torch("orchid_ranker.kt")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

__all__ = [
    "AKTTracer",
    "KTRecommendation",
    "SAKTTrainingExample",
    "SAKTTracer",
    "build_sakt_examples",
]


@dataclass(frozen=True)
class SAKTTrainingExample:
    """One supervised next-response example for SAKT training."""

    user_id: Any
    query_item_id: Any
    label: int
    history_item_ids: Tuple[Any, ...]
    history_correct: Tuple[int, ...]


@dataclass(frozen=True)
class KTRecommendation:
    """Practice recommendation scored by predicted correctness and stretch fit."""

    item_id: Any
    p_correct: float
    score: float


def _label(value: Any, threshold: float) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("correct labels must be finite")
    return int(numeric >= threshold)


def build_sakt_examples(
    interactions: pd.DataFrame,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    correct_col: str = "correct",
    timestamp_col: Optional[str] = None,
    max_seq_len: int = 50,
    correct_threshold: float = 0.5,
) -> List[SAKTTrainingExample]:
    """Build leakage-safe next-response examples from learner interactions.

    For each user sequence, the example at position ``t`` uses only events
    before ``t`` as history and the event at ``t`` as the query label. The
    first event for each user is skipped because it has no prior history.
    """
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be >= 1")

    required = {user_col, item_col, correct_col}
    if timestamp_col is not None:
        required.add(timestamp_col)
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions missing required columns: {sorted(missing)}")

    work = interactions.copy()
    work["__orchid_order__"] = np.arange(len(work))
    sort_cols = [user_col]
    if timestamp_col is not None:
        sort_cols.append(timestamp_col)
    sort_cols.append("__orchid_order__")
    work = work.sort_values(sort_cols, kind="mergesort")

    examples: List[SAKTTrainingExample] = []
    for user_id, group in work.groupby(user_col, sort=False):
        history_items: List[Any] = []
        history_correct: List[int] = []
        for item_id, raw_correct in group[[item_col, correct_col]].itertuples(index=False, name=None):
            correct = _label(raw_correct, correct_threshold)
            if history_items:
                examples.append(
                    SAKTTrainingExample(
                        user_id=user_id,
                        query_item_id=item_id,
                        label=correct,
                        history_item_ids=tuple(history_items[-max_seq_len:]),
                        history_correct=tuple(history_correct[-max_seq_len:]),
                    )
                )
            history_items.append(item_id)
            history_correct.append(correct)
    return examples


class _SAKTModel(nn.Module):
    """Minimal self-attentive knowledge tracing model."""

    def __init__(
        self,
        *,
        num_items: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.interaction_emb = nn.Embedding(2 * self.num_items + 1, d_model, padding_idx=0)
        self.query_item_emb = nn.Embedding(self.num_items + 1, d_model, padding_idx=0)
        self.position_emb = nn.Embedding(self.max_seq_len, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(
        self,
        history_items: torch.Tensor,
        history_correct: torch.Tensor,
        query_items: torch.Tensor,
    ) -> torch.Tensor:
        pad_mask = history_items.eq(0)
        interaction_codes = history_items + history_correct.long() * self.num_items
        interaction_codes = interaction_codes.masked_fill(pad_mask, 0)

        positions = torch.arange(self.max_seq_len, device=history_items.device).unsqueeze(0)
        x = self.interaction_emb(interaction_codes) + self.position_emb(positions)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        key_padding_mask = pad_mask.clone()
        all_padding = key_padding_mask.all(dim=1)
        if bool(all_padding.any()):
            key_padding_mask[all_padding, 0] = False

        query = self.query_item_emb(query_items).unsqueeze(1)
        attn_out, _ = self.attention(
            query,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.norm1(query + attn_out)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return self.out(hidden.squeeze(1)).squeeze(-1)


class _AKTModel(nn.Module):
    """Compact AKT-inspired model with difficulty and monotonic attention."""

    def __init__(
        self,
        *,
        num_items: int,
        max_seq_len: int,
        d_model: int,
        dropout: float,
        item_difficulty: torch.Tensor,
        monotonic_decay: float,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.interaction_emb = nn.Embedding(2 * self.num_items + 1, d_model, padding_idx=0)
        self.query_item_emb = nn.Embedding(self.num_items + 1, d_model, padding_idx=0)
        self.position_emb = nn.Embedding(self.max_seq_len, d_model)
        self.difficulty_proj = nn.Linear(1, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)
        self.register_buffer("item_difficulty", item_difficulty.float())
        self.monotonic_decay = nn.Parameter(torch.tensor(float(monotonic_decay)))
        self.scale = float(d_model) ** -0.5

    def forward(
        self,
        history_items: torch.Tensor,
        history_correct: torch.Tensor,
        query_items: torch.Tensor,
    ) -> torch.Tensor:
        pad_mask = history_items.eq(0)
        interaction_codes = history_items + history_correct.long() * self.num_items
        interaction_codes = interaction_codes.masked_fill(pad_mask, 0)

        positions = torch.arange(self.max_seq_len, device=history_items.device).unsqueeze(0)
        hist_difficulty = self.item_difficulty.index_select(0, history_items.reshape(-1)).view(
            history_items.shape[0], self.max_seq_len, 1
        )
        x = (
            self.interaction_emb(interaction_codes)
            + self.position_emb(positions)
            + self.difficulty_proj(hist_difficulty)
        )
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        query_difficulty = self.item_difficulty.index_select(0, query_items).unsqueeze(-1)
        query = self.query_item_emb(query_items) + self.difficulty_proj(query_difficulty)
        q = self.query_proj(query)
        k = self.key_proj(x)
        v = self.value_proj(x)

        scores = torch.sum(k * q.unsqueeze(1), dim=-1) * self.scale
        distances = torch.arange(
            self.max_seq_len - 1,
            -1,
            -1,
            device=history_items.device,
            dtype=scores.dtype,
        )
        distances = distances / max(1, self.max_seq_len - 1)
        scores = scores - torch.nn.functional.softplus(self.monotonic_decay) * distances

        key_padding_mask = pad_mask.clone()
        all_padding = key_padding_mask.all(dim=1)
        if bool(all_padding.any()):
            key_padding_mask[all_padding, -1] = False
        scores = scores.masked_fill(key_padding_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), v).squeeze(1)

        hidden = self.norm1(query + self.dropout(context))
        hidden = self.norm2(hidden + self.ffn(hidden))
        return self.out(hidden).squeeze(-1)


class SAKTTracer:
    """SAKT-style knowledge tracer for adaptive-learning recommendation.

    The tracer trains on learner event sequences and predicts the probability
    that a learner will answer a candidate item correctly. It is designed as a
    compact experimental baseline, not a benchmark-validated SOTA claim.
    """

    def __init__(
        self,
        *,
        max_seq_len: int = 50,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 128,
        correct_threshold: float = 0.5,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        if n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.correct_threshold = float(correct_threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state

        self.model: Optional[nn.Module] = None
        self.training_examples_: List[SAKTTrainingExample] = []
        self.result_: Dict[str, float] = {}
        self._item2idx: Dict[Any, int] = {}
        self._idx2item: Dict[int, Any] = {}
        self._histories: Dict[Any, List[Tuple[int, int]]] = {}

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    @property
    def item_ids_(self) -> List[Any]:
        return [self._idx2item[i] for i in sorted(self._idx2item)]

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        correct_col: str = "correct",
        timestamp_col: Optional[str] = None,
    ) -> "SAKTTracer":
        """Fit the tracer from historical learner outcomes."""
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            np.random.seed(int(self.random_state))

        examples = build_sakt_examples(
            interactions,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
            max_seq_len=self.max_seq_len,
            correct_threshold=self.correct_threshold,
        )
        if not examples:
            raise ValueError("SAKTTracer requires at least one user with two or more interactions")

        item_ids = sorted(interactions[item_col].drop_duplicates().tolist(), key=lambda value: str(value))
        self._item2idx = {item_id: idx + 1 for idx, item_id in enumerate(item_ids)}
        self._idx2item = {idx: item_id for item_id, idx in self._item2idx.items()}
        self.training_examples_ = examples
        self._after_item_mapping(interactions, item_col=item_col)
        self._histories = self._build_histories(
            interactions,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
        )

        history_items, history_correct, query_items, labels = self._encode_examples(examples)
        dataset = torch.utils.data.TensorDataset(history_items, history_correct, query_items, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._make_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        last_loss = 0.0
        for _ in range(self.epochs):
            for batch_history, batch_correct, batch_query, batch_labels in loader:
                batch_history = batch_history.to(self.device)
                batch_correct = batch_correct.to(self.device)
                batch_query = batch_query.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_history, batch_correct, batch_query)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
        self.result_ = {"train_loss": last_loss, "num_examples": float(len(examples))}
        return self

    def _after_item_mapping(self, interactions: pd.DataFrame, *, item_col: str) -> None:
        del interactions, item_col

    def _make_model(self) -> nn.Module:
        return _SAKTModel(
            num_items=len(self._item2idx),
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )

    def predict_correct(self, user_id: Any, item_id: Any) -> float:
        """Predict the probability that ``user_id`` answers ``item_id`` correctly."""
        return float(self.predict_many(user_id, [item_id])[item_id])

    def predict_many(self, user_id: Any, item_ids: Sequence[Any]) -> Dict[Any, float]:
        """Predict correctness probabilities for a candidate item sequence."""
        self._require_fitted()
        if not item_ids:
            return {}
        internal_items = [self._internal_item_id(item_id) for item_id in item_ids]
        probs = self._predict_internal(user_id, internal_items)
        return {item_id: float(prob) for item_id, prob in zip(item_ids, probs)}

    def state_vector(self, user_id: Any, candidate_item_ids: Optional[Sequence[Any]] = None) -> np.ndarray:
        """Return predicted-correctness probabilities as a learner state vector."""
        candidates = list(candidate_item_ids) if candidate_item_ids is not None else self.item_ids_
        if not candidates:
            return np.zeros((0,), dtype=np.float32)
        values = [self.predict_correct(user_id, item_id) for item_id in candidates]
        return np.asarray(values, dtype=np.float32)

    def recommend_practice(
        self,
        user_id: Any,
        candidate_item_ids: Sequence[Any],
        *,
        top_k: int = 5,
        target_correct: float = 0.70,
    ) -> List[KTRecommendation]:
        """Rank practice items by stretch-zone fit around ``target_correct``."""
        if not 0.0 <= target_correct <= 1.0:
            raise ValueError("target_correct must be in [0, 1]")
        predictions = self.predict_many(user_id, candidate_item_ids)
        ranked = [
            KTRecommendation(
                item_id=item_id,
                p_correct=prob,
                score=1.0 - abs(prob - target_correct),
            )
            for item_id, prob in predictions.items()
        ]
        ranked.sort(key=lambda rec: (rec.score, rec.p_correct, str(rec.item_id)), reverse=True)
        return ranked[: max(0, int(top_k))]

    def observe(self, user_id: Any, item_id: Any, correct: Any) -> int:
        """Append one live outcome to the in-memory learner history."""
        self._require_fitted()
        internal_item = self._internal_item_id(item_id)
        outcome = _label(correct, self.correct_threshold)
        history = self._histories.setdefault(user_id, [])
        history.append((internal_item, outcome))
        if len(history) > self.max_seq_len:
            del history[: len(history) - self.max_seq_len]
        return len(history)

    def history_for(self, user_id: Any) -> List[Tuple[Any, int]]:
        """Return external item IDs and correctness labels for a learner history."""
        history = self._histories.get(user_id, [])
        return [(self._idx2item[item_idx], correct) for item_idx, correct in history]

    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("SAKTTracer must be fitted before prediction")

    def _internal_item_id(self, item_id: Any) -> int:
        try:
            return self._item2idx[item_id]
        except KeyError as exc:
            raise KeyError(f"Unknown item_id={item_id!r}") from exc

    def _build_histories(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str,
        item_col: str,
        correct_col: str,
        timestamp_col: Optional[str],
    ) -> Dict[Any, List[Tuple[int, int]]]:
        work = interactions.copy()
        work["__orchid_order__"] = np.arange(len(work))
        sort_cols = [user_col]
        if timestamp_col is not None:
            sort_cols.append(timestamp_col)
        sort_cols.append("__orchid_order__")
        work = work.sort_values(sort_cols, kind="mergesort")

        histories: Dict[Any, List[Tuple[int, int]]] = {}
        for user_id, group in work.groupby(user_col, sort=False):
            rows: List[Tuple[int, int]] = []
            for item_id, raw_correct in group[[item_col, correct_col]].itertuples(index=False, name=None):
                rows.append(
                    (
                        self._internal_item_id(item_id),
                        _label(raw_correct, self.correct_threshold),
                    )
                )
            histories[user_id] = rows[-self.max_seq_len:]
        return histories

    def _encode_examples(
        self,
        examples: Sequence[SAKTTrainingExample],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_items = np.zeros((len(examples), self.max_seq_len), dtype=np.int64)
        history_correct = np.zeros((len(examples), self.max_seq_len), dtype=np.int64)
        query_items = np.zeros((len(examples),), dtype=np.int64)
        labels = np.zeros((len(examples),), dtype=np.float32)

        for row_idx, example in enumerate(examples):
            encoded_items = [self._internal_item_id(item_id) for item_id in example.history_item_ids]
            encoded_correct = list(example.history_correct)
            offset = self.max_seq_len - len(encoded_items)
            history_items[row_idx, offset:] = encoded_items
            history_correct[row_idx, offset:] = encoded_correct
            query_items[row_idx] = self._internal_item_id(example.query_item_id)
            labels[row_idx] = float(example.label)

        return (
            torch.as_tensor(history_items, dtype=torch.long),
            torch.as_tensor(history_correct, dtype=torch.long),
            torch.as_tensor(query_items, dtype=torch.long),
            torch.as_tensor(labels, dtype=torch.float32),
        )

    def _history_tensors(self, user_id: Any, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        history = self._histories.get(user_id, [])[-self.max_seq_len:]
        items = np.zeros((1, self.max_seq_len), dtype=np.int64)
        correct = np.zeros((1, self.max_seq_len), dtype=np.int64)
        if history:
            offset = self.max_seq_len - len(history)
            items[0, offset:] = [item for item, _ in history]
            correct[0, offset:] = [label for _, label in history]
        item_tensor = torch.as_tensor(items, dtype=torch.long, device=self.device).repeat(count, 1)
        correct_tensor = torch.as_tensor(correct, dtype=torch.long, device=self.device).repeat(count, 1)
        return item_tensor, correct_tensor

    def _predict_internal(self, user_id: Any, internal_items: Sequence[int]) -> np.ndarray:
        self._require_fitted()
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            history_items, history_correct = self._history_tensors(user_id, len(internal_items))
            query = torch.as_tensor(internal_items, dtype=torch.long, device=self.device)
            logits = self.model(history_items, history_correct, query)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return probs.astype(np.float32)


class AKTTracer(SAKTTracer):
    """AKT-inspired tracer with difficulty-aware monotonic attention.

    This experimental model keeps the same public API as :class:`SAKTTracer`
    while adding two AKT-style ingredients:

    - item difficulty embeddings, supplied from a difficulty column or mapping
    - recency-biased monotonic attention over the learner history

    It is intentionally named "AKT-inspired" in docs until benchmarked on
    public adaptive-learning datasets.
    """

    def __init__(
        self,
        *,
        max_seq_len: int = 50,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 128,
        correct_threshold: float = 0.5,
        monotonic_decay: float = 0.25,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        del n_heads  # Custom single-query attention does not use multi-head splitting.
        super().__init__(
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=1,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            correct_threshold=correct_threshold,
            device=device,
            random_state=random_state,
        )
        if monotonic_decay < 0:
            raise ValueError("monotonic_decay must be non-negative")
        self.monotonic_decay = float(monotonic_decay)
        self.item_difficulty_: Dict[Any, float] = {}
        self._item_difficulty_tensor: Optional[torch.Tensor] = None
        self._fit_item_difficulty_col: Optional[str] = None
        self._fit_item_difficulty_map: Optional[Dict[Any, float]] = None

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        correct_col: str = "correct",
        timestamp_col: Optional[str] = None,
        item_difficulty_col: Optional[str] = None,
        item_difficulty_map: Optional[Dict[Any, float]] = None,
    ) -> "AKTTracer":
        """Fit the tracer from learner outcomes and optional item difficulty."""
        if item_difficulty_col is not None and item_difficulty_col not in interactions.columns:
            raise ValueError(f"item_difficulty_col={item_difficulty_col!r} not present in interactions")
        self._fit_item_difficulty_col = item_difficulty_col
        self._fit_item_difficulty_map = dict(item_difficulty_map or {})
        super().fit(
            interactions,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
        )
        return self

    def _after_item_mapping(self, interactions: pd.DataFrame, *, item_col: str) -> None:
        difficulty: Dict[Any, float] = {item_id: 0.5 for item_id in self._item2idx}
        if self._fit_item_difficulty_col is not None:
            means = interactions.groupby(item_col)[self._fit_item_difficulty_col].mean()
            difficulty.update({item_id: float(value) for item_id, value in means.items()})
        difficulty.update(self._fit_item_difficulty_map or {})

        values = np.zeros((len(self._item2idx) + 1,), dtype=np.float32)
        for external_id, internal_id in self._item2idx.items():
            raw_value = float(difficulty.get(external_id, 0.5))
            if not np.isfinite(raw_value) or not 0.0 <= raw_value <= 1.0:
                raise ValueError(f"difficulty for item_id={external_id!r} must be finite and in [0, 1]")
            values[internal_id] = raw_value
        self.item_difficulty_ = {item_id: float(difficulty.get(item_id, 0.5)) for item_id in self._item2idx}
        self._item_difficulty_tensor = torch.as_tensor(values, dtype=torch.float32)

    def _make_model(self) -> nn.Module:
        if self._item_difficulty_tensor is None:
            raise RuntimeError("AKTTracer item difficulty tensor was not initialized")
        return _AKTModel(
            num_items=len(self._item2idx),
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            dropout=self.dropout,
            item_difficulty=self._item_difficulty_tensor,
            monotonic_decay=self.monotonic_decay,
        )
