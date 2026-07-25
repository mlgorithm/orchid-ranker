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

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

__all__ = [
    "AKTTracer",
    "DKTTracer",
    "DKVMNTracer",
    "KTRecommendation",
    "SAKTTrainingExample",
    "SAKTTracer",
    "SAINTPlusTracer",
    "SAINTTracer",
    "build_sakt_examples",
]


@dataclass(frozen=True)
class SAKTTrainingExample:
    """One full-sequence training window for knowledge tracing.

    This is the FULL-SEQUENCE training unit (the framework was migrated away from
    the old single-query "(history, one query) -> one label" form). Each window
    is a chronological slice of a single learner's interactions of length up to
    ``max_seq_len``. At every position ``t`` in the window:

    - ``item_ids[t]`` is the queried exercise (the model may see this id),
    - ``correct[t]`` is the response/target at ``t`` (the model must NOT see its
      own response when predicting position ``t`` -- enforced by causal masking),
    - the model predicts ``correct[t]`` using interactions at positions ``< t``
      plus the exercise id at ``t``.

    The model produces a prediction for EVERY position in one forward pass.

    ``query_item_id`` / ``label`` are retained as convenience views of the LAST
    (most recent) position in the window so older single-query call sites and
    introspection keep working.
    """

    user_id: Any
    item_ids: Tuple[Any, ...]
    correct: Tuple[int, ...]
    elapsed: Tuple[float, ...] = ()
    lag: Tuple[float, ...] = ()

    @property
    def query_item_id(self) -> Any:
        """The most recent (last) queried item in this window."""
        return self.item_ids[-1]

    @property
    def label(self) -> int:
        """The response/target at the most recent (last) position."""
        return self.correct[-1]

    def __len__(self) -> int:
        return len(self.item_ids)


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
    """Build leakage-safe FULL-SEQUENCE training windows from interactions.

    For each user, the chronological interaction sequence is sliced into
    non-overlapping windows of length up to ``max_seq_len``. Each returned
    :class:`SAKTTrainingExample` holds the items and responses for one window; a
    full-sequence model predicts the response at EVERY position ``t`` using only
    interactions at positions ``< t`` plus the exercise id at ``t`` (the causal,
    no-leakage contract -- see :class:`SAKTTrainingExample`).

    The per-window ``elapsed``/``lag`` temporal features are aligned to each
    window position: ``elapsed[t]`` is the gap from ``t-1`` to ``t`` and
    ``lag[t]`` is the gap from ``t`` to the LAST position of the window (the most
    recent query). Windows with a single interaction are still emitted -- a
    full-sequence model trains on position 0 with an empty (all-padded) causal
    context, which is the cold-start prediction.
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
        items: List[Any] = []
        responses: List[int] = []
        times: List[float] = []
        cols = [item_col, correct_col] if timestamp_col is None else [item_col, correct_col, timestamp_col]
        for row in group[cols].itertuples(index=False, name=None):
            items.append(row[0])
            responses.append(_label(row[1], correct_threshold))
            if timestamp_col is not None:
                times.append(_time_value(row[2]))

        # Slice the user's chronological stream into windows of <= max_seq_len.
        for start in range(0, len(items), max_seq_len):
            win_items = items[start : start + max_seq_len]
            win_correct = responses[start : start + max_seq_len]
            if timestamp_col is not None:
                win_times = times[start : start + max_seq_len]
                win_elapsed = tuple(_elapsed_history(win_times))
                query_time = float(win_times[-1])
                win_lag = tuple(max(0.0, query_time - t) for t in win_times)
            else:
                win_elapsed = ()
                win_lag = ()
            examples.append(
                SAKTTrainingExample(
                    user_id=user_id,
                    item_ids=tuple(win_items),
                    correct=tuple(win_correct),
                    elapsed=win_elapsed,
                    lag=win_lag,
                )
            )
    return examples


def _time_value(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
    else:
        result = float(pd.Timestamp(value).value) / 1_000_000_000.0
    if not np.isfinite(result):
        raise ValueError("timestamp values must be finite")
    return result


def _elapsed_history(times: Sequence[float]) -> List[float]:
    elapsed: List[float] = []
    previous: Optional[float] = None
    for value in times:
        if previous is None:
            elapsed.append(0.0)
        else:
            elapsed.append(max(0.0, float(value) - previous))
        previous = float(value)
    return elapsed


def _bucket_time_values(values: Sequence[float], max_bucket: int) -> np.ndarray:
    raw = np.asarray(list(values), dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0,), dtype=np.int64)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    buckets = np.zeros(raw.shape, dtype=np.int64)
    nonzero = raw > 0.0
    buckets[nonzero] = np.floor(np.log1p(raw[nonzero])).astype(np.int64) + 1
    return np.clip(buckets, 0, int(max_bucket)).astype(np.int64)


class _SAKTModel(nn.Module):
    """Full-sequence self-attentive knowledge tracing model (SAKT).

    Faithful to Pandey & Karypis (2019): for a sequence of interactions the
    model produces a correctness prediction at EVERY position in a single
    forward pass via causal self-attention.

    Inputs (all right-padded, shape ``(B, L)`` where ``L == max_seq_len``):

    - ``history_items[b, t]`` exercise id at position ``t`` (0 == padding),
    - ``history_correct[b, t]`` response at position ``t``,
    - ``query_items[b, t]`` the exercise id being predicted at ``t`` (for full
      training windows this equals ``history_items`` shifted appropriately; the
      caller passes the per-position query exercise ids).

    No-leakage contract (causal masking): the query at position ``t`` attends to
    interaction embeddings (exercise+response) at positions ``< t`` ONLY. It sees
    the exercise id at ``t`` (through the query embedding) but NEVER the response
    at ``t``. This is enforced by a strict lower-triangular (diagonal excluded)
    causal mask, so the prediction at ``t`` is independent of ``correct[t]``.
    """

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
        # history_items/history_correct/query_items: (B, L), right-padded with 0.
        pad_mask = history_items.eq(0)  # (B, L) True where padded interaction.
        seq_len = history_items.shape[1]
        device = history_items.device

        interaction_codes = history_items + history_correct.long() * self.num_items
        interaction_codes = interaction_codes.masked_fill(pad_mask, 0)

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        # Interaction (key/value) embeddings: exercise+response at each position.
        x = self.interaction_emb(interaction_codes) + self.position_emb(positions)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Per-position query embeddings: exercise id only (NO response).
        query = self.query_item_emb(query_items) + self.position_emb(positions)

        # Strict causal mask: position t may attend to interactions at j < t only,
        # so the response at t (and any future response) is never visible.
        # nn.MultiheadAttention treats True entries in attn_mask as "not allowed".
        # triu(diagonal=0) is True on AND above the main diagonal, which disallows
        # attending to self (j == t) and the future (j > t).
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=0,
        )  # (L, L): True on and above the diagonal == disallowed.

        # Key padding: padded interaction positions are never attended to.
        key_padding_mask = pad_mask  # (B, L)

        # A query at position t whose entire allowed window (j < t) is masked
        # (e.g. t == 0, or all-padded rows) would yield NaN from a fully-masked
        # softmax. MultiheadAttention guards padded queries, but the causal mask
        # alone leaves position 0 with no valid key. Allow each position to attend
        # to a single safe slot and zero out its contribution afterwards instead.
        attn_out, _ = self.attention(
            query,
            x,
            x,
            attn_mask=causal,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        # Rows with no valid key (fully masked) produce NaN; replace with 0 so the
        # residual leaves a pure query-embedding (cold-start) representation.
        attn_out = torch.nan_to_num(attn_out, nan=0.0)

        hidden = self.norm1(query + attn_out)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return cast(torch.Tensor, self.out(hidden).squeeze(-1))


class _AKTModel(nn.Module):
    """Full-sequence AKT tracer (Ghosh et al. 2020).

    Two defining AKT mechanisms are implemented faithfully:

    **(a) Rasch embeddings.** Each item has a LEARNED scalar difficulty
    ``mu_q`` (``difficulty`` embedding of shape ``(num_items, 1)``). The
    question embedding is ``c_c + mu_q * d_c`` where ``c_c`` is the concept
    embedding and ``d_c`` the question-variation embedding. The interaction
    embedding is the base response embedding plus ``mu_q * f_(c,r)`` where
    ``f_(c,r)`` is a learned variation over the (concept, response) interaction.
    When an external difficulty prior is provided it WARM-STARTS
    ``difficulty.weight`` and then trains freely.

    **(b) Monotonic context-aware attention.** A content-dependent, distance
    decayed multi-head attention. The decay is NOT a fixed positional ramp: it
    uses the (detached) softmaxed attention to accumulate an effective distance
    between query ``t`` and key ``tau`` and applies a learned per-head decay
    ``gammas`` (negative via ``-softplus``). See :meth:`_monotonic_attention`.

    No-leakage contract: a strict lower-triangular causal mask (diagonal
    excluded) lets position ``t`` attend to interaction embeddings at
    ``tau < t`` only. The query stream carries the exercise id (and its Rasch
    difficulty) at ``t`` but never the response at ``t``.
    """

    def __init__(
        self,
        *,
        num_items: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        item_difficulty: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.d_head = self.d_model // self.n_heads

        # Rasch embeddings.
        self.concept_emb = nn.Embedding(self.num_items + 1, d_model, padding_idx=0)  # c_c
        self.variation_emb = nn.Embedding(self.num_items + 1, d_model, padding_idx=0)  # d_c
        self.difficulty = nn.Embedding(self.num_items + 1, 1, padding_idx=0)  # mu_q (learned scalar)
        # Interaction (response) embeddings: base response embedding + variation.
        self.response_emb = nn.Embedding(2 * self.num_items + 1, d_model, padding_idx=0)
        self.response_variation_emb = nn.Embedding(2 * self.num_items + 1, d_model, padding_idx=0)

        self.position_emb = nn.Embedding(self.max_seq_len, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
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

        # Learned per-head monotonic decay (one gamma per head). Negated through
        # softplus at use-time so the effective decay is always <= 0.
        self.gammas = nn.Parameter(torch.zeros(self.n_heads, 1, 1))
        self.scale = float(self.d_head) ** -0.5

        # Optional warm-start of the learned difficulty scalar.
        if item_difficulty is not None:
            with torch.no_grad():
                init = item_difficulty.float().reshape(-1, 1)
                if init.shape[0] == self.difficulty.weight.shape[0]:
                    self.difficulty.weight.copy_(init)

    def _question_embedding(self, item: torch.Tensor) -> torch.Tensor:
        """Rasch question embedding: c_c + mu_q * d_c."""
        mu = self.difficulty(item)  # (B, L, 1)
        out: torch.Tensor = self.concept_emb(item) + mu * self.variation_emb(item)
        return out

    def _interaction_embedding(
        self, item: torch.Tensor, interaction_codes: torch.Tensor
    ) -> torch.Tensor:
        """Rasch interaction embedding: e_(c,r) + mu_q * f_(c,r)."""
        mu = self.difficulty(item)  # (B, L, 1)
        out: torch.Tensor = self.response_emb(interaction_codes) + mu * self.response_variation_emb(interaction_codes)
        return out

    def _monotonic_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: torch.Tensor,
    ) -> torch.Tensor:
        """AKT distance-decayed, content-dependent multi-head attention.

        ``q``/``k``/``v``: (B, H, L, d_head). ``causal``: (L, L) bool, True where
        attention is DISALLOWED (on and above the diagonal -> future + self).
        """
        bsz, n_heads, seq_len, _ = q.shape
        device = q.device
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        disallowed = causal.view(1, 1, seq_len, seq_len)

        with torch.no_grad():
            # Softmaxed attention restricted to the causal (strictly-past) window.
            scores_ = torch.softmax(scores.masked_fill(disallowed, -1e32), dim=-1)
            scores_ = scores_.masked_fill(disallowed, 0.0)  # (B, H, L, L)

            x1 = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1).float()
            x2 = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len).float()
            position_effect = torch.abs(x1 - x2)  # |t - tau|, (1,1,L,L)

            # AKT cumulative attention distance. For query row t, sum the
            # attention mass lying to the RIGHT of key tau (closer to t), so keys
            # far from t but with little intervening mass decay slowly.
            # disttotal_scores: total attention to the right of (and including)
            # each tau; distcum_scores: cumulative attention strictly to the left.
            distcum_scores = torch.cumsum(scores_, dim=-1)  # cumulative over tau
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)  # (B,H,L,1)
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = torch.sqrt(dist_scores.detach())

        # Learned per-head decay, strictly non-positive.
        theta = -torch.nn.functional.softplus(self.gammas)  # (H, 1, 1)
        total_effect = torch.clamp(
            torch.exp(theta.unsqueeze(0) * dist_scores), min=1e-5, max=1e5
        )  # (B, H, L, L)

        scores = scores * total_effect
        scores = scores.masked_fill(disallowed, -1e32)
        attn = torch.softmax(scores, dim=-1)
        # A fully-disallowed row (e.g. query t == 0, which may attend to nothing)
        # yields a UNIFORM softmax over all keys -- which would leak future
        # interactions. Force such rows to contribute zero context (cold start).
        fully_masked = disallowed.all(dim=-1, keepdim=True)  # (1,1,L,1)
        attn = attn.masked_fill(fully_masked, 0.0)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)
        context = torch.matmul(attn, v)  # (B, H, L, d_head)
        return context

    def forward(
        self,
        history_items: torch.Tensor,
        history_correct: torch.Tensor,
        query_items: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = history_items.shape
        device = history_items.device
        pad_mask = history_items.eq(0)
        interaction_codes = history_items + history_correct.long() * self.num_items
        interaction_codes = interaction_codes.masked_fill(pad_mask, 0)

        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # Key/value stream: Rasch interaction embeddings at each past position.
        x = self._interaction_embedding(history_items, interaction_codes) + self.position_emb(positions)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Query stream: Rasch question embedding (exercise id only, NO response).
        query = self._question_embedding(query_items) + self.position_emb(positions)

        def _heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        q = _heads(self.query_proj(query))
        k = _heads(self.key_proj(x))
        v = _heads(self.value_proj(x))

        # Strict causal mask: position t attends to tau < t only (excludes self).
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=0
        )
        context = self._monotonic_attention(q, k, v, causal)  # (B, H, L, d_head)
        context = context.transpose(1, 2).reshape(bsz, seq_len, self.d_model)

        hidden = self.norm1(query + self.dropout(context))
        hidden = self.norm2(hidden + self.ffn(hidden))
        return cast(torch.Tensor, self.out(hidden).squeeze(-1))


class _SAINTModel(nn.Module):
    """Full-sequence SAINT encoder-decoder tracer (Choi et al. 2020).

    Correct encoder/decoder routing:

    - The ENCODER ingests the EXERCISE sequence (exercise embeddings +
      positional) with causal self-attention, producing exercise memory.
    - The DECODER ingests the RESPONSE sequence SHIFTED RIGHT by one (a start
      token at position 0, then ``r_0 .. r_{t-1}``) + positional, with causal
      self-attention and cross-attention to the encoder memory. Position ``t``
      predicts response ``r_t``.

    No-leakage contract: the exercise at position ``t`` IS available (the
    encoder may attend to exercise ``t`` causally, and the decoder cross-attends
    to it). Only response ``t`` is hidden -- the right-shift means the decoder at
    position ``t`` has seen responses ``< t`` only, and the causal decoder
    self-attention enforces the same. The model emits a logit at EVERY position.
    """

    def __init__(
        self,
        *,
        num_items: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)
        # Encoder stream: exercise (question) embeddings.
        self.exercise_emb = nn.Embedding(self.num_items + 1, d_model, padding_idx=0)
        # Decoder stream: response embeddings. Index 0 is the start/pad token,
        # 1 == incorrect, 2 == correct (response value + 1).
        self.response_emb = nn.Embedding(3, d_model, padding_idx=0)
        self.enc_position_emb = nn.Embedding(self.max_seq_len, d_model)
        self.dec_position_emb = nn.Embedding(self.max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=max(1, int(num_layers)))
        self.out = nn.Linear(d_model, 1)

    def _decoder_input(
        self,
        history_correct: torch.Tensor,
        pad_mask: torch.Tensor,
        positions: torch.Tensor,
        history_elapsed: Optional[torch.Tensor] = None,
        history_lag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Right-shifted response stream (base SAINT: no time features)."""
        del history_elapsed, history_lag
        # Response tokens: 1 + r (0 reserved for start/pad). Shift right by one.
        resp_tokens = (history_correct.long() + 1).masked_fill(pad_mask, 0)
        shifted = torch.zeros_like(resp_tokens)
        shifted[:, 1:] = resp_tokens[:, :-1]  # position t sees response t-1
        return cast(torch.Tensor, self.response_emb(shifted) + self.dec_position_emb(positions))

    def forward(
        self,
        history_items: torch.Tensor,
        history_correct: torch.Tensor,
        query_items: torch.Tensor,
        history_elapsed: Optional[torch.Tensor] = None,
        history_lag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = history_items.shape
        device = history_items.device
        pad_mask = history_items.eq(0)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # nn.Transformer needs a non-fully-masked key row to avoid NaNs; allow
        # all-padding rows to attend to one safe slot (zeroed out by output use).
        key_padding_mask = pad_mask.clone()
        all_padding = key_padding_mask.all(dim=1)
        if bool(all_padding.any()):
            key_padding_mask[all_padding, -1] = False

        # Causal mask (True == disallowed) on/above the diagonal: position t may
        # attend to <= t in the encoder (exercise t is allowed) and decoder.
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )

        # ENCODER over the exercise sequence (query_items == per-position id).
        enc_in = self.exercise_emb(query_items) + self.enc_position_emb(positions)
        enc_in = enc_in.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        memory = self.encoder(
            enc_in, mask=causal, src_key_padding_mask=key_padding_mask
        )

        # DECODER over the right-shifted response stream, cross-attending memory.
        dec_in = self._decoder_input(
            history_correct, pad_mask, positions, history_elapsed, history_lag
        )
        hidden = self.decoder(
            dec_in,
            memory,
            tgt_mask=causal,
            memory_mask=causal,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )
        hidden = torch.nan_to_num(hidden, nan=0.0)
        return cast(torch.Tensor, self.out(hidden).squeeze(-1))


class _SAINTPlusModel(_SAINTModel):
    """Full-sequence SAINT+ tracer (Shin et al. 2021).

    SAINT plus elapsed-time and lag-time embeddings ADDED TO THE DECODER input
    (the response stream), per the paper. The encoder (exercise stream) is
    unchanged from SAINT.
    """

    def __init__(
        self,
        *,
        num_items: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        num_layers: int = 1,
        num_time_buckets: int = 128,
    ) -> None:
        super().__init__(
            num_items=num_items,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.num_time_buckets = int(num_time_buckets)
        self.elapsed_emb = nn.Embedding(self.num_time_buckets + 1, d_model, padding_idx=0)
        self.lag_emb = nn.Embedding(self.num_time_buckets + 1, d_model, padding_idx=0)

    def _decoder_input(
        self,
        history_correct: torch.Tensor,
        pad_mask: torch.Tensor,
        positions: torch.Tensor,
        history_elapsed: Optional[torch.Tensor] = None,
        history_lag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dec_in = super()._decoder_input(history_correct, pad_mask, positions)
        # Time features are aligned to the response at position t (elapsed[t] is
        # the time taken to answer item t; lag[t] the gap before item t). These
        # describe the CURRENT step's timing, which the paper feeds to the
        # decoder; they do not reveal the response value at t.
        if history_elapsed is None:
            history_elapsed = torch.zeros_like(history_correct)
        if history_lag is None:
            history_lag = torch.zeros_like(history_correct)
        return cast(
            torch.Tensor,
            dec_in + self.elapsed_emb(history_elapsed.long()) + self.lag_emb(history_lag.long()),
        )


class _DKTModel(nn.Module):
    """Full-sequence recurrent knowledge tracer (DKT).

    Faithful to Piech et al. (2015): a recurrent network over the interaction
    sequence that emits a prediction at EVERY position in one pass. The original
    paper uses an LSTM (the framework previously used a GRU); this implementation
    uses :class:`torch.nn.LSTM` for paper fidelity.

    No-leakage contract: the recurrent state read out at position ``t`` is
    produced from interactions at positions ``< t`` only. We achieve this by
    feeding the LSTM the interaction sequence shifted right by one step (position
    ``t`` consumes interaction ``t-1``; position 0 consumes a zero vector), then
    reading the per-position hidden state against the query item embedding at
    ``t``. The response at ``t`` therefore never influences the prediction at
    ``t``.
    """

    def __init__(
        self,
        *,
        num_items: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.interaction_emb = nn.Embedding(2 * self.num_items + 1, d_model, padding_idx=0)
        self.query_item_emb = nn.Embedding(self.num_items + 1, d_model, padding_idx=0)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        history_items: torch.Tensor,
        history_correct: torch.Tensor,
        query_items: torch.Tensor,
    ) -> torch.Tensor:
        # All tensors (B, L), right-padded with 0.
        pad_mask = history_items.eq(0)
        interaction_codes = history_items + history_correct.long() * self.num_items
        interaction_codes = interaction_codes.masked_fill(pad_mask, 0)
        x = self.interaction_emb(interaction_codes)  # (B, L, D)

        # Shift right by one: input at position t is interaction t-1, so the
        # hidden state read out at t never depends on the response at t.
        shifted = torch.zeros_like(x)
        shifted[:, 1:, :] = x[:, :-1, :]

        outputs, _ = self.lstm(shifted)  # (B, L, D): per-position hidden states.
        query = self.query_item_emb(query_items)  # (B, L, D)
        return cast(torch.Tensor, self.out(torch.cat([outputs, query], dim=-1)).squeeze(-1))


class _DKVMNModel(nn.Module):
    """Full-sequence DKVMN key-value memory tracer (Zhang et al. 2017).

    Real key-value memory:

    - ``M_k``: a STATIC learned key matrix of shape ``(N_slots, d_k)``.
    - ``M_v0``: a learned INITIAL value matrix of shape ``(N_slots, d_v)``,
      broadcast per learner and mutated over time.

    Sequential over ``t`` (carrying the working value memory ``M_v``):

    1. correlation weight ``w_t = softmax(exercise_key(item_t) @ M_k^T)``,
    2. READ ``r_t = w_t @ M_v`` from the CURRENT value memory,
    3. predict ``logit_t`` from ``[r_t, exercise_key(item_t)]``,
    4. WRITE interaction ``t``: ``e_t = sigmoid(W_e v_t)``,
       ``a_t = tanh(W_a v_t)``, then
       ``M_v <- M_v * (1 - w_t^T e_t) + w_t^T a_t``.

    No-leakage contract: the prediction at ``t`` uses the read taken BEFORE the
    interaction-``t`` write, so it depends on interactions ``< t`` and the
    exercise id at ``t`` only -- never response ``t``.
    """

    def __init__(
        self,
        *,
        num_items: int,
        d_model: int,
        dropout: float,
        memory_size: int = 20,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.memory_size = int(memory_size)
        self.d_k = int(d_model)
        self.d_v = int(d_model)

        # Embeddings feeding the key (addressing) and value (write) computations.
        self.exercise_key_emb = nn.Embedding(self.num_items + 1, self.d_k, padding_idx=0)
        self.interaction_value_emb = nn.Embedding(2 * self.num_items + 1, self.d_v, padding_idx=0)

        # Static key matrix and learned initial value matrix.
        self.key_matrix = nn.Parameter(torch.randn(self.memory_size, self.d_k) * 0.1)  # M_k
        self.value_matrix_init = nn.Parameter(torch.randn(self.memory_size, self.d_v) * 0.1)  # M_v0

        self.erase = nn.Linear(self.d_v, self.d_v)
        self.add = nn.Linear(self.d_v, self.d_v)
        self.summary = nn.Linear(self.d_v + self.d_k, d_model)
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        history_items: torch.Tensor,
        history_correct: torch.Tensor,
        query_items: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = history_items.shape
        pad_mask = history_items.eq(0)
        interaction_codes = history_items + history_correct.long() * self.num_items
        interaction_codes = interaction_codes.masked_fill(pad_mask, 0)

        # Per-position embeddings.
        q_keys = self.exercise_key_emb(query_items)  # (B, L, d_k): addressing/query key
        write_values = self.interaction_value_emb(interaction_codes)  # (B, L, d_v)

        # Working value memory M_v, broadcast per learner from the learned init.
        value_matrix = self.value_matrix_init.unsqueeze(0).expand(bsz, -1, -1).contiguous()  # (B, N, d_v)

        logits = []
        for t in range(seq_len):
            k_t = q_keys[:, t, :]  # (B, d_k)
            # Correlation weight over memory slots.
            w_t = torch.softmax(k_t @ self.key_matrix.t(), dim=-1)  # (B, N)

            # READ before writing interaction t -> causal, no leakage.
            r_t = torch.bmm(w_t.unsqueeze(1), value_matrix).squeeze(1)  # (B, d_v)
            summary = torch.tanh(self.summary(torch.cat([r_t, k_t], dim=-1)))
            logits.append(self.out(summary).squeeze(-1))  # (B,)

            # WRITE interaction t into the value memory.
            v_t = write_values[:, t, :]  # (B, d_v)
            e_t = torch.sigmoid(self.erase(v_t))  # (B, d_v)
            a_t = torch.tanh(self.add(v_t))  # (B, d_v)
            w_col = w_t.unsqueeze(-1)  # (B, N, 1)
            erase_term = w_col * e_t.unsqueeze(1)  # (B, N, d_v)
            add_term = w_col * a_t.unsqueeze(1)  # (B, N, d_v)
            new_value = value_matrix * (1.0 - erase_term) + add_term
            # Padded steps must not mutate the memory (keeps cold-start clean).
            step_pad = pad_mask[:, t].view(bsz, 1, 1)
            value_matrix = torch.where(step_pad, value_matrix, new_value)

        return torch.stack(logits, dim=1)  # (B, L)


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
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= correct_threshold <= 1.0:
            raise ValueError("correct_threshold must be in [0, 1]")

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

        # All tracers (SAKT, DKT, AKT, SAINT, SAINT+, DKVMN) are full-sequence:
        # one forward pass emits a causal, no-leakage prediction at every
        # position. The flag is retained so the masked-BCE training path in
        # ``fit`` stays explicit.
        self._full_sequence: bool = True

        self.model: Optional[nn.Module] = None
        self.training_examples_: List[SAKTTrainingExample] = []
        self.result_: Dict[str, float] = {}
        self._item2idx: Dict[Any, int] = {}
        self._idx2item: Dict[int, Any] = {}
        self._histories: Dict[Any, List[Tuple[int, int]]] = {}
        self._history_times: Dict[Any, List[float]] = {}
        # Records whether ``fit`` was given a ``timestamp_col``. Used by
        # timestamp-aware subclasses (e.g. :class:`SAINTPlusTracer`) to warn when
        # ``observe`` is later called without a timestamp at serving time.
        self._fit_used_timestamps: bool = False

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

        self._fit_used_timestamps = timestamp_col is not None

        examples = build_sakt_examples(
            interactions,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
            max_seq_len=self.max_seq_len,
            correct_threshold=self.correct_threshold,
        )
        if not examples or all(len(ex) < 2 for ex in examples):
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

        encoded = self._encode_examples(examples)
        dataset = torch.utils.data.TensorDataset(*encoded)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._make_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Full-sequence loss is averaged over NON-PADDED positions only (masked
        # BCE). Single-query loss is a plain per-example BCE.
        loss_fn = nn.BCEWithLogitsLoss(reduction="none" if self._full_sequence else "mean")

        self.model.train()
        last_loss = 0.0
        for _ in range(self.epochs):
            for raw_batch in loader:
                batch = tuple(tensor.to(self.device) for tensor in raw_batch)

                optimizer.zero_grad()
                if self._full_sequence:
                    # Encoded layout: (*model_inputs, labels, mask).
                    mask = batch[-1].float()
                    batch_labels = batch[-2]
                    logits = self._logits_from_batch(batch[:-2])
                    per_pos = loss_fn(logits, batch_labels)
                    denom = mask.sum().clamp_min(1.0)
                    loss = (per_pos * mask).sum() / denom
                else:
                    batch_labels = batch[-1]
                    logits = self._logits_from_batch(batch[:-1])
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

    def _logits_from_batch(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        assert self.model is not None
        history_items, history_correct, query_items = batch
        return cast(torch.Tensor, self.model(history_items, history_correct, query_items))

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
        times = self._history_times.setdefault(user_id, [])
        next_time = times[-1] + 1.0 if times else 0.0
        times.append(next_time)
        if len(times) > self.max_seq_len:
            del times[: len(times) - self.max_seq_len]
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
        time_histories: Dict[Any, List[float]] = {}
        for user_id, group in work.groupby(user_col, sort=False):
            rows: List[Tuple[int, int]] = []
            times: List[float] = []
            cols = [item_col, correct_col] if timestamp_col is None else [item_col, correct_col, timestamp_col]
            for row in group[cols].itertuples(index=False, name=None):
                item_id = row[0]
                raw_correct = row[1]
                rows.append(
                    (
                        self._internal_item_id(item_id),
                        _label(raw_correct, self.correct_threshold),
                    )
                )
                if timestamp_col is not None:
                    times.append(_time_value(row[2]))
            histories[user_id] = rows[-self.max_seq_len:]
            if timestamp_col is not None:
                time_histories[user_id] = times[-self.max_seq_len:]
        self._history_times = time_histories
        return histories

    def _encode_examples(
        self,
        examples: Sequence[SAKTTrainingExample],
    ) -> Tuple[torch.Tensor, ...]:
        return self._encode_full_sequence(examples)

    def _encode_full_sequence(
        self,
        examples: Sequence[SAKTTrainingExample],
    ) -> Tuple[torch.Tensor, ...]:
        """Encode full-sequence windows into right-padded per-position tensors.

        Returns ``(history_items, history_correct, query_items, labels, mask)``,
        all of shape ``(N, max_seq_len)``. ``query_items`` equals
        ``history_items`` (each position queries its own exercise id); ``labels``
        are the per-position responses; ``mask`` is 1 on real positions, 0 on
        right-padding. RIGHT-padding is used consistently with serving.
        """
        history_items = np.zeros((len(examples), self.max_seq_len), dtype=np.int64)
        history_correct = np.zeros((len(examples), self.max_seq_len), dtype=np.int64)
        labels = np.zeros((len(examples), self.max_seq_len), dtype=np.float32)
        mask = np.zeros((len(examples), self.max_seq_len), dtype=np.float32)

        for row_idx, example in enumerate(examples):
            encoded_items = [self._internal_item_id(item_id) for item_id in example.item_ids]
            length = len(encoded_items)
            history_items[row_idx, :length] = encoded_items
            history_correct[row_idx, :length] = list(example.correct)
            labels[row_idx, :length] = [float(c) for c in example.correct]
            mask[row_idx, :length] = 1.0

        return (
            torch.as_tensor(history_items, dtype=torch.long),
            torch.as_tensor(history_correct, dtype=torch.long),
            torch.as_tensor(history_items, dtype=torch.long),  # query == own item id
            torch.as_tensor(labels, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.float32),
        )

    def _predict_internal(self, user_id: Any, internal_items: Sequence[int]) -> np.ndarray:
        self._require_fitted()
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            probs = self._predict_full_sequence(user_id, internal_items)
        return probs.astype(np.float32)

    def _predict_full_sequence(self, user_id: Any, internal_items: Sequence[int]) -> np.ndarray:
        """Serve next-item predictions from the full-sequence (causal) model.

        The learner's observed interactions occupy positions ``0..N-1`` (right-
        padded). Each candidate is placed at the query position ``N`` with NO
        response, and we read the per-position logit at that query slot. Causal
        masking guarantees position ``N`` attends only to the real history
        (positions ``< N``), matching the old serving contract: "given observed
        history, predict the next item".
        """
        assert self.model is not None
        # Reserve the last slot for the query, so keep at most max_seq_len-1 of
        # the observed history (mirrors training windows ending at the query).
        budget = max(0, self.max_seq_len - 1)
        history = self._histories.get(user_id, [])[-budget:] if budget else []
        n = len(history)

        count = len(internal_items)
        items = np.zeros((count, self.max_seq_len), dtype=np.int64)
        correct = np.zeros((count, self.max_seq_len), dtype=np.int64)
        if history:
            items[:, :n] = [item for item, _ in history]
            correct[:, :n] = [label for _, label in history]
        # Place each candidate at the query position N (response stays 0/unseen).
        items[:, n] = list(internal_items)

        history_items = torch.as_tensor(items, dtype=torch.long, device=self.device)
        history_correct = torch.as_tensor(correct, dtype=torch.long, device=self.device)
        query_items = history_items  # full-sequence: query id == item id at each position
        logits = self._predict_logits(user_id, history_items, history_correct, query_items)
        # logits: (count, max_seq_len) -> take the query position N.
        query_logits = logits[:, n]
        return torch.sigmoid(query_logits).detach().cpu().numpy().astype(np.float32)

    def _predict_logits(
        self,
        user_id: Any,
        history_items: torch.Tensor,
        history_correct: torch.Tensor,
        query_items: torch.Tensor,
    ) -> torch.Tensor:
        del user_id
        assert self.model is not None
        return cast(torch.Tensor, self.model(history_items, history_correct, query_items))


class SAINTTracer(SAKTTracer):
    """SAINT-style encoder-decoder tracer for next-response prediction.

    The implementation keeps Orchid's tracer API while moving from the compact
    single-query SAKT block to a small transformer encoder-decoder. It is
    intentionally lightweight so it can run in CI and serve as a benchmarkable
    in-repo backbone before larger model-zoo integrations.
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
        num_layers: int = 1,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        super().__init__(
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            correct_threshold=correct_threshold,
            device=device,
            random_state=random_state,
        )
        self.num_layers = int(num_layers)
        self._full_sequence = True

    def _make_model(self) -> nn.Module:
        return _SAINTModel(
            num_items=len(self._item2idx),
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
            num_layers=self.num_layers,
        )


class SAINTPlusTracer(SAINTTracer):
    """SAINT-style tracer with elapsed-time and lag-time history features.

    When ``timestamp_col`` is supplied, training examples include two temporal
    signals inspired by SAINT+: elapsed time between historical attempts and lag
    time from each historical attempt to the queried attempt. Without timestamps
    the tracer still fits, but temporal embeddings are zero and it behaves like
    :class:`SAINTTracer`.
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
        num_layers: int = 1,
        num_time_buckets: int = 128,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if num_time_buckets < 1:
            raise ValueError("num_time_buckets must be >= 1")
        super().__init__(
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            correct_threshold=correct_threshold,
            num_layers=num_layers,
            device=device,
            random_state=random_state,
        )
        self.num_time_buckets = int(num_time_buckets)
        # Emit the missing-timestamp serving warning at most once per instance.
        self._warned_missing_observe_timestamp: bool = False

    def observe(self, user_id: Any, item_id: Any, correct: Any, timestamp: Optional[Any] = None) -> int:
        """Append one live outcome and preserve timestamp features when present.

        Parameters
        ----------
        timestamp : optional
            Wall-clock time of this interaction. When omitted, the base tracer
            synthesizes a placeholder gap of ``+1.0`` time unit from the previous
            interaction.

        Notes
        -----
        If the tracer was fit **with** a ``timestamp_col`` but ``observe`` is
        called **without** a ``timestamp``, the synthesized placeholder gap will
        not match the temporal distribution seen during training, so the live
        elapsed-time and lag-time features (and therefore predictions) may be
        inaccurate. A :class:`UserWarning` is emitted once per tracer instance in
        that case. The numeric fallback is unchanged; pass ``timestamp`` to
        supply a real value and silence the warning.
        """
        if (
            timestamp is None
            and self._fit_used_timestamps
            and not self._warned_missing_observe_timestamp
        ):
            warnings.warn(
                "SAINTPlusTracer was fit with a timestamp_col, but observe() was "
                "called without a timestamp. A placeholder time gap is being used, "
                "so the live elapsed/lag temporal features may not match training "
                "and predictions may be inaccurate. Pass timestamp=... to observe() "
                "to provide a real value.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_missing_observe_timestamp = True
        length = super().observe(user_id, item_id, correct)
        if timestamp is not None:
            self._history_times.setdefault(user_id, [])[-1] = _time_value(timestamp)
        return length

    def _make_model(self) -> nn.Module:
        return _SAINTPlusModel(
            num_items=len(self._item2idx),
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
            num_layers=self.num_layers,
            num_time_buckets=self.num_time_buckets,
        )

    def _encode_examples(
        self,
        examples: Sequence[SAKTTrainingExample],
    ) -> Tuple[torch.Tensor, ...]:
        """Full-sequence encoding plus per-position elapsed/lag time buckets.

        Returns ``(history_items, history_correct, query_items, elapsed, lag,
        labels, mask)``. The base full-sequence encoder supplies the first three
        plus labels/mask; we splice the two time-feature tensors in BEFORE the
        ``labels, mask`` tail so the training loop's ``batch[:-2]`` model inputs
        carry the time features.
        """
        history_items, history_correct, query_items, labels, mask = self._encode_full_sequence(examples)
        elapsed = np.zeros((len(examples), self.max_seq_len), dtype=np.int64)
        lag = np.zeros((len(examples), self.max_seq_len), dtype=np.int64)

        for row_idx, example in enumerate(examples):
            length = len(example.item_ids)
            if example.elapsed:
                elapsed_values = _bucket_time_values(example.elapsed[: self.max_seq_len], self.num_time_buckets)
                elapsed[row_idx, : elapsed_values.size] = elapsed_values
            if example.lag:
                lag_values = _bucket_time_values(example.lag[: self.max_seq_len], self.num_time_buckets)
                lag[row_idx, : lag_values.size] = lag_values
            del length

        return (
            history_items,
            history_correct,
            query_items,
            torch.as_tensor(elapsed, dtype=torch.long),
            torch.as_tensor(lag, dtype=torch.long),
            labels,
            mask,
        )

    def _logits_from_batch(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        assert self.model is not None
        history_items, history_correct, query_items, history_elapsed, history_lag = batch
        return cast(torch.Tensor, self.model(history_items, history_correct, query_items, history_elapsed, history_lag))

    def _predict_full_sequence(self, user_id: Any, internal_items: Sequence[int]) -> np.ndarray:
        """Serve next-item predictions with the query placed at slot ``N``.

        Mirrors the base full-sequence serving but threads the learner's elapsed
        and lag time features (aligned to the same right-padded layout, with the
        query slot's elapsed = 0 and lag = 0 since the next attempt has no
        recorded timing yet).
        """
        assert self.model is not None
        budget = max(0, self.max_seq_len - 1)
        history = self._histories.get(user_id, [])[-budget:] if budget else []
        times = self._history_times.get(user_id, [])[-budget:] if budget else []
        n = len(history)

        count = len(internal_items)
        items = np.zeros((count, self.max_seq_len), dtype=np.int64)
        correct = np.zeros((count, self.max_seq_len), dtype=np.int64)
        elapsed = np.zeros((count, self.max_seq_len), dtype=np.int64)
        lag = np.zeros((count, self.max_seq_len), dtype=np.int64)
        if history:
            items[:, :n] = [item for item, _ in history]
            correct[:, :n] = [label for _, label in history]
        items[:, n] = list(internal_items)
        if times and len(times) == n and n > 0:
            elapsed_values = _bucket_time_values(_elapsed_history(times), self.num_time_buckets)
            query_time = float(times[-1])
            lag_values = _bucket_time_values([max(0.0, query_time - t) for t in times], self.num_time_buckets)
            elapsed[:, :n] = elapsed_values[:n]
            lag[:, :n] = lag_values[:n]

        history_items = torch.as_tensor(items, dtype=torch.long, device=self.device)
        history_correct = torch.as_tensor(correct, dtype=torch.long, device=self.device)
        history_elapsed = torch.as_tensor(elapsed, dtype=torch.long, device=self.device)
        history_lag = torch.as_tensor(lag, dtype=torch.long, device=self.device)
        query_items = history_items
        logits = cast(
            torch.Tensor,
            self.model(history_items, history_correct, query_items, history_elapsed, history_lag),
        )
        query_logits = logits[:, n]
        return torch.sigmoid(query_logits).detach().cpu().numpy().astype(np.float32)


class DKTTracer(SAKTTracer):
    """DKT-style recurrent tracer for next-response prediction.

    DKT is still a useful adaptive-learning baseline because it tests whether a
    simple recurrent learner-state model is enough before adding attention,
    item difficulty, or temporal transformer features.
    """

    def _make_model(self) -> nn.Module:
        return _DKTModel(
            num_items=len(self._item2idx),
            d_model=self.d_model,
            dropout=self.dropout,
        )


class DKVMNTracer(SAKTTracer):
    """Compact DKVMN-style memory-network tracer.

    This is a curated, dependency-light baseline for concept/item memory
    tracing. It is not a line-by-line reproduction of every DKVMN paper detail;
    benchmark docs should describe it as "DKVMN-style".
    """

    def __init__(
        self,
        *,
        max_seq_len: int = 50,
        d_model: int = 64,
        n_heads: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 128,
        correct_threshold: float = 0.5,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        del n_heads  # Memory attention is single-head in this compact baseline.
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
        self._full_sequence = True

    def _make_model(self) -> nn.Module:
        return _DKVMNModel(
            num_items=len(self._item2idx),
            d_model=self.d_model,
            dropout=self.dropout,
        )


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
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            correct_threshold=correct_threshold,
            device=device,
            random_state=random_state,
        )
        self._full_sequence = True
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
        # The difficulty tensor (per-item prior in [0, 1]) WARM-STARTS the learned
        # Rasch difficulty embedding ``mu_q``; it then trains freely.
        return _AKTModel(
            num_items=len(self._item2idx),
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout,
            item_difficulty=self._item_difficulty_tensor,
        )
