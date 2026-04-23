"""Scaling primitives for large deployments (1M--100M+ users).

The default :class:`StreamingAdaptiveRanker` stores per-user state in a
dense embedding table and a single-lock BKT dictionary. That design is
simple and fast for up to ~1M users. Beyond that, memory and lock
contention become bottlenecks.

This module provides drop-in replacements:

* :class:`SparseEmbeddingTable` -- hash-map embeddings with LRU eviction.
* :class:`SparseOnlineUserAdapter` -- residual adapter using sparse storage.
* :class:`ShardedBKTStateProvider` -- N-shard BKT state with per-shard locks.
* :class:`ScalingConfig` -- configuration dataclass.

Usage::

    from orchid_ranker.scaling import SparseOnlineUserAdapter, ShardedBKTStateProvider

    adapter = SparseOnlineUserAdapter(emb_dim=32, max_active_users=5_000_000)
    state = ShardedBKTStateProvider(num_shards=16)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from orchid_ranker.knowledge_tracing import BayesianKnowledgeTracing

__all__ = [
    "SparseEmbeddingTable",
    "SparseOnlineUserAdapter",
    "ShardedBKTStateProvider",
    "ScalingConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ScalingConfig:
    """Configuration for large-scale deployment.

    Parameters
    ----------
    max_active_users : int
        Maximum number of active user embeddings kept in memory by
        :class:`SparseEmbeddingTable`.  When the table exceeds this
        limit the least-recently-used entries are evicted.
    num_state_shards : int
        Number of independent BKT state shards used by
        :class:`ShardedBKTStateProvider`.  Each shard has its own lock,
        so concurrent ``observe()`` calls for users in different shards
        do not contend.
    eviction_policy : str
        ``"lru"`` for least-recently-used eviction (default) or ``"ttl"``
        for time-to-live based expiration.
    ttl_seconds : float
        When *eviction_policy* is ``"ttl"``, entries older than this many
        seconds since last access are eligible for eviction.  Ignored
        under the ``"lru"`` policy.
    enable_metrics : bool
        If ``True``, the scaling components emit additional log messages
        at ``DEBUG`` level for observability dashboards.
    """

    max_active_users: int = 1_000_000
    num_state_shards: int = 16
    eviction_policy: str = "lru"  # "lru" or "ttl"
    ttl_seconds: float = 86400.0  # 24 h default for TTL policy
    enable_metrics: bool = True


# ---------------------------------------------------------------------------
# Sparse embedding table
# ---------------------------------------------------------------------------
class SparseEmbeddingTable:
    """Hash-map backed embedding table with LRU eviction.

    Unlike :class:`torch.nn.Embedding` which allocates a dense
    ``(num_users, emb_dim)`` matrix, this only stores embeddings for
    users who have been observed.  Inactive users are evicted when the
    table exceeds *max_entries*, freeing memory.

    Evicted users transparently get zero embeddings on next access (same
    as a new user), so the system degrades gracefully under memory
    pressure.

    Parameters
    ----------
    emb_dim : int
        Dimensionality of each embedding vector.
    max_entries : int
        Maximum number of user embeddings kept in the table.  When
        exceeded, the least-recently-used entry is evicted.
    device : str
        Torch device for allocated tensors (default ``"cpu"``).

    Notes
    -----
    Thread-safety is ensured via a :class:`threading.RLock`.  All public
    methods acquire the lock before mutating internal state.

    The underlying storage is a :class:`collections.OrderedDict` which
    provides O(1) ``move_to_end`` (on access) and O(1) ``popitem``
    (for eviction).
    """

    def __init__(
        self,
        emb_dim: int,
        *,
        max_entries: int = 1_000_000,
        device: str = "cpu",
    ) -> None:
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}")
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")

        self._emb_dim = int(emb_dim)
        self._max_entries = int(max_entries)
        self._device = str(device)
        # OrderedDict preserves insertion order; we move keys to the end on
        # access so that the *first* item is always the least-recently-used.
        self._data: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._lock = threading.RLock()

        logger.debug(
            "SparseEmbeddingTable created: emb_dim=%d, max_entries=%s, device=%s",
            self._emb_dim,
            f"{self._max_entries:,}",
            self._device,
        )

    # ---- mapping interface ----

    def __getitem__(self, user_id: int) -> torch.Tensor:
        """Get embedding for *user_id*.

        Returns a zero tensor if the user has not yet been observed
        (i.e. never assigned via ``__setitem__``).  Accessing an
        existing entry marks it as most-recently-used.

        Parameters
        ----------
        user_id : int
            User identifier.

        Returns
        -------
        torch.Tensor
            Shape ``(emb_dim,)`` tensor on ``self._device``.
        """
        uid = int(user_id)
        with self._lock:
            if uid in self._data:
                self._data.move_to_end(uid)
                return self._data[uid]
        # Not stored -- return a fresh zero tensor (do NOT allocate a slot).
        return torch.zeros(self._emb_dim, device=self._device)

    def __setitem__(self, user_id: int, value: torch.Tensor) -> None:
        """Set embedding for *user_id*.

        If the table is at capacity and *user_id* is a new key, the
        least-recently-used entry is evicted first.

        Parameters
        ----------
        user_id : int
            User identifier.
        value : torch.Tensor
            Embedding tensor.  Must have shape ``(emb_dim,)``.
        """
        uid = int(user_id)
        with self._lock:
            if uid in self._data:
                # Update existing -- move to end (most recent).
                self._data[uid] = value.detach().to(self._device)
                self._data.move_to_end(uid)
            else:
                # New entry -- evict if at capacity.
                if len(self._data) >= self._max_entries:
                    self._evict_lru()
                self._data[uid] = value.detach().to(self._device)

    def __contains__(self, user_id: int) -> bool:  # type: ignore[override]
        """Return ``True`` if *user_id* has a stored embedding."""
        with self._lock:
            return int(user_id) in self._data

    def __len__(self) -> int:
        """Number of currently stored embeddings."""
        with self._lock:
            return len(self._data)

    # ---- eviction ----

    def evict(self, user_id: int) -> None:
        """Manually evict a user's embedding.

        No-op if the user is not in the table.

        Parameters
        ----------
        user_id : int
            User identifier to remove.
        """
        uid = int(user_id)
        with self._lock:
            if uid in self._data:
                del self._data[uid]
                logger.debug("Manually evicted user %d from SparseEmbeddingTable", uid)

    def _evict_lru(self) -> None:
        """Evict the single least-recently-used entry.

        Must be called while holding ``self._lock``.
        """
        if not self._data:
            return
        evicted_uid, _ = self._data.popitem(last=False)
        logger.warning(
            "SparseEmbeddingTable at capacity (%s): evicted user %d",
            f"{self._max_entries:,}",
            evicted_uid,
        )

    # ---- diagnostics ----

    def memory_bytes(self) -> int:
        """Estimate current memory usage of stored embeddings.

        Returns
        -------
        int
            Approximate bytes consumed by embedding tensors.  Does not
            account for Python object overhead or the OrderedDict
            bookkeeping.
        """
        with self._lock:
            n = len(self._data)
        # Each entry is a float32 tensor of shape (emb_dim,).
        bytes_per_entry = self._emb_dim * 4  # float32 = 4 bytes
        return n * bytes_per_entry

    @property
    def capacity(self) -> int:
        """Maximum number of entries the table can hold before eviction."""
        return self._max_entries

    @property
    def occupancy(self) -> float:
        """Fraction of capacity currently in use, in ``[0.0, 1.0]``."""
        with self._lock:
            return len(self._data) / max(self._max_entries, 1)

    def __repr__(self) -> str:
        return (
            f"SparseEmbeddingTable(emb_dim={self._emb_dim}, "
            f"entries={len(self)}/{self._max_entries}, "
            f"occupancy={self.occupancy:.1%})"
        )


# ---------------------------------------------------------------------------
# Sparse online user adapter
# ---------------------------------------------------------------------------
class SparseOnlineUserAdapter(nn.Module):
    """Per-user residual adapter with sparse storage and LRU eviction.

    API-compatible with :class:`~orchid_ranker.streaming.OnlineUserAdapter`
    but scales to 100M+ users by only storing embeddings for active users.
    There is no ``num_users`` parameter -- the user-id space is unbounded.

    The residual for a user starts at zero.  After each ``observe()`` call
    the residual is updated via one step of logistic SGD (identical to the
    dense variant).  Users whose embeddings are evicted due to capacity
    pressure silently revert to zero residuals on next access, matching the
    cold-start behaviour of a new user.

    Parameters
    ----------
    emb_dim : int
        Must match the two-tower's projection dimension.
    lr : float
        SGD learning rate for the residual update.
    l2 : float
        Shrinkage coefficient applied to the residual in the gradient.
    clip : float
        Per-update L2 norm clip on the residual vector.
    max_active_users : int
        Maximum active-user embeddings kept in memory.
    device : str
        Torch device for tensor operations.
    """

    def __init__(
        self,
        emb_dim: int,
        *,
        lr: float = 0.05,
        l2: float = 1e-3,
        clip: float = 1.0,
        max_active_users: int = 1_000_000,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if emb_dim <= 0:
            raise ValueError("emb_dim must be positive")

        self.emb_dim = int(emb_dim)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.clip = float(clip)
        self._device = str(device)

        self._embeddings = SparseEmbeddingTable(
            emb_dim=self.emb_dim,
            max_entries=max_active_users,
            device=self._device,
        )
        self._lock = threading.RLock()
        self._update_count: Dict[int, int] = defaultdict(int)

        logger.info(
            "SparseOnlineUserAdapter created: emb_dim=%d, lr=%.4f, l2=%.1e, "
            "clip=%.2f, max_active_users=%s",
            self.emb_dim,
            self.lr,
            self.l2,
            self.clip,
            f"{max_active_users:,}",
        )

    @torch.no_grad()
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Look up residual embeddings for a batch of user ids.

        Parameters
        ----------
        user_ids : torch.Tensor
            1-D integer tensor of user identifiers.

        Returns
        -------
        torch.Tensor
            Shape ``(len(user_ids), emb_dim)`` residual embeddings.
        """
        ids = user_ids.long().view(-1)
        out = torch.stack([self._embeddings[int(uid)] for uid in ids])
        return out.to(self._device)

    @torch.no_grad()
    def observe(
        self,
        user_id: int,
        u_base: torch.Tensor,
        item_emb: torch.Tensor,
        y: float,
    ) -> float:
        """Apply one SGD step and return the squared-norm of the updated residual.

        The update rule is identical to
        :meth:`~orchid_ranker.streaming.OnlineUserAdapter.observe`::

            r = embeddings[uid]          # zeros if new / evicted
            u_full = u_base + r
            logit  = dot(u_full, item_emb)
            p      = sigmoid(logit)
            grad   = (p - y) * item_emb + l2 * r
            r_new  = r - lr * grad
            if norm(r_new) > clip:
                r_new *= clip / norm(r_new)
            embeddings[uid] = r_new

        Parameters
        ----------
        user_id : int
            User identifier (any non-negative integer).
        u_base : torch.Tensor
            Pre-residual user embedding from the frozen tower, shape ``(emb_dim,)``.
        item_emb : torch.Tensor
            Item embedding from the frozen tower, shape ``(emb_dim,)``.
        y : float
            Observed label in ``[0, 1]`` (e.g. correct / accept).

        Returns
        -------
        float
            Squared L2 norm of the updated residual.
        """
        if u_base.shape[-1] != self.emb_dim or item_emb.shape[-1] != self.emb_dim:
            raise ValueError("u_base and item_emb must have shape [emb_dim]")

        uid = int(user_id)
        with self._lock:
            r = self._embeddings[uid]
            u_full = u_base.detach().to(r.dtype).to(r.device) + r
            i = item_emb.detach().to(r.dtype).to(r.device)

            logit = torch.dot(u_full, i)
            p = torch.sigmoid(logit)
            grad = (p - float(y)) * i + self.l2 * r
            r_new = r - self.lr * grad

            # Norm clip on the residual itself.
            n = torch.linalg.vector_norm(r_new)
            if float(n) > self.clip:
                r_new = r_new * (self.clip / float(n))

            self._embeddings[uid] = r_new
            self._update_count[uid] += 1
            return float(torch.dot(r_new, r_new))

    def updates_for(self, user_id: int) -> int:
        """Return the number of SGD updates applied for *user_id*."""
        return int(self._update_count.get(int(user_id), 0))

    def reset_user(self, user_id: int) -> None:
        """Reset a user's residual to zero and clear update count."""
        uid = int(user_id)
        with self._lock:
            self._embeddings.evict(uid)
            self._update_count.pop(uid, None)
        logger.debug("Reset user %d in SparseOnlineUserAdapter", uid)

    @property
    def active_users(self) -> int:
        """Number of users with a stored (non-zero) residual."""
        return len(self._embeddings)

    @property
    def memory_bytes(self) -> int:
        """Approximate memory consumed by stored residual embeddings."""
        return self._embeddings.memory_bytes()

    def __repr__(self) -> str:
        return (
            f"SparseOnlineUserAdapter(emb_dim={self.emb_dim}, "
            f"active_users={self.active_users}, "
            f"memory={self.memory_bytes / 1024:.1f}KB)"
        )


# ---------------------------------------------------------------------------
# Per-user telemetry (mirrors streaming._UserTelemetry)
# ---------------------------------------------------------------------------
@dataclass
class _UserTelemetry:
    """Per-user rolling stats used to derive [fatigue, trust, engagement].

    Mirrors the identically-named dataclass in :mod:`orchid_ranker.streaming`
    so that each shard is fully self-contained and does not import mutable
    state from the streaming module.
    """

    events: Deque[Tuple[float, bool]] = field(
        default_factory=lambda: deque(maxlen=64)
    )
    last_seen: Optional[float] = None

    def record(self, correct: bool, ts: float) -> None:
        self.events.append((float(ts), bool(correct)))
        self.last_seen = float(ts)

    def trust(self) -> float:
        """Proxy for trust: recent correctness rate in [0, 1]."""
        if not self.events:
            return 0.5
        return float(sum(c for _, c in self.events) / len(self.events))

    def engagement(self, now: float, window: float = 300.0) -> float:
        """Proxy for engagement: interactions-per-minute, squashed."""
        if not self.events:
            return 0.0
        recent = [t for t, _ in self.events if (now - t) <= window]
        if not recent:
            return 0.0
        rate_per_min = 60.0 * len(recent) / max(window, 1e-6)
        return float(math.tanh(rate_per_min / 4.0))

    def fatigue(self, now: float, window: float = 900.0) -> float:
        """Proxy for fatigue: saturating function of recent event count."""
        if not self.events:
            return 0.0
        recent = [t for t, _ in self.events if (now - t) <= window]
        return float(1.0 - math.exp(-len(recent) / 15.0))


# ---------------------------------------------------------------------------
# Sharded BKT state provider
# ---------------------------------------------------------------------------
class _BKTShard:
    """A single shard of BKT state, with its own lock and data structures.

    This is an internal building block of :class:`ShardedBKTStateProvider`.
    Each shard independently manages a subset of users (those whose
    ``user_id % num_shards == shard_index``).

    Parameters
    ----------
    shard_index : int
        Index of this shard (for logging).
    bkt_kwargs : dict
        Forwarded to :class:`BayesianKnowledgeTracing` constructors.
    default_category : str
        Key used when an interaction does not specify a category.
    max_users : int
        Maximum number of tracked users before LRU eviction kicks in.
    """

    def __init__(
        self,
        shard_index: int,
        bkt_kwargs: dict,
        default_category: str,
        max_users: int,
    ) -> None:
        self.shard_index = shard_index
        self._bkt_kwargs = dict(bkt_kwargs)
        self.default_category = default_category
        self._max_users = max_users

        # user_id -> { category -> BayesianKnowledgeTracing }
        self._trackers: Dict[int, Dict[str, BayesianKnowledgeTracing]] = {}
        self._telemetry: Dict[int, _UserTelemetry] = {}
        # Track access order for LRU eviction.  Keys are user_ids; the
        # first key is the least recently used.
        self._access_order: OrderedDict[int, None] = OrderedDict()
        self._lock = threading.RLock()

    def _touch(self, uid: int) -> None:
        """Mark *uid* as most recently accessed.  Must hold lock."""
        if uid in self._access_order:
            self._access_order.move_to_end(uid)
        else:
            self._access_order[uid] = None

    def _ensure_capacity(self) -> None:
        """Evict users if the shard exceeds capacity.  Must hold lock."""
        while len(self._trackers) > self._max_users and self._access_order:
            evicted_uid, _ = self._access_order.popitem(last=False)
            self._trackers.pop(evicted_uid, None)
            self._telemetry.pop(evicted_uid, None)
            logger.warning(
                "BKT shard %d at capacity (%s): evicted user %d",
                self.shard_index,
                f"{self._max_users:,}",
                evicted_uid,
            )

    def observe(
        self,
        user_id: int,
        correct: bool,
        *,
        category: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> float:
        """Record an interaction and return updated competence for the category."""
        category = str(category or self.default_category)
        ts = float(time.time() if timestamp is None else timestamp)
        uid = int(user_id)

        with self._lock:
            # Ensure user tracker dict exists.
            if uid not in self._trackers:
                self._trackers[uid] = {}
                self._ensure_capacity()
            bkts = self._trackers[uid]
            if category not in bkts:
                bkts[category] = BayesianKnowledgeTracing(**self._bkt_kwargs)

            p_known = bkts[category].update(bool(correct))

            # Telemetry.
            if uid not in self._telemetry:
                self._telemetry[uid] = _UserTelemetry()
            self._telemetry[uid].record(correct, ts)

            self._touch(uid)
            return float(p_known)

    def state_vector(
        self,
        user_id: int,
        *,
        now: Optional[float] = None,
    ) -> List[float]:
        """Return ``[competence, fatigue, trust, engagement]`` for *user_id*."""
        t_now = float(time.time() if now is None else now)
        uid = int(user_id)

        with self._lock:
            bkts = self._trackers.get(uid, {})
            if bkts:
                comp = float(np.mean([bk.p_known for bk in bkts.values()]))
            else:
                comp = float(self._bkt_kwargs.get("p_init", 0.1))

            tel = self._telemetry.get(uid)
            if tel is None:
                fatigue = 0.0
                trust = 0.5
                engagement = 0.0
            else:
                fatigue = tel.fatigue(t_now)
                trust = tel.trust()
                engagement = tel.engagement(t_now)

            self._touch(uid)

        return [comp, fatigue, trust, engagement]

    def competence(
        self,
        user_id: int,
        category: Optional[str] = None,
    ) -> float:
        """Return current competence for a user's category."""
        category = str(category or self.default_category)
        uid = int(user_id)
        with self._lock:
            bk = self._trackers.get(uid, {}).get(category)
            return (
                float(bk.p_known)
                if bk is not None
                else float(self._bkt_kwargs.get("p_init", 0.1))
            )

    @property
    def num_users(self) -> int:
        """Number of users tracked by this shard."""
        with self._lock:
            return len(self._trackers)


class ShardedBKTStateProvider:
    """Partitions BKT state across N shards for parallel access.

    Each shard has its own lock, so concurrent ``observe()`` calls for
    users in different shards do not contend.  This alone gives ~Nx
    throughput improvement under concurrent load.

    For cross-machine sharding, combine with user_id-based Kafka
    partitioning: partition K's events go to shard K's ingestor.

    Parameters
    ----------
    num_shards : int
        Number of independent shards.  More shards reduce contention but
        increase per-shard overhead.  16 is a sensible default for single-
        machine deployments with up to ~64 threads.
    state_dim : int
        Dimensionality of the state vector.  Must be 4 (the only value
        currently supported by the tower).
    bkt_kwargs : dict or None
        Forwarded to :class:`BayesianKnowledgeTracing` when creating a
        new per-user tracker.
    max_users_per_shard : int
        Maximum tracked users per shard.  When a shard exceeds this
        limit the least-recently-used user is evicted.
    default_category : str
        Key used when an interaction does not specify a category.
    """

    def __init__(
        self,
        num_shards: int = 16,
        *,
        state_dim: int = 4,
        bkt_kwargs: dict | None = None,
        max_users_per_shard: int = 500_000,
        default_category: str = "__default__",
    ) -> None:
        if num_shards <= 0:
            raise ValueError(f"num_shards must be positive, got {num_shards}")
        if state_dim != 4:
            raise ValueError(
                "ShardedBKTStateProvider currently only supports state_dim=4 "
                "([competence, fatigue, trust, engagement])."
            )

        self.num_shards = int(num_shards)
        self.state_dim = int(state_dim)
        self.default_category = str(default_category)
        self._bkt_kwargs = dict(bkt_kwargs or {})

        self._shards: List[_BKTShard] = [
            _BKTShard(
                shard_index=i,
                bkt_kwargs=self._bkt_kwargs,
                default_category=self.default_category,
                max_users=max_users_per_shard,
            )
            for i in range(self.num_shards)
        ]

        logger.info(
            "ShardedBKTStateProvider created: %d shards, max_users_per_shard=%s",
            self.num_shards,
            f"{max_users_per_shard:,}",
        )

    def _shard_for(self, user_id: int) -> _BKTShard:
        """Route a user to the appropriate shard."""
        return self._shards[int(user_id) % self.num_shards]

    # ---- mutation ----

    def observe(
        self,
        user_id: int,
        correct: bool = True,
        *,
        category: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> float:
        """Record one interaction and return the updated competence.

        Parameters
        ----------
        user_id : int
            User identifier.
        correct : bool
            Whether the user's response was correct.
        category : str or None
            Category / skill label.  Falls back to the default category
            if ``None``.
        timestamp : float or None
            Unix timestamp of the interaction.  Defaults to ``time.time()``.

        Returns
        -------
        float
            Updated P(known) for the category, in ``[0, 1]``.
        """
        shard = self._shard_for(user_id)
        return shard.observe(
            user_id,
            bool(correct),
            category=category,
            timestamp=timestamp,
        )

    # ---- read ----

    def state_vector(
        self,
        user_id: int,
        *,
        now: Optional[float] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the current user state vector as a ``1 x state_dim`` tensor.

        The vector is ``[competence, fatigue, trust, engagement]`` with each
        component in ``[0, 1]``.

        Parameters
        ----------
        user_id : int
            User identifier.
        now : float or None
            Reference timestamp for telemetry-derived components.  Defaults
            to ``time.time()``.
        device : str or torch.device
            Target device for the returned tensor.
        dtype : torch.dtype
            Data type for the returned tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(1, state_dim)`` tensor.
        """
        shard = self._shard_for(user_id)
        vec = shard.state_vector(user_id, now=now)
        return torch.tensor([vec], device=device, dtype=dtype)

    # Alias matching BKTStateProvider.state_vec() for drop-in compatibility
    # with StreamingAdaptiveRanker, which calls self.state.state_vec().
    state_vec = state_vector

    def competence(
        self,
        user_id: int,
        category: Optional[str] = None,
    ) -> float:
        """Return the current competence for a user's category.

        Parameters
        ----------
        user_id : int
            User identifier.
        category : str or None
            Category label.  Falls back to the default category if ``None``.

        Returns
        -------
        float
            P(known) for the category, in ``[0, 1]``.
        """
        shard = self._shard_for(user_id)
        return shard.competence(user_id, category=category)

    # ---- diagnostics ----

    @property
    def total_tracked_users(self) -> int:
        """Total number of users tracked across all shards."""
        return sum(s.num_users for s in self._shards)

    @property
    def shard_sizes(self) -> list[int]:
        """Number of tracked users per shard, ordered by shard index."""
        return [s.num_users for s in self._shards]

    def __repr__(self) -> str:
        return (
            f"ShardedBKTStateProvider(shards={self.num_shards}, "
            f"total_users={self.total_tracked_users:,})"
        )
