"""Tests for orchid_ranker.scaling -- sparse embeddings, adapters, sharded BKT."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import torch

from orchid_ranker.scaling import (
    ScalingConfig,
    ShardedBKTStateProvider,
    SparseEmbeddingTable,
    SparseOnlineUserAdapter,
)

# =========================================================================
# SparseEmbeddingTable
# =========================================================================


class TestSparseEmbeddingTable:
    """Unit tests for SparseEmbeddingTable."""

    def test_empty_table_returns_zeros(self) -> None:
        table = SparseEmbeddingTable(emb_dim=16, max_entries=100)
        vec = table[999]
        assert vec.shape == (16,)
        assert torch.allclose(vec, torch.zeros(16))

    def test_set_and_get(self) -> None:
        table = SparseEmbeddingTable(emb_dim=8, max_entries=100)
        v = torch.randn(8)
        table[42] = v
        retrieved = table[42]
        assert torch.allclose(retrieved, v)

    def test_contains(self) -> None:
        table = SparseEmbeddingTable(emb_dim=4, max_entries=100)
        table[10] = torch.ones(4)
        assert 10 in table
        assert 99 not in table

    def test_len(self) -> None:
        table = SparseEmbeddingTable(emb_dim=4, max_entries=100)
        assert len(table) == 0
        table[1] = torch.ones(4)
        table[2] = torch.ones(4)
        table[3] = torch.ones(4)
        assert len(table) == 3

    def test_lru_eviction(self) -> None:
        table = SparseEmbeddingTable(emb_dim=4, max_entries=3)
        for uid in [1, 2, 3]:
            table[uid] = torch.full((4,), float(uid))
        # All three present.
        assert len(table) == 3
        assert 1 in table
        # Adding a 4th should evict user 1 (least recently used).
        table[4] = torch.full((4,), 4.0)
        assert len(table) == 3
        assert 1 not in table
        assert 4 in table

    def test_access_refreshes_lru(self) -> None:
        table = SparseEmbeddingTable(emb_dim=4, max_entries=3)
        for uid in [1, 2, 3]:
            table[uid] = torch.full((4,), float(uid))
        # Access user 1 to refresh it (moves to end of LRU).
        _ = table[1]
        # Now user 2 is LRU. Adding user 4 should evict user 2.
        table[4] = torch.full((4,), 4.0)
        assert 1 in table, "user 1 should survive (was refreshed)"
        assert 2 not in table, "user 2 should be evicted (was LRU)"

    def test_manual_evict(self) -> None:
        table = SparseEmbeddingTable(emb_dim=4, max_entries=100)
        table[5] = torch.ones(4)
        assert 5 in table
        table.evict(5)
        assert 5 not in table
        assert len(table) == 0
        # Evicting a non-existent user is a no-op.
        table.evict(999)

    def test_memory_bytes(self) -> None:
        table = SparseEmbeddingTable(emb_dim=32, max_entries=100)
        assert table.memory_bytes() == 0
        table[1] = torch.randn(32)
        table[2] = torch.randn(32)
        # Each float32 entry: 32 * 4 = 128 bytes. Two entries: 256.
        assert table.memory_bytes() == 256

    def test_occupancy(self) -> None:
        table = SparseEmbeddingTable(emb_dim=4, max_entries=10)
        assert table.occupancy == pytest.approx(0.0)
        for i in range(5):
            table[i] = torch.ones(4)
        assert table.occupancy == pytest.approx(0.5)

    def test_thread_safety(self) -> None:
        table = SparseEmbeddingTable(emb_dim=8, max_entries=200)
        errors: list[Exception] = []

        def worker(offset: int) -> None:
            try:
                for i in range(50):
                    uid = offset * 100 + i
                    table[uid] = torch.randn(8)
                    _ = table[uid]
                    _ = len(table)
                    _ = uid in table
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"


# =========================================================================
# SparseOnlineUserAdapter
# =========================================================================


class TestSparseOnlineUserAdapter:
    """Unit tests for SparseOnlineUserAdapter."""

    def test_new_user_returns_zeros(self) -> None:
        adapter = SparseOnlineUserAdapter(emb_dim=16)
        out = adapter.forward(torch.tensor([999]))
        assert out.shape == (1, 16)
        assert torch.allclose(out, torch.zeros(1, 16))

    def test_observe_creates_residual(self) -> None:
        adapter = SparseOnlineUserAdapter(emb_dim=8)
        u_base = torch.randn(8)
        item_emb = torch.randn(8)
        adapter.observe(user_id=1, u_base=u_base, item_emb=item_emb, y=1.0)
        out = adapter.forward(torch.tensor([1]))
        assert not torch.allclose(out, torch.zeros(1, 8)), (
            "After observe(), residual should be non-zero"
        )

    def test_observe_sgd_direction(self) -> None:
        adapter = SparseOnlineUserAdapter(emb_dim=8, lr=0.1, l2=0.0, clip=10.0)
        u_base = torch.zeros(8)
        item_emb = torch.ones(8) / (8 ** 0.5)  # unit norm
        # y=1 (positive label): gradient pushes residual toward item direction.
        adapter.observe(user_id=1, u_base=u_base, item_emb=item_emb, y=1.0)
        residual = adapter.forward(torch.tensor([1]))[0]
        dot_product = torch.dot(residual, item_emb).item()
        assert dot_product > 0, (
            "Residual should move toward item direction for positive label"
        )

    def test_updates_for(self) -> None:
        adapter = SparseOnlineUserAdapter(emb_dim=4)
        assert adapter.updates_for(1) == 0
        u = torch.randn(4)
        v = torch.randn(4)
        adapter.observe(1, u, v, 1.0)
        adapter.observe(1, u, v, 0.0)
        adapter.observe(1, u, v, 1.0)
        assert adapter.updates_for(1) == 3

    def test_reset_user(self) -> None:
        adapter = SparseOnlineUserAdapter(emb_dim=4)
        u = torch.randn(4)
        v = torch.randn(4)
        adapter.observe(1, u, v, 1.0)
        assert adapter.updates_for(1) == 1
        adapter.reset_user(1)
        out = adapter.forward(torch.tensor([1]))
        assert torch.allclose(out, torch.zeros(1, 4))
        assert adapter.updates_for(1) == 0

    def test_active_users_count(self) -> None:
        adapter = SparseOnlineUserAdapter(emb_dim=4)
        assert adapter.active_users == 0
        u = torch.randn(4)
        v = torch.randn(4)
        for uid in [10, 20, 30]:
            adapter.observe(uid, u, v, 1.0)
        assert adapter.active_users == 3

    def test_eviction_under_pressure(self) -> None:
        adapter = SparseOnlineUserAdapter(emb_dim=4, max_active_users=5)
        u = torch.randn(4)
        v = torch.randn(4)
        for uid in range(10):
            adapter.observe(uid, u, v, 1.0)
        assert adapter.active_users <= 5

    def test_api_compatible_with_original(self) -> None:
        """The observe() signature matches the dense OnlineUserAdapter."""
        adapter = SparseOnlineUserAdapter(emb_dim=8)
        result = adapter.observe(
            user_id=42,
            u_base=torch.randn(8),
            item_emb=torch.randn(8),
            y=0.5,
        )
        # Returns squared norm of residual (a float).
        assert isinstance(result, float)
        assert result >= 0.0


# =========================================================================
# ShardedBKTStateProvider
# =========================================================================


class TestShardedBKTStateProvider:
    """Unit tests for ShardedBKTStateProvider."""

    def test_observe_and_competence(self) -> None:
        provider = ShardedBKTStateProvider(num_shards=4)
        initial = provider.competence(user_id=1)
        # Observing correct interactions should increase competence.
        for _ in range(10):
            provider.observe(user_id=1, correct=True)
        updated = provider.competence(user_id=1)
        assert updated > initial

    def test_sharding_distributes_users(self) -> None:
        provider = ShardedBKTStateProvider(num_shards=4)
        # Users with ids 0..15 should spread across shards.
        for uid in range(16):
            provider.observe(user_id=uid, correct=True)
        sizes = provider.shard_sizes
        assert len(sizes) == 4
        # Each shard should have exactly 4 users (0..15 mod 4).
        assert all(s == 4 for s in sizes)

    def test_state_vector_shape(self) -> None:
        provider = ShardedBKTStateProvider(num_shards=2)
        provider.observe(user_id=7, correct=True)
        sv = provider.state_vector(user_id=7)
        assert isinstance(sv, torch.Tensor)
        assert sv.shape == (1, 4)

    def test_total_tracked_users(self) -> None:
        provider = ShardedBKTStateProvider(num_shards=4)
        assert provider.total_tracked_users == 0
        for uid in [10, 20, 30, 40, 50]:
            provider.observe(user_id=uid, correct=True)
        assert provider.total_tracked_users == 5

    def test_shard_sizes(self) -> None:
        provider = ShardedBKTStateProvider(num_shards=3)
        for uid in range(9):
            provider.observe(user_id=uid, correct=True)
        sizes = provider.shard_sizes
        assert len(sizes) == 3
        assert sum(sizes) == 9

    def test_concurrent_observes(self) -> None:
        provider = ShardedBKTStateProvider(num_shards=4)
        errors: list[Exception] = []

        def worker(uid_start: int) -> None:
            try:
                for uid in range(uid_start, uid_start + 25):
                    provider.observe(user_id=uid, correct=True)
                    _ = provider.competence(uid)
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, i * 100) for i in range(4)]
            for f in as_completed(futures):
                f.result()

        assert errors == [], f"Concurrent observe errors: {errors}"
        assert provider.total_tracked_users == 100

    def test_eviction_per_shard(self) -> None:
        provider = ShardedBKTStateProvider(
            num_shards=2, max_users_per_shard=3,
        )
        # Add many users that all hash to shard 0 (even ids).
        even_ids = [0, 2, 4, 6, 8, 10]
        for uid in even_ids:
            provider.observe(user_id=uid, correct=True)
        # Shard 0 should have at most 3 users.
        assert provider.shard_sizes[0] <= 3


# =========================================================================
# ScalingConfig
# =========================================================================


class TestScalingConfig:
    """Unit tests for the ScalingConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = ScalingConfig()
        assert cfg.max_active_users == 1_000_000
        assert cfg.num_state_shards == 16
        assert cfg.eviction_policy == "lru"
        assert cfg.ttl_seconds == 86400.0
        assert cfg.enable_metrics is True

    def test_custom_values(self) -> None:
        cfg = ScalingConfig(
            max_active_users=500,
            num_state_shards=8,
            eviction_policy="ttl",
            ttl_seconds=3600.0,
            enable_metrics=False,
        )
        assert cfg.max_active_users == 500
        assert cfg.num_state_shards == 8
        assert cfg.eviction_policy == "ttl"
        assert cfg.ttl_seconds == 3600.0
        assert cfg.enable_metrics is False
