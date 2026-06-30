"""Concurrency tests for StreamingAdaptiveRanker.

The library ships a background ingestor that calls ``observe()`` on a daemon
thread while the service serves ``rank()`` from request threads. ``observe()``
mutates the shared tower state / BKT state / residual adapter; ``rank()`` reads
them. These tests exercise both paths concurrently and assert that:

* no thread raises (FIX 1: rank() now holds the ranker RLock while it touches
  shared state, and OnlineUserAdapter.forward() returns a clone so a reader
  never sees a half-written residual row),
* every rank() result stays well-formed (unique ids, finite scores).
"""
from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from orchid_ranker.agents.two_tower import TwoTowerRecommender
from orchid_ranker.streaming import OnlineUserAdapter, StreamingAdaptiveRanker

NUM_USERS = 16
NUM_ITEMS = 32
FEAT_DIM = 6
EMB_DIM = 8


def _make_ranker(**kwargs) -> StreamingAdaptiveRanker:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    uf = torch.tensor(rng.normal(size=(NUM_USERS, FEAT_DIM)).astype(np.float32))
    ifeat = torch.tensor(rng.normal(size=(NUM_ITEMS, FEAT_DIM)).astype(np.float32))
    tower = TwoTowerRecommender(
        num_users=NUM_USERS, num_items=NUM_ITEMS,
        user_dim=FEAT_DIM, item_dim=FEAT_DIM,
        hidden=16, emb_dim=EMB_DIM, state_dim=4,
        device="cpu",
    ).eval()
    return StreamingAdaptiveRanker(tower, uf, ifeat, **kwargs)


class TestRankObserveConcurrency:
    NUM_THREADS = 8
    ITERS = 80

    def test_concurrent_rank_and_observe_no_errors(self) -> None:
        ranker = _make_ranker(lr=0.5)
        cand = list(range(NUM_ITEMS))
        errors: list[Exception] = []
        barrier = threading.Barrier(self.NUM_THREADS)

        def ranker_worker(seed: int) -> None:
            try:
                barrier.wait()
                for k in range(self.ITERS):
                    uid = (seed + k) % NUM_USERS
                    out = ranker.rank(user_id=uid, candidate_item_ids=cand, top_k=5)
                    ids = [i for i, _ in out]
                    assert len(ids) == len(set(ids)), "ranked ids must be unique"
                    assert all(math.isfinite(s) for _, s in out), "scores must be finite"
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        def observe_worker(seed: int) -> None:
            try:
                barrier.wait()
                for k in range(self.ITERS):
                    uid = (seed + k) % NUM_USERS
                    ranker.observe(user_id=uid, item_id=k % NUM_ITEMS, correct=bool(k % 2))
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        half = self.NUM_THREADS // 2
        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as pool:
            futures = []
            for t in range(half):
                futures.append(pool.submit(ranker_worker, t))
            for t in range(self.NUM_THREADS - half):
                futures.append(pool.submit(observe_worker, t))
            for f in as_completed(futures):
                f.result()

        assert errors == [], f"threads raised: {errors}"

    def test_concurrent_rank_and_observe_scaled_backend(self) -> None:
        from orchid_ranker.scaling import ScalingConfig

        cfg = ScalingConfig(max_active_users=64, num_state_shards=4)
        ranker = _make_ranker(lr=0.5, scaling_config=cfg)
        cand = list(range(NUM_ITEMS))
        errors: list[Exception] = []

        def worker(seed: int) -> None:
            try:
                for k in range(self.ITERS):
                    uid = (seed + k) % NUM_USERS
                    ranker.observe(user_id=uid, item_id=k % NUM_ITEMS, correct=bool(k % 2))
                    out = ranker.rank(user_id=uid, candidate_item_ids=cand, top_k=4)
                    assert all(math.isfinite(s) for _, s in out)
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as pool:
            futures = [pool.submit(worker, t) for t in range(self.NUM_THREADS)]
            for f in as_completed(futures):
                f.result()

        assert errors == [], f"threads raised: {errors}"


class TestAdapterForwardSnapshot:
    def test_forward_returns_independent_snapshot(self) -> None:
        """forward() must not alias the live residual weight (FIX 1)."""
        ad = OnlineUserAdapter(num_users=4, emb_dim=EMB_DIM, lr=0.5)
        u = torch.zeros(EMB_DIM)
        i = torch.ones(EMB_DIM)
        ad.observe(0, u, i, y=1.0)
        snap = ad(torch.tensor([0]))
        before = snap.clone()
        # A subsequent in-place update of the live weight must NOT mutate the
        # tensor a previous caller is holding.
        ad.observe(0, u, i, y=1.0)
        assert torch.allclose(snap, before), (
            "forward() result aliased the live embedding weight"
        )

    def test_forward_snapshot_storage_is_not_shared(self) -> None:
        ad = OnlineUserAdapter(num_users=2, emb_dim=EMB_DIM, lr=0.5)
        ad.observe(1, torch.zeros(EMB_DIM), torch.ones(EMB_DIM), y=1.0)
        snap = ad(torch.tensor([1]))
        # Must not share storage with the underlying Embedding weight.
        assert snap.data_ptr() != ad.residual.weight[1].data_ptr()
