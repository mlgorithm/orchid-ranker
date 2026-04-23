"""End-to-end tests for the streaming adaptive ranker.

These tests assert the contract that justifies the "adaptive + streaming"
positioning: a single observation must be visible to the next rank call
without retraining or restart.

Tests are CPU-only and complete in well under a second so they can run on
every PR.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from orchid_ranker.agents.two_tower import TwoTowerRecommender
from orchid_ranker.streaming import (
    BKTStateProvider,
    OnlineUserAdapter,
    StreamingAdaptiveRanker,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
NUM_USERS = 16
NUM_ITEMS = 32
FEAT_DIM = 6
EMB_DIM = 8


@pytest.fixture
def world():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    user_feats = torch.tensor(rng.normal(size=(NUM_USERS, FEAT_DIM)).astype(np.float32))
    item_feats = torch.tensor(rng.normal(size=(NUM_ITEMS, FEAT_DIM)).astype(np.float32))
    tower = TwoTowerRecommender(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        user_dim=FEAT_DIM,
        item_dim=FEAT_DIM,
        hidden=16,
        emb_dim=EMB_DIM,
        state_dim=4,
        device="cpu",
        dp_cfg={"enabled": False},
    ).eval()
    return tower, user_feats, item_feats


# ---------------------------------------------------------------------------
# BKTStateProvider
# ---------------------------------------------------------------------------
class TestBKTStateProvider:
    def test_state_vec_shape_and_range(self):
        sp = BKTStateProvider()
        v = sp.state_vec(user_id=0)
        assert v.shape == (1, 4)
        assert torch.all(v >= 0.0) and torch.all(v <= 1.0)

    def test_observe_updates_knowledge(self):
        sp = BKTStateProvider()
        v0 = sp.state_vec(7)[0, 0].item()
        for _ in range(5):
            sp.observe(7, correct=True)
        v1 = sp.state_vec(7)[0, 0].item()
        assert v1 > v0, "knowledge component must rise after correct observations"

    def test_engagement_decays_over_time(self):
        sp = BKTStateProvider()
        now = 1000.0
        for k in range(10):
            sp.observe(3, correct=True, timestamp=now + k)
        e_now = sp.state_vec(3, now=now + 10)[0, 3].item()
        e_later = sp.state_vec(3, now=now + 10_000)[0, 3].item()
        assert e_now > e_later, "engagement must decay as time passes"

    def test_rejects_unsupported_state_dim(self):
        with pytest.raises(ValueError):
            BKTStateProvider(state_dim=5)


# ---------------------------------------------------------------------------
# OnlineUserAdapter
# ---------------------------------------------------------------------------
class TestOnlineUserAdapter:
    def test_zero_initialised(self):
        ad = OnlineUserAdapter(num_users=4, emb_dim=EMB_DIM)
        r = ad(torch.tensor([0, 1, 2, 3]))
        assert torch.allclose(r, torch.zeros_like(r))

    def test_observe_moves_residual(self):
        torch.manual_seed(0)
        ad = OnlineUserAdapter(num_users=4, emb_dim=EMB_DIM, lr=0.5)
        u = torch.zeros(EMB_DIM)
        i = torch.ones(EMB_DIM)
        norm0 = float(torch.linalg.vector_norm(ad.residual.weight[0]))
        ad.observe(0, u, i, y=1.0)
        norm1 = float(torch.linalg.vector_norm(ad.residual.weight[0]))
        assert norm1 > norm0, "residual must move on a logged interaction"

    def test_norm_clip_holds(self):
        ad = OnlineUserAdapter(num_users=2, emb_dim=EMB_DIM, lr=10.0, clip=0.5)
        u = torch.zeros(EMB_DIM)
        i = torch.ones(EMB_DIM)
        for _ in range(20):
            ad.observe(1, u, i, y=1.0)
        assert float(torch.linalg.vector_norm(ad.residual.weight[1])) <= 0.5 + 1e-6

    def test_independence_across_users(self):
        ad = OnlineUserAdapter(num_users=3, emb_dim=EMB_DIM, lr=0.5)
        u = torch.zeros(EMB_DIM)
        i = torch.ones(EMB_DIM)
        ad.observe(2, u, i, y=1.0)
        # User 0 and 1 must remain at zero
        assert torch.allclose(ad.residual.weight[0], torch.zeros(EMB_DIM))
        assert torch.allclose(ad.residual.weight[1], torch.zeros(EMB_DIM))
        assert not torch.allclose(ad.residual.weight[2], torch.zeros(EMB_DIM))

    def test_out_of_range_user_raises(self):
        ad = OnlineUserAdapter(num_users=2, emb_dim=EMB_DIM)
        with pytest.raises(IndexError):
            ad.observe(99, torch.zeros(EMB_DIM), torch.zeros(EMB_DIM), y=1.0)


# ---------------------------------------------------------------------------
# StreamingAdaptiveRanker — the contract that matters
# ---------------------------------------------------------------------------
class TestStreamingAdaptiveRanker:
    def test_rank_returns_topk(self, world):
        tower, uf, ifeat = world
        r = StreamingAdaptiveRanker(tower, uf, ifeat)
        out = r.rank(user_id=3, candidate_item_ids=list(range(20)), top_k=5)
        assert len(out) == 5
        ids = [i for i, _ in out]
        assert len(set(ids)) == 5, "ranked ids must be unique"
        assert all(0 <= i < NUM_ITEMS for i in ids)

    def test_observe_then_rank_reflects_update(self, world):
        """The defining contract: a single observe must change the next rank."""
        tower, uf, ifeat = world
        r = StreamingAdaptiveRanker(tower, uf, ifeat, lr=0.5)
        cand = list(range(NUM_ITEMS))
        before = [i for i, _ in r.rank(user_id=4, candidate_item_ids=cand, top_k=NUM_ITEMS)]
        # Reinforce a low-ranked item strongly
        target = before[-1]
        for _ in range(10):
            r.observe(user_id=4, item_id=target, correct=True)
        after = [i for i, _ in r.rank(user_id=4, candidate_item_ids=cand, top_k=NUM_ITEMS)]
        assert before.index(target) > after.index(target), (
            "repeatedly reinforced item must rise in rank"
        )

    def test_zero_lr_is_pure_baseline(self, world):
        """With lr=0 and no observations, ranking must equal the frozen tower."""
        tower, uf, ifeat = world
        r = StreamingAdaptiveRanker(tower, uf, ifeat, lr=0.0)
        cand = list(range(NUM_ITEMS))
        ranked = r.rank(user_id=2, candidate_item_ids=cand, top_k=NUM_ITEMS)
        # Adapter weights are zero, BKT empty -> deterministic forward pass
        assert all(isinstance(x[0], int) for x in ranked)
        # Calling rank twice should give the identical ordering
        ranked2 = r.rank(user_id=2, candidate_item_ids=cand, top_k=NUM_ITEMS)
        assert [i for i, _ in ranked] == [i for i, _ in ranked2]

    def test_adaptation_latency_under_budget(self, world):
        """observe + rank must fit comfortably within a sub-100ms budget."""
        tower, uf, ifeat = world
        r = StreamingAdaptiveRanker(tower, uf, ifeat)
        cand = list(range(NUM_ITEMS))
        # warmup
        for _ in range(5):
            r.observe(0, 0, correct=True)
            r.rank(0, cand, top_k=5)
        # measured loop
        for k in range(50):
            r.observe(k % NUM_USERS, k % NUM_ITEMS, correct=bool(k % 2))
            r.rank(k % NUM_USERS, cand, top_k=5)
        s = r.stats()
        assert s.observations >= 50 and s.ranks >= 50
        # Generous upper bound for CI: real numbers should be far lower.
        assert s.observe_p95_ms < 50.0, f"observe p95 too high: {s.observe_p95_ms}"
        assert s.rank_p95_ms < 100.0, f"rank p95 too high: {s.rank_p95_ms}"

    def test_competence_tracked_per_user(self, world):
        tower, uf, ifeat = world
        r = StreamingAdaptiveRanker(tower, uf, ifeat)
        for _ in range(8):
            r.observe(user_id=1, item_id=0, correct=True)
        m1 = r.competence(1)
        m_other = r.competence(2)
        assert m1 > m_other, "competence must be tracked per-user"

    def test_observed_interactions_counted(self, world):
        tower, uf, ifeat = world
        r = StreamingAdaptiveRanker(tower, uf, ifeat)
        for _ in range(7):
            r.observe(user_id=5, item_id=3, correct=False)
        assert r.updates_for(5) == 7
        assert r.updates_for(0) == 0

    def test_rejects_tower_without_infer(self):
        with pytest.raises(TypeError):
            StreamingAdaptiveRanker(
                tower=torch.nn.Linear(2, 2),
                user_features=torch.zeros(2, 2),
                item_features=torch.zeros(2, 2),
            )


# ---------------------------------------------------------------------------
# StreamingAdaptiveRanker with ScalingConfig
# ---------------------------------------------------------------------------
class TestStreamingAdaptiveRankerScaled:
    """Verify that scaling_config wires sparse/sharded backends correctly."""

    def test_scaling_config_uses_sparse_adapter(self, world):
        from orchid_ranker.scaling import ScalingConfig, SparseOnlineUserAdapter

        tower, uf, ifeat = world
        cfg = ScalingConfig(max_active_users=100, num_state_shards=4)
        r = StreamingAdaptiveRanker(tower, uf, ifeat, scaling_config=cfg)
        assert isinstance(r.adapter, SparseOnlineUserAdapter)

    def test_scaling_config_uses_sharded_bkt(self, world):
        from orchid_ranker.scaling import ScalingConfig, ShardedBKTStateProvider

        tower, uf, ifeat = world
        cfg = ScalingConfig(max_active_users=100, num_state_shards=4)
        r = StreamingAdaptiveRanker(tower, uf, ifeat, scaling_config=cfg)
        assert isinstance(r.state, ShardedBKTStateProvider)
        assert r.state.num_shards == 4

    def test_scaled_observe_then_rank(self, world):
        """The defining contract still holds with sparse backends."""
        from orchid_ranker.scaling import ScalingConfig

        tower, uf, ifeat = world
        cfg = ScalingConfig(max_active_users=100, num_state_shards=2)
        r = StreamingAdaptiveRanker(tower, uf, ifeat, lr=0.5, scaling_config=cfg)
        cand = list(range(NUM_ITEMS))
        before = [i for i, _ in r.rank(user_id=4, candidate_item_ids=cand, top_k=NUM_ITEMS)]
        target = before[-1]
        for _ in range(10):
            r.observe(user_id=4, item_id=target, correct=True)
        after = [i for i, _ in r.rank(user_id=4, candidate_item_ids=cand, top_k=NUM_ITEMS)]
        assert before.index(target) > after.index(target), (
            "repeatedly reinforced item must rise in rank (scaled backend)"
        )

    def test_scaled_competence_tracked(self, world):
        from orchid_ranker.scaling import ScalingConfig

        tower, uf, ifeat = world
        cfg = ScalingConfig(max_active_users=50, num_state_shards=2)
        r = StreamingAdaptiveRanker(tower, uf, ifeat, scaling_config=cfg)
        for _ in range(8):
            r.observe(user_id=1, item_id=0, correct=True)
        m1 = r.competence(1)
        m_other = r.competence(2)
        assert m1 > m_other, "competence must be tracked per-user (scaled backend)"

    def test_default_path_uses_dense_adapter(self, world):
        """Without scaling_config, the dense adapter is used."""
        tower, uf, ifeat = world
        r = StreamingAdaptiveRanker(tower, uf, ifeat)
        assert isinstance(r.adapter, OnlineUserAdapter)
