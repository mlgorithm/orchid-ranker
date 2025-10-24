import torch
import pytest

from orchid_ranker.dp import get_dp_config, DP_PRESETS
from orchid_ranker.agents.simple_dp import SimpleDPConfig
from orchid_ranker.agents.recommender_agent import TwoTowerRecommender
from orchid_ranker.dp_accountant import build_accountant


def test_dp_config_known_presets_round_trip():
    for name, cfg in DP_PRESETS.items():
        resolved = get_dp_config(name)
        assert resolved == cfg


def test_dp_accountant_monotonicity():
    cfg = SimpleDPConfig(enabled=True, sample_rate=0.05, noise_multiplier=1.2, delta=1e-5)
    accountant = build_accountant("per_sample", cfg)
    _, eps0 = accountant.step(0)
    assert eps0 == 0.0

    incr1, eps1 = accountant.step(10)
    incr2, eps2 = accountant.step(10)

    assert incr1 >= 0.0 and incr2 >= 0.0
    assert eps2 > eps1 > eps0
    assert abs((incr1 + incr2) - (eps2 - eps0)) < 1e-9


def test_dp_per_sample_update_executes():
    dp_cfg = {
        "enabled": True,
        "noise_multiplier": 0.5,
        "sample_rate": 1.0,
        "delta": 1e-5,
        "max_grad": 1.0,
        "engine": "per_sample",
    }

    model = TwoTowerRecommender(
        num_users=1,
        num_items=2,
        user_dim=4,
        item_dim=4,
        device="cpu",
        dp_cfg=dp_cfg,
    )

    user_vec = torch.randn(1, 4)
    state_vec = torch.zeros(1, model.state_dim)
    user_ids = torch.tensor([0])
    item_matrix = torch.randn(2, 4)
    item_ids = torch.tensor([0, 1])
    feedback = {int(item_ids[0].item()): 1}

    stats = model.update(
        feedback=feedback,
        user_vec=user_vec,
        state_vec=state_vec,
        user_ids=user_ids,
        item_matrix=item_matrix,
        item_ids=item_ids,
        epochs=1,
    )

    assert "loss" in stats
    assert stats["epsilon_cum"] >= 0.0


def test_opacus_accountant_available_when_installed():
    pytest.importorskip("opacus")
    cfg = SimpleDPConfig(enabled=True, sample_rate=0.05, noise_multiplier=1.0, delta=1e-5)
    accountant = build_accountant("opacus", cfg)
    incr1, eps1 = accountant.step(5)
    incr2, eps2 = accountant.step(5)
    assert incr1 >= 0.0 and incr2 >= 0.0
    assert eps2 >= eps1 >= 0.0
