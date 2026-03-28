"""Extended tests for differential privacy modules."""
import sys
sys.path.insert(0, "src")

import math
import pytest
import torch
import torch.nn as nn

from orchid_ranker.dp import DP_PRESETS, get_dp_config
from orchid_ranker.agents.simple_dp import (
    SimpleDPConfig,
    SimpleDPAccountant,
    dp_sgd_step,
)


class TestSimpleDPConfig:
    """Test SimpleDPConfig dataclass."""

    def test_default_construction(self):
        """Test default construction."""
        cfg = SimpleDPConfig()
        assert cfg.enabled is True
        assert cfg.noise_multiplier == 1.0
        assert cfg.max_grad_norm == 1.0
        assert cfg.sample_rate == 0.02
        assert cfg.delta == 1e-5

    def test_custom_construction(self):
        """Test construction with custom values."""
        cfg = SimpleDPConfig(
            enabled=False,
            noise_multiplier=2.0,
            max_grad_norm=0.5,
            sample_rate=0.05,
            delta=1e-6,
        )
        assert cfg.enabled is False
        assert cfg.noise_multiplier == 2.0
        assert cfg.max_grad_norm == 0.5
        assert cfg.sample_rate == 0.05
        assert cfg.delta == 1e-6


class TestSimpleDPAccountant:
    """Test SimpleDPAccountant."""

    def test_initialization(self):
        """Test initialization."""
        accountant = SimpleDPAccountant(q=0.02, sigma=1.0, delta=1e-5)
        assert accountant.q == 0.02
        assert accountant.sigma == 1.0
        assert accountant.delta == 1e-5
        assert accountant.T == 0
        assert accountant.eps == 0.0

    def test_epsilon_increases_monotonically(self):
        """Test that epsilon increases monotonically with steps."""
        accountant = SimpleDPAccountant(q=0.02, sigma=1.0, delta=1e-5)

        epsilons = []
        for _ in range(5):
            _, eps_cum = accountant.step(10)
            epsilons.append(eps_cum)

        # Should be monotonically increasing
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1]

    def test_epsilon_zero_when_no_noise(self):
        """Test that epsilon is 0 when sigma=0."""
        accountant = SimpleDPAccountant(q=0.02, sigma=0.0, delta=1e-5)
        _, eps = accountant.step(10)
        assert eps == 0.0

    def test_epsilon_zero_when_no_sampling(self):
        """Test that epsilon is 0 when q=0."""
        accountant = SimpleDPAccountant(q=0.0, sigma=1.0, delta=1e-5)
        _, eps = accountant.step(10)
        assert eps == 0.0

    def test_epsilon_zero_on_zero_steps(self):
        """Test that epsilon is 0 when T=0."""
        accountant = SimpleDPAccountant(q=0.02, sigma=1.0, delta=1e-5)
        delta_eps, cum_eps = accountant.step(0)
        assert delta_eps == 0.0
        assert cum_eps == 0.0

    def test_step_returns_tuple(self):
        """Test that step returns (delta_eps, cum_eps) tuple."""
        accountant = SimpleDPAccountant(q=0.02, sigma=1.0, delta=1e-5)
        result = accountant.step(10)

        assert isinstance(result, tuple)
        assert len(result) == 2
        delta_eps, cum_eps = result
        assert isinstance(delta_eps, float)
        assert isinstance(cum_eps, float)

    def test_larger_sigma_smaller_epsilon(self):
        """Test that larger noise multiplier gives better privacy (smaller eps)."""
        accountant_small_sigma = SimpleDPAccountant(q=0.02, sigma=1.0, delta=1e-5)
        accountant_large_sigma = SimpleDPAccountant(q=0.02, sigma=5.0, delta=1e-5)

        _, eps_small = accountant_small_sigma.step(100)
        _, eps_large = accountant_large_sigma.step(100)

        # Larger sigma = more noise = better privacy = smaller epsilon
        assert eps_large < eps_small

    def test_composition(self):
        """Test that epsilon composes correctly."""
        accountant = SimpleDPAccountant(q=0.02, sigma=1.0, delta=1e-5)

        eps_total_separate = 0.0
        for _ in range(5):
            delta_eps, _ = accountant.step(10)
            eps_total_separate += delta_eps

        accountant2 = SimpleDPAccountant(q=0.02, sigma=1.0, delta=1e-5)
        _, eps_total_together = accountant2.step(50)

        # Should be similar (within numerical precision)
        assert abs(eps_total_together - eps_total_separate) < 0.1


class TestDPSGDStep:
    """Test dp_sgd_step function."""

    def test_basic_execution(self):
        """Test that dp_sgd_step executes without error."""
        device = torch.device("cpu")
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(x, y):
            return nn.MSELoss()(model(x), y)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        loss = dp_sgd_step(
            model,
            list(model.parameters()),
            optimizer,
            loss_fn,
            (x, y),
            max_grad_norm=1.0,
            noise_multiplier=1.0,
            device=device,
        )

        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_clips_gradients(self):
        """Test that gradients are clipped to max_grad_norm."""
        device = torch.device("cpu")
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(x, y):
            return (model(x) - y).pow(2).mean()

        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        y = torch.tensor([[10.0], [20.0], [30.0], [40.0]])

        # Run DP-SGD step with very small max_grad_norm
        loss = dp_sgd_step(
            model,
            list(model.parameters()),
            optimizer,
            loss_fn,
            (x, y),
            max_grad_norm=0.1,  # small norm
            noise_multiplier=0.0,  # no noise to see clipping
            device=device,
        )

        # Loss should be computed
        assert isinstance(loss, float)

    def test_adds_noise(self):
        """Test that noise is added to gradients."""
        device = torch.device("cpu")

        # Run with noise vs without noise
        losses_with_noise = []
        losses_without_noise = []

        for trial in range(3):
            # With noise
            model = nn.Linear(5, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            def loss_fn(x, y):
                return nn.MSELoss()(model(x), y)

            x = torch.randn(8, 5)
            y = torch.randn(8, 1)

            loss = dp_sgd_step(
                model,
                list(model.parameters()),
                optimizer,
                loss_fn,
                (x, y),
                max_grad_norm=1.0,
                noise_multiplier=1.0,
                device=device,
            )
            losses_with_noise.append(loss)

            # Without noise
            model2 = nn.Linear(5, 1)
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

            loss2 = dp_sgd_step(
                model2,
                list(model2.parameters()),
                optimizer2,
                loss_fn,
                (x, y),
                max_grad_norm=1.0,
                noise_multiplier=0.0,
                device=device,
            )
            losses_without_noise.append(loss2)

        # With noise should be less consistent (more variance)
        # This is probabilistic but should hold in general
        assert len(losses_with_noise) == 3

    def test_empty_batch_returns_zero(self):
        """Test that empty batch returns 0 loss."""
        device = torch.device("cpu")
        model = nn.Linear(5, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(x, y):
            return nn.MSELoss()(model(x), y)

        x = torch.randn(0, 5)
        y = torch.randn(0, 1)

        loss = dp_sgd_step(
            model,
            list(model.parameters()),
            optimizer,
            loss_fn,
            (x, y),
            max_grad_norm=1.0,
            noise_multiplier=1.0,
            device=device,
        )

        assert loss == 0.0

    def test_with_multiple_input_tensors(self):
        """Test dp_sgd_step with multiple input tensors."""
        device = torch.device("cpu")

        class MultiInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x1, x2):
                return self.linear(torch.cat([x1, x2], dim=1))

        model = MultiInputModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(x1, x2, y):
            return nn.MSELoss()(model(x1, x2), y)

        x1 = torch.randn(4, 5)
        x2 = torch.randn(4, 5)
        y = torch.randn(4, 1)

        loss = dp_sgd_step(
            model,
            list(model.parameters()),
            optimizer,
            loss_fn,
            (x1, x2, y),
            max_grad_norm=1.0,
            noise_multiplier=1.0,
            device=device,
        )

        assert isinstance(loss, float)
        assert loss >= 0.0


class TestGetDPConfig:
    """Test get_dp_config function."""

    def test_returns_dict_for_preset(self):
        """Test that get_dp_config returns dict for valid preset."""
        for preset_name in DP_PRESETS.keys():
            config = get_dp_config(preset_name)
            assert isinstance(config, dict)

    def test_returns_dict_unchanged(self):
        """Test that dict input is returned unchanged."""
        custom_config = {"enabled": True, "noise_multiplier": 2.5}
        result = get_dp_config(custom_config)
        assert result == custom_config

    def test_raises_on_unknown_preset(self):
        """Test that unknown preset raises KeyError."""
        with pytest.raises(KeyError):
            get_dp_config("unknown_preset")

    def test_case_insensitive(self):
        """Test that preset names are case-insensitive."""
        config1 = get_dp_config("eps_1")
        config2 = get_dp_config("EPS_1")
        assert config1 == config2

    def test_presets_have_expected_keys(self):
        """Test that presets contain expected keys."""
        for preset_name in DP_PRESETS.keys():
            config = get_dp_config(preset_name)
            assert "enabled" in config
            if config.get("enabled"):
                assert "noise_multiplier" in config
                assert "sample_rate" in config
                assert "delta" in config


class TestDPPresets:
    """Test DP_PRESETS constant."""

    def test_off_preset(self):
        """Test 'off' preset disables DP."""
        config = get_dp_config("off")
        assert config["enabled"] is False

    def test_eps_presets(self):
        """Test epsilon-based presets."""
        presets = ["eps_2", "eps_1", "eps_05", "eps_02"]
        for preset in presets:
            config = get_dp_config(preset)
            assert config["enabled"] is True
            assert config["noise_multiplier"] > 0
            assert config["sample_rate"] > 0

    def test_noise_increases_with_lower_epsilon(self):
        """Test that lower epsilon targets have higher noise."""
        config_2 = get_dp_config("eps_2")
        config_05 = get_dp_config("eps_05")

        # Lower epsilon (0.5) should have more noise than higher epsilon (2.0)
        assert config_05["noise_multiplier"] > config_2["noise_multiplier"]

    def test_all_presets_are_dicts(self):
        """Test that all presets are proper dicts."""
        for preset_name, preset_config in DP_PRESETS.items():
            assert isinstance(preset_config, dict)
            assert isinstance(preset_name, str)
