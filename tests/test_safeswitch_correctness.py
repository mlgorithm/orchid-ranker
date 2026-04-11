"""Correctness tests for SafeSwitchDR and DRConfidenceSequence.

Tests focus on exact formula verification:
- Anytime-valid delta_t formula
- Welford variance tracking
- Empirical Bernstein radius
- LCB correctness
- DR estimator formula
- Clipping logic
- Acceptance guardrail
- Step-up/step-down behavior
- Boundary behavior
"""
import sys
sys.path.insert(0, "src")

import math
import pytest

from orchid_ranker.safety.dr_cs import DRConfidenceSequence, DRCSConfig
from orchid_ranker.safety.safeswitch_dr import SafeSwitchDR, SafeSwitchDRConfig


class TestDRDeltaTFormula:
    """Test anytime-valid delta_t formula: delta_t = delta/2 for t<=1, else 6*delta/(π²*t²)."""

    def test_delta_t_at_t_equals_0(self):
        """When t=0 (before first update), delta_t should be delta/2."""
        cfg = DRCSConfig(delta=0.01)
        dr = DRConfidenceSequence(cfg)
        delta_t = dr._delta_t()
        assert pytest.approx(delta_t, rel=1e-10) == 0.01 / 2.0

    def test_delta_t_at_t_equals_1(self):
        """When t=1 (after first update), delta_t should be delta/2."""
        cfg = DRCSConfig(delta=0.01)
        dr = DRConfidenceSequence(cfg)
        dr.update(True, 0.5, 0.5, 0.3, 0.5)
        delta_t = dr._delta_t()
        assert pytest.approx(delta_t, rel=1e-10) == 0.01 / 2.0

    def test_delta_t_at_t_equals_2(self):
        """When t=2, delta_t should be 6*delta/(π²*t²)."""
        cfg = DRCSConfig(delta=0.01)
        dr = DRConfidenceSequence(cfg)
        dr.update(True, 0.5, 0.5, 0.3, 0.5)
        dr.update(True, 0.6, 0.5, 0.3, 0.5)
        delta_t = dr._delta_t()
        expected = 6.0 * 0.01 / (math.pi**2 * 4.0)
        assert pytest.approx(delta_t, rel=1e-10) == expected

    def test_delta_t_at_t_equals_10(self):
        """When t=10, delta_t should follow 6*delta/(π²*100)."""
        cfg = DRCSConfig(delta=0.01)
        dr = DRConfidenceSequence(cfg)
        for _ in range(10):
            dr.update(True, 0.5, 0.5, 0.3, 0.5)
        delta_t = dr._delta_t()
        expected = 6.0 * 0.01 / (math.pi**2 * 100.0)
        assert pytest.approx(delta_t, rel=1e-10) == expected

    def test_delta_t_decreases_with_t(self):
        """delta_t should decrease as t increases (for t > 1)."""
        cfg = DRCSConfig(delta=0.01)
        dr = DRConfidenceSequence(cfg)

        delta_t_values = []
        for _ in range(20):
            dr.update(True, 0.5, 0.5, 0.3, 0.5)
            delta_t_values.append(dr._delta_t())

        # After t=2, delta_t should be decreasing
        assert delta_t_values[2] > delta_t_values[10]
        assert delta_t_values[10] > delta_t_values[19]


class TestWelfordVarianceTracking:
    """Test Welford variance tracking (M2 accumulation and mean updates)."""

    def test_constant_sequence_zero_variance(self):
        """If all observations are equal, M2 should be 0."""
        cfg = DRCSConfig()
        dr = DRConfidenceSequence(cfg)

        # Feed constant z = 0.5
        for _ in range(10):
            dr.update(True, 0.5, 0.5, 0.3, 0.5)

        var_hat = dr.M2 / max(dr.t - 1, 1)
        assert pytest.approx(var_hat, abs=1e-8) == 0.0

    def test_mean_convergence_simple_sequence(self):
        """Mean should converge to actual mean of observations."""
        cfg = DRCSConfig()
        dr = DRConfidenceSequence(cfg)

        # Simple sequence: z values
        z_values = [0.0, 1.0, 0.5]
        for z in z_values:
            # Construct inputs so DR estimator produces z
            # For simplicity: served_adaptive=True, u=z, Qa=z, Qf=0
            dr.update(True, z, z, 0.0, 0.5)

        expected_mean = sum(z_values) / len(z_values)
        assert pytest.approx(dr.mean, rel=1e-10) == expected_mean

    def test_variance_tracking_known_sequence(self):
        """Test variance tracking on a known sequence."""
        cfg = DRCSConfig()  # u_max=1.0 by default
        dr = DRConfidenceSequence(cfg)

        # Sequence within [0, u_max] to avoid clamping.
        # z_i = [0.2, 0.5, 0.8] -> mean = 0.5, variance = 0.09
        z_values = [0.2, 0.5, 0.8]
        for z in z_values:
            dr.update(True, z, z, 0.0, 0.5)

        var_hat = dr.M2 / max(dr.t - 1, 1)
        expected_var = 0.09  # variance of [0.2, 0.5, 0.8]
        assert pytest.approx(var_hat, rel=1e-6) == expected_var

    def test_m2_accumulation_order(self):
        """M2 should be calculated correctly regardless of observation order."""
        cfg = DRCSConfig()
        dr1 = DRConfidenceSequence(cfg)
        dr2 = DRConfidenceSequence(cfg)

        z_values = [0.1, 0.5, 0.9]

        for z in z_values:
            dr1.update(True, z, z, 0.0, 0.5)

        for z in reversed(z_values):
            dr2.update(True, z, z, 0.0, 0.5)

        # Both should have same mean and variance
        assert pytest.approx(dr1.mean, rel=1e-10) == dr2.mean
        assert pytest.approx(dr1.M2, rel=1e-10) == dr2.M2


class TestEmpiricalBernsteinRadius:
    """Test empirical Bernstein radius: sqrt(2*var*log/n) + (2*B*log)/(3*n)."""

    def test_radius_formula_components(self):
        """Verify radius formula computes correctly."""
        cfg = DRCSConfig(delta=0.01, u_max=1.0, p_min=0.05)
        dr = DRConfidenceSequence(cfg)

        # Add 10 observations
        for _ in range(10):
            dr.update(True, 0.5, 0.5, 0.3, 0.5)

        # Manually compute expected radius
        n = dr.t
        var_hat = max(0.0, dr.M2 / max(n - 1, 1))
        log = math.log(2.0 / max(dr._delta_t(), 1e-12))
        expected_radius = math.sqrt(2.0 * var_hat * log / n) + (2.0 * dr.B * log) / (3.0 * n)

        actual_radius = dr._radius()
        assert pytest.approx(actual_radius, rel=1e-10) == expected_radius

    def test_radius_positive(self):
        """Radius should always be non-negative."""
        cfg = DRCSConfig()
        dr = DRConfidenceSequence(cfg)

        for _ in range(5):
            dr.update(True, 0.5, 0.5, 0.3, 0.5)
            radius = dr._radius()
            assert radius >= 0.0

    def test_radius_decreases_with_sample_size(self):
        """Radius should decrease as n increases."""
        cfg = DRCSConfig()
        dr = DRConfidenceSequence(cfg)

        radii = []
        for _ in range(20):
            dr.update(True, 0.5, 0.5, 0.3, 0.5)
            radii.append(dr._radius())

        # After enough samples, radius should generally decrease
        # (allowing for some non-monotonicity due to variance changes)
        assert radii[-1] <= radii[0]

    def test_radius_with_high_variance(self):
        """Radius should be larger with higher variance."""
        cfg = DRCSConfig()
        dr_low_var = DRConfidenceSequence(cfg)
        dr_high_var = DRConfidenceSequence(cfg)

        # Low variance: constant z
        for _ in range(10):
            dr_low_var.update(True, 0.5, 0.5, 0.3, 0.5)

        # High variance: alternating z
        for i in range(10):
            z = 0.0 if i % 2 == 0 else 1.0
            dr_high_var.update(True, z, z, 0.0, 0.5)

        assert dr_high_var._radius() > dr_low_var._radius()


class TestLCBCorrectness:
    """Test LCB = mean - radius."""

    def test_lcb_formula(self):
        """LCB should equal mean - radius."""
        cfg = DRCSConfig()
        dr = DRConfidenceSequence(cfg)

        for _ in range(5):
            dr.update(True, 0.5, 0.5, 0.3, 0.5)

        lcb = dr.lcb()
        expected_lcb = dr.mean - dr._radius()
        assert pytest.approx(lcb, rel=1e-10) == expected_lcb

    def test_lcb_less_than_mean(self):
        """LCB should always be <= mean."""
        cfg = DRCSConfig()
        dr = DRConfidenceSequence(cfg)

        for _ in range(10):
            dr.update(True, 0.5, 0.5, 0.3, 0.5)
            lcb = dr.lcb()
            assert lcb <= dr.mean + 1e-10

    def test_lcb_with_negative_mean(self):
        """LCB can be negative if mean is small."""
        cfg = DRCSConfig()
        dr = DRConfidenceSequence(cfg)

        for _ in range(5):
            dr.update(True, 0.1, 0.1, 0.0, 0.5)

        lcb = dr.lcb()
        # LCB may be negative if radius > mean
        assert lcb <= dr.mean


class TestDREstimatorFormula:
    """Test DR estimator formula: z = (Qa-Qf) + (u-Qa)/p when served_adaptive=True."""

    def test_dr_estimator_when_adaptive_served(self):
        """When served_adaptive=True, z should equal (Qa-Qf) + (u-Qa)/p."""
        cfg = DRCSConfig(p_min=0.05)
        dr = DRConfidenceSequence(cfg)

        # Test with known values
        served_adaptive = True
        reward = 0.8
        Qa = 0.7
        Qf = 0.5
        p_used = 0.5

        dr.update(served_adaptive, reward, Qa, Qf, p_used)

        # Expected z = (0.7 - 0.5) + (0.8 - 0.7) / 0.5
        #           = 0.2 + 0.2 = 0.4
        expected_z = (Qa - Qf) + (reward - Qa) / p_used
        assert pytest.approx(dr.mean, rel=1e-10) == expected_z

    def test_dr_estimator_when_baseline_served(self):
        """When served_adaptive=False, z should equal (Qa-Qf) - (u-Qf)/(1-p)."""
        cfg = DRCSConfig(p_min=0.05)
        dr = DRConfidenceSequence(cfg)

        # Test with known values
        served_adaptive = False
        reward = 0.4
        Qa = 0.7
        Qf = 0.5
        p_used = 0.5

        dr.update(served_adaptive, reward, Qa, Qf, p_used)

        # Expected z = (0.7 - 0.5) - (0.4 - 0.5) / (1 - 0.5)
        #           = 0.2 - (-0.1) / 0.5 = 0.2 + 0.2 = 0.4
        expected_z = (Qa - Qf) - (reward - Qf) / (1.0 - p_used)
        assert pytest.approx(dr.mean, rel=1e-10) == expected_z


class TestClippingLogic:
    """Test that reward, Qa, Qf, p are clipped to valid ranges."""

    def test_reward_clipped_to_u_max(self):
        """Reward should be clipped to [0, u_max]."""
        cfg = DRCSConfig(u_max=1.0)
        dr = DRConfidenceSequence(cfg)

        # Feed reward > u_max
        dr.update(True, 2.0, 0.5, 0.3, 0.5)

        # The clipped reward should be 1.0, so z includes clipped value
        # z = (0.5 - 0.3) + (1.0 - 0.5) / 0.5 = 0.2 + 1.0 = 1.2
        expected_z = (0.5 - 0.3) + (1.0 - 0.5) / 0.5
        assert pytest.approx(dr.mean, rel=1e-10) == expected_z

    def test_reward_clipped_to_zero_floor(self):
        """Negative reward should be clipped to 0."""
        cfg = DRCSConfig(u_max=1.0)
        dr = DRConfidenceSequence(cfg)

        dr.update(True, -0.5, 0.5, 0.3, 0.5)

        expected_z = (0.5 - 0.3) + (0.0 - 0.5) / 0.5
        assert pytest.approx(dr.mean, rel=1e-10) == expected_z

    def test_qa_clipped_to_u_max(self):
        """Qa should be clipped to [0, u_max]."""
        cfg = DRCSConfig(u_max=1.0)
        dr = DRConfidenceSequence(cfg)

        dr.update(True, 0.8, 2.0, 0.3, 0.5)

        # Qa clipped to 1.0
        expected_z = (1.0 - 0.3) + (0.8 - 1.0) / 0.5
        assert pytest.approx(dr.mean, rel=1e-10) == expected_z

    def test_qf_clipped_to_u_max(self):
        """Qf should be clipped to [0, u_max]."""
        cfg = DRCSConfig(u_max=1.0)
        dr = DRConfidenceSequence(cfg)

        dr.update(True, 0.8, 0.5, 2.0, 0.5)

        # Qf clipped to 1.0
        expected_z = (0.5 - 1.0) + (0.8 - 0.5) / 0.5
        assert pytest.approx(dr.mean, rel=1e-10) == expected_z

    def test_p_clipped_to_valid_range(self):
        """p_used should be clipped to [p_min, 1 - p_min]."""
        cfg = DRCSConfig(p_min=0.05, u_max=1.0)
        dr = DRConfidenceSequence(cfg)

        # Feed p_used < p_min
        dr.update(True, 0.8, 0.5, 0.3, 0.01)

        # p should be clipped to 0.05
        expected_z = (0.5 - 0.3) + (0.8 - 0.5) / 0.05
        assert pytest.approx(dr.mean, rel=1e-10) == expected_z


class TestAcceptanceGuardrail:
    """Test acceptance guardrail: t>=5 AND acc_lcb < accept_floor => p=0."""

    def test_acceptance_guardrail_not_active_before_t5(self):
        """Guardrail should not activate before t=5."""
        cfg = SafeSwitchDRConfig(
            delta=0.01,
            p_min=0.1,
            p_max=1.0,
            accept_floor=0.5,
        )
        gate = SafeSwitchDR(cfg)

        # Feed low acceptance (which would trigger guardrail if t >= 5)
        for _ in range(4):
            gate.update(True, 0.5, 0.0, 0.5, 0.3, gate.p)

        # Even with low acceptance, p should not be 0 yet (t < 5)
        assert gate.p > 0.0

    def test_acceptance_guardrail_activates_at_t5_with_low_acceptance(self):
        """Guardrail should activate at t=5 if acc_lcb < accept_floor."""
        cfg = SafeSwitchDRConfig(
            delta=0.01,
            p_min=0.1,
            p_max=1.0,
            accept_floor=2.0,
            a_max=1.0,
        )
        gate = SafeSwitchDR(cfg)

        # Feed zero acceptance (will have low LCB)
        for _ in range(5):
            gate.update(True, 0.5, 0.0, 0.5, 0.3, gate.p)

        # acc_lcb should be < accept_floor, so p should be 0
        assert pytest.approx(gate.p, abs=1e-10) == 0.0

    def test_acceptance_guardrail_respects_acceptance_floor(self):
        """Guardrail should only activate when acc_lcb < accept_floor."""
        cfg = SafeSwitchDRConfig(
            delta=0.01,
            p_min=0.1,
            p_max=1.0,
            accept_floor=0.5,
            a_max=5.0,
        )
        gate = SafeSwitchDR(cfg)

        # Feed high acceptance — need enough observations so the confidence
        # radius shrinks and acc_lcb rises above accept_floor.
        for _ in range(100):
            gate.update(True, 0.5, 3.0, 0.5, 0.3, gate.p)

        # acc_lcb should be >= accept_floor, so guardrail should not activate
        assert gate.p > 0.0

    def test_decide_returns_false_when_guardrail_active(self):
        """decide() should return (False, 0.0) when guardrail is active."""
        cfg = SafeSwitchDRConfig(
            delta=0.01,
            p_min=0.1,
            accept_floor=2.0,
            a_max=1.0,
        )
        gate = SafeSwitchDR(cfg)

        for _ in range(5):
            gate.update(True, 0.5, 0.0, 0.5, 0.3, gate.p)

        decision, p = gate.decide()
        assert decision is False
        assert p == 0.0


class TestStepUpStepDown:
    """Test step-up/step-down behavior."""

    def test_step_up_when_dr_lcb_positive(self):
        """p should increase by step_up when dr.lcb() > 0."""
        cfg = SafeSwitchDRConfig(
            delta=0.05,
            p_min=0.1,
            p_max=1.0,
            step_up=0.05,
            step_down=0.5,
            accept_floor=0.0,
        )
        gate = SafeSwitchDR(cfg)

        initial_p = gate.p

        # Mock the DR to return positive LCB
        class MockDR:
            def __init__(self):
                self._lcb = 0.5

            def update(self, *args, **kwargs):
                pass

            def lcb(self):
                return self._lcb

            def summary(self):
                return {"t": 1, "mean": 0.5, "rad": 0.0, "lcb": 0.5}

        gate.dr = MockDR()
        gate.update(True, 0.5, 1.0, 0.5, 0.3, gate.p)

        expected_p = min(cfg.p_max, initial_p + cfg.step_up)
        assert pytest.approx(gate.p, rel=1e-10) == expected_p

    def test_step_down_when_dr_lcb_non_positive(self):
        """p should be multiplied by step_down when dr.lcb() <= 0."""
        cfg = SafeSwitchDRConfig(
            delta=0.05,
            p_min=0.1,
            p_max=1.0,
            step_up=0.05,
            step_down=0.5,
            accept_floor=0.0,
        )
        gate = SafeSwitchDR(cfg)

        initial_p = gate.p

        class MockDR:
            def __init__(self):
                self._lcb = -0.1

            def update(self, *args, **kwargs):
                pass

            def lcb(self):
                return self._lcb

            def summary(self):
                return {"t": 1, "mean": -0.1, "rad": 0.2, "lcb": -0.1}

        gate.dr = MockDR()
        gate.update(True, 0.5, 1.0, 0.5, 0.3, gate.p)

        expected_p = max(cfg.p_min, initial_p * cfg.step_down)
        assert pytest.approx(gate.p, rel=1e-10) == expected_p

    def test_step_down_respects_p_min(self):
        """p should never go below p_min after step-down."""
        cfg = SafeSwitchDRConfig(
            delta=0.05,
            p_min=0.2,
            p_max=1.0,
            step_up=0.05,
            step_down=0.1,  # Small step_down
            accept_floor=0.0,
        )
        gate = SafeSwitchDR(cfg)
        gate.p = 0.25

        class MockDR:
            def lcb(self):
                return -1.0

            def update(self, *args, **kwargs):
                pass

            def summary(self):
                return {"t": 1, "mean": -1.0, "rad": 0.5, "lcb": -1.0}

        gate.dr = MockDR()
        gate.update(True, 0.5, 1.0, 0.5, 0.3, gate.p)

        # 0.25 * 0.1 = 0.025 < p_min=0.2, so should be clamped to 0.2
        assert pytest.approx(gate.p, rel=1e-10) == cfg.p_min


class TestBoundaryBehavior:
    """Test boundary behavior: p never exceeds p_max, never below p_min (except guardrail)."""

    def test_p_clamped_to_p_max_on_step_up(self):
        """p should not exceed p_max even after step_up."""
        cfg = SafeSwitchDRConfig(
            p_min=0.05,
            p_max=0.5,
            step_up=0.2,
            accept_floor=0.0,
        )
        gate = SafeSwitchDR(cfg)
        gate.p = 0.45

        class MockDR:
            def lcb(self):
                return 1.0

            def update(self, *args, **kwargs):
                pass

            def summary(self):
                return {"t": 1, "mean": 1.0, "rad": 0.0, "lcb": 1.0}

        gate.dr = MockDR()
        gate.update(True, 0.5, 1.0, 0.5, 0.3, gate.p)

        # 0.45 + 0.2 = 0.65 > p_max=0.5, should be clamped
        assert pytest.approx(gate.p, rel=1e-10) == cfg.p_max

    def test_p_clamped_to_p_min_on_step_down(self):
        """p should not go below p_min after step_down (unless guardrail)."""
        cfg = SafeSwitchDRConfig(
            p_min=0.1,
            p_max=1.0,
            step_up=0.05,
            step_down=0.5,
            accept_floor=0.0,
        )
        gate = SafeSwitchDR(cfg)
        gate.p = 0.15

        class MockDR:
            def lcb(self):
                return -1.0

            def update(self, *args, **kwargs):
                pass

            def summary(self):
                return {"t": 1, "mean": -1.0, "rad": 0.5, "lcb": -1.0}

        gate.dr = MockDR()
        gate.update(True, 0.5, 1.0, 0.5, 0.3, gate.p)

        # 0.15 * 0.5 = 0.075 < p_min=0.1, should be clamped
        assert pytest.approx(gate.p, rel=1e-10) == cfg.p_min

    def test_guardrail_can_set_p_to_zero(self):
        """Only guardrail can set p to 0; normal step-down cannot."""
        cfg = SafeSwitchDRConfig(
            p_min=0.05,
            p_max=1.0,
            step_down=0.5,
            accept_floor=2.0,
            a_max=1.0,
        )
        gate = SafeSwitchDR(cfg)

        # Feed low acceptance at t >= 5
        for _ in range(5):
            gate.update(True, 0.5, 0.0, 0.5, 0.3, gate.p)

        # Guardrail should set p to 0
        assert gate.p == 0.0

    def test_p_initialization_respects_p_min(self):
        """p should be initialized to p_min."""
        cfg = SafeSwitchDRConfig(p_min=0.15, p_max=1.0)
        gate = SafeSwitchDR(cfg)

        assert pytest.approx(gate.p, rel=1e-10) == cfg.p_min


class TestAcceptanceLCBFormula:
    """Test acceptance LCB formula in SafeSwitchDR."""

    def test_acc_lcb_formula(self):
        """Test that _acc_lcb computes correctly using Welford stats."""
        cfg = SafeSwitchDRConfig(
            delta=0.01,
            a_max=5.0,
        )
        gate = SafeSwitchDR(cfg)

        # Feed constant acceptance = 2.0
        for _ in range(5):
            gate.update(True, 0.5, 2.0, 0.5, 0.3, gate.p)

        # Manually compute expected acc_lcb
        n = gate.t
        var_hat = max(0.0, gate.acc_M2 / max(n - 1, 1))
        delta_t = gate._delta_t()
        log = math.log(2.0 / max(delta_t, 1e-12))
        rad = math.sqrt(2.0 * var_hat * log / n) + (2.0 * cfg.a_max * log) / (3.0 * n)
        expected_lcb = gate.acc_mean - rad

        actual_lcb = gate._acc_lcb()
        assert pytest.approx(actual_lcb, rel=1e-10) == expected_lcb

    def test_acc_lcb_decreases_with_constant_acceptance(self):
        """With constant acceptance, acc_lcb should increase toward the mean."""
        cfg = SafeSwitchDRConfig(delta=0.01, a_max=5.0)
        gate = SafeSwitchDR(cfg)

        lcbs = []
        for _ in range(15):
            gate.update(True, 0.5, 2.0, 0.5, 0.3, gate.p)
            lcbs.append(gate._acc_lcb())

        # acc_lcb should increase (confidence radius shrinks)
        assert lcbs[-1] > lcbs[0]


class TestSafeSwitchTelemetry:
    """Test that telemetry returns all expected keys."""

    def test_telemetry_includes_all_keys(self):
        """telemetry() should include t, mean, rad, lcb, p, acc_lcb."""
        cfg = SafeSwitchDRConfig()
        gate = SafeSwitchDR(cfg)

        for _ in range(3):
            gate.update(True, 0.5, 0.5, 0.3, 0.5, gate.p)

        tel = gate.telemetry()

        assert "t" in tel
        assert "mean" in tel
        assert "rad" in tel
        assert "lcb" in tel
        assert "p" in tel
        assert "acc_lcb" in tel


class TestIntegrationScenarios:
    """Integration tests combining multiple correctness properties."""

    def test_full_scenario_positive_uplift(self):
        """Full scenario: positive uplift should increase p over time.

        Use accept_floor=0 to disable the acceptance guardrail so that
        positive DR uplift can drive p upward.
        """
        cfg = SafeSwitchDRConfig(
            delta=0.05,
            p_min=0.1,
            p_max=1.0,
            step_up=0.1,
            step_down=0.5,
            accept_floor=0.0,
        )
        gate = SafeSwitchDR(cfg)

        p_values = [gate.p]
        for _ in range(100):
            gate.update(True, 0.9, 0.8, 0.5, 0.3, gate.p)
            p_values.append(gate.p)

        # p should generally increase (though not strictly monotonic)
        assert p_values[-1] >= p_values[0]

    def test_full_scenario_negative_uplift(self):
        """Full scenario: negative uplift should decrease p over time."""
        cfg = SafeSwitchDRConfig(
            delta=0.05,
            p_min=0.05,
            p_max=1.0,
            step_up=0.1,
            step_down=0.3,
            accept_floor=0.5,
        )
        gate = SafeSwitchDR(cfg)

        p_values = [gate.p]
        for _ in range(10):
            gate.update(True, 0.3, 0.5, 0.7, 0.3, gate.p)
            p_values.append(gate.p)

        # p should generally decrease
        assert p_values[-1] <= p_values[0]
