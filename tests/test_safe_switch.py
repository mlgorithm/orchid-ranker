import math

import pytest

from orchid_ranker.safety import SafeSwitchDR, SafeSwitchDRConfig


def _simulate_rounds(gate, n, *, reward, accepts, qa, qf):
    for _ in range(n):
        decision, p = gate.decide()
        gate.update(decision, reward, accepts, qa, qf, p)


def test_safe_switch_acceptance_floor_shuts_off():
    cfg = SafeSwitchDRConfig(
        delta=0.05,
        p_min=0.2,
        step_up=0.2,
        step_down=0.5,
        u_max=1.0,
        a_max=1.0,
        accept_floor=0.6,
    )
    gate = SafeSwitchDR(cfg)

    _simulate_rounds(gate, 6, reward=0.2, accepts=0.0, qa=0.2, qf=0.2)

    decision, p = gate.decide()
    assert decision is False
    assert math.isclose(p, 0.0)
    assert gate.telemetry()["acc_lcb"] < cfg.accept_floor


def test_safe_switch_increases_probability_on_positive_uplift():
    cfg = SafeSwitchDRConfig(
        delta=0.05,
        p_min=0.1,
        step_up=0.2,
        step_down=0.5,
        u_max=1.0,
        a_max=1.0,
        accept_floor=0.0,
    )
    gate = SafeSwitchDR(cfg)

    initial_p = gate.p
    assert math.isclose(initial_p, cfg.p_min)

    class _DeterministicDR:
        def __init__(self):
            self._lcb = -1.0

        def update(self, *args, **kwargs):
            self._lcb = 0.5

        def lcb(self):
            return self._lcb

        def summary(self):
            return {"lcb": self._lcb}

    gate.dr = _DeterministicDR()

    gate.update(True, 1.0, 1.0, 0.95, 0.1, gate.p)
    assert gate.p > initial_p
