from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from .dr_cs import DRConfidenceSequence, DRCSConfig


@dataclass
class SafeSwitchDRConfig:
    delta: float = 0.01
    p_min: float = 0.05
    p_max: float = 1.0
    step_up: float = 0.05
    step_down: float = 0.5
    u_max: float = 1.0
    a_max: float = 6.0
    accept_floor: float = 2.0


class SafeSwitchDR:
    """
    Mixture gate between a fixed teacher and adaptive student using a DR uplift
    confidence sequence + acceptance guardrail.
    """

    def __init__(self, cfg: SafeSwitchDRConfig):
        self.cfg = cfg
        self.p = cfg.p_min
        self.t = 0
        self.dr = DRConfidenceSequence(
            DRCSConfig(delta=cfg.delta, u_max=cfg.u_max, p_min=cfg.p_min)
        )
        self.acc_mean = 0.0
        self.acc_M2 = 0.0
        self._last_decision = (True, cfg.p_min)

    def _delta_t(self) -> float:
        if self.t <= 1:
            return self.cfg.delta / 2.0
        return 6.0 * self.cfg.delta / (math.pi**2 * (self.t**2))

    def _acc_lcb(self) -> float:
        n = max(self.t, 1)
        var_hat = max(0.0, self.acc_M2 / max(n - 1, 1))
        log = math.log(2.0 / max(self._delta_t(), 1e-12))
        rad = math.sqrt(2.0 * var_hat * log / n) + (2.0 * self.cfg.a_max * log) / (
            3.0 * n
        )
        return self.acc_mean - rad

    def decide(self) -> tuple[bool, float]:
        if self.t >= 5 and self._acc_lcb() < self.cfg.accept_floor:
            self.p = 0.0
            self._last_decision = (False, 0.0)
            return self._last_decision
        use_adaptive = random.random() < self.p
        self._last_decision = (use_adaptive, self.p)
        return self._last_decision

    def update(
        self,
        served_adaptive: bool,
        reward: float,
        accepts_per_user: float,
        Qa_pred: float,
        Qf_pred: float,
        p_used: float,
    ) -> None:
        self.t += 1
        acc = max(0.0, min(self.cfg.a_max, float(accepts_per_user)))
        prev = self.acc_mean
        self.acc_mean += (acc - self.acc_mean) / self.t
        self.acc_M2 += (acc - self.acc_mean) * (acc - prev)

        self.dr.update(served_adaptive, reward, Qa_pred, Qf_pred, p_used)

        if self.t >= 5 and self._acc_lcb() < self.cfg.accept_floor:
            self.p = 0.0
            return

        if self.dr.lcb() > 0.0:
            self.p = min(self.cfg.p_max, self.p + self.cfg.step_up)
        else:
            self.p = max(self.cfg.p_min, self.p * self.cfg.step_down)

    def telemetry(self) -> dict:
        info = self.dr.summary()
        info.update({"p": self.p, "acc_lcb": self._acc_lcb()})
        return info

