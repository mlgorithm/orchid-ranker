from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .dr_cs import DRConfidenceSequence, DRCSConfig


@dataclass
class SafeSwitchDRConfig:
    """Configuration for safe policy switching with doubly-robust estimation.

    Attributes
    ----------
    delta : float
        Failure probability (default: 0.01).
    p_min : float
        Minimum deployment probability (default: 0.05).
    p_max : float
        Maximum deployment probability (default: 1.0).
    step_up : float
        Increase in p per positive signal (default: 0.05).
    step_down : float
        Multiplicative decrease in p per negative signal (default: 0.5).
    u_max : float
        Maximum reward value (default: 1.0).
    a_max : float
        Maximum acceptance rate (default: 6.0).
    accept_floor : float
        Minimum acceptance rate LCB threshold (default: 2.0).
    """

    delta: float = 0.01
    p_min: float = 0.05
    p_max: float = 1.0
    step_up: float = 0.05
    step_down: float = 0.5
    u_max: float = 1.0
    a_max: float = 6.0
    accept_floor: float = 2.0


class SafeSwitchDR:
    """Adaptive mixture gate between baseline and adaptive policies using DR confidence sequences.

    Maintains two confidence sequences: one for reward uplift (DR-based) and one for
    acceptance rate. Automatically adjusts the deployment probability based on LCBs,
    with a safety guardrail that stops adaptive deployment if acceptance drops too low.

    Parameters
    ----------
    cfg : SafeSwitchDRConfig
        Configuration object.
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
        """Compute anytime-valid failure probability for current time step."""
        if self.t <= 1:
            return self.cfg.delta / 2.0
        return 6.0 * self.cfg.delta / (math.pi**2 * (self.t**2))

    def _acc_lcb(self) -> float:
        """Compute lower confidence bound on acceptance rate."""
        n = max(self.t, 1)
        var_hat = max(0.0, self.acc_M2 / max(n - 1, 1))
        log = math.log(2.0 / max(self._delta_t(), 1e-12))
        rad = math.sqrt(2.0 * var_hat * log / n) + (2.0 * self.cfg.a_max * log) / (
            3.0 * n
        )
        return self.acc_mean - rad

    def decide(self) -> tuple[bool, float]:
        """Make a deployment decision for this round.

        Checks acceptance guardrail and samples from Bernoulli(p).

        Returns
        -------
        tuple
            (use_adaptive: bool, p_used: float)
        """
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
        """Update both confidence sequences and adjust deployment probability.

        Parameters
        ----------
        served_adaptive : bool
            Whether adaptive policy was deployed.
        reward : float
            Observed reward.
        accepts_per_user : float
            Number of items accepted by user.
        Qa_pred : float
            Predicted value for adaptive policy.
        Qf_pred : float
            Predicted value for baseline policy.
        p_used : float
            Logging probability used in this round.
        """
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
        """Get diagnostic telemetry.

        Returns
        -------
        dict
            Keys: t, mean (uplift), rad (radius), lcb (DR LCB), p (current deployment prob),
            acc_lcb (acceptance LCB).
        """
        info = self.dr.summary()
        info.update({"p": self.p, "acc_lcb": self._acc_lcb()})
        return info


__all__ = [
    "SafeSwitchDRConfig",
    "SafeSwitchDR",
]

