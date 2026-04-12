from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DRCSConfig:
    """Configuration for doubly-robust confidence sequence.

    Attributes
    ----------
    delta : float
        Failure probability (default: 0.01).
    u_max : float
        Maximum reward value (default: 1.0).
    p_min : float
        Minimum logging probability (default: 0.05).
    """

    delta: float = 0.01
    u_max: float = 1.0
    p_min: float = 0.05


class DRConfidenceSequence:
    """Doubly-Robust (DR) uplift estimator with empirical-Bernstein confidence sequence.

    Tracks uplift between an adaptive and baseline policy using doubly-robust
    estimation under logging mixture probability p_t. Maintains a confidence lower
    bound (LCB) for safe deployment.

    Parameters
    ----------
    cfg : DRCSConfig
        Configuration object with delta, u_max, p_min parameters.
    """

    def __init__(self, cfg: DRCSConfig):
        self.cfg = cfg
        self.t = 0
        self.mean = 0.0
        self.M2 = 0.0
        # worst-case one-step range bound
        self.B = (
            (2.0 * cfg.u_max)
            + (cfg.u_max / cfg.p_min)
            + (cfg.u_max / (1.0 - cfg.p_min))
        )

    def _delta_t(self) -> float:
        """Compute anytime-valid failure probability for current time step."""
        if self.t <= 1:
            return self.cfg.delta / 2.0
        return 6.0 * self.cfg.delta / (math.pi**2 * (self.t**2))

    def _radius(self) -> float:
        """Compute confidence radius from empirical variance and failure probability."""
        n = max(self.t, 1)
        var_hat = max(0.0, self.M2 / max(n - 1, 1))
        log = math.log(2.0 / max(self._delta_t(), 1e-12))
        return math.sqrt(2.0 * var_hat * log / n) + (2.0 * self.B * log) / (3.0 * n)

    def update(
        self,
        served_adaptive: bool,
        reward: float,
        Qa: float,
        Qf: float,
        p_used: float,
    ) -> None:
        """Update estimator with observed outcome.

        Parameters
        ----------
        served_adaptive : bool
            Whether the adaptive policy was deployed in this round.
        reward : float
            Observed reward.
        Qa : float
            Predicted Q-value (value function) for adaptive policy.
        Qf : float
            Predicted Q-value for baseline policy.
        p_used : float
            Logging probability mixture (proportion of adaptive deployment).
        """
        self.t += 1
        u = max(0.0, min(self.cfg.u_max, float(reward)))
        Qa = max(0.0, min(self.cfg.u_max, float(Qa)))
        Qf = max(0.0, min(self.cfg.u_max, float(Qf)))
        p = float(p_used)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p_used must be in [0, 1], got {p_used}")
        if served_adaptive and p == 0.0:
            raise ValueError("served_adaptive=True is impossible when p_used=0.0")
        if (not served_adaptive) and p == 1.0:
            raise ValueError("served_adaptive=False is impossible when p_used=1.0")

        if served_adaptive:
            if p == 1.0:
                z = u - Qf
            else:
                z = (Qa - Qf) + (u - Qa) / p
        else:
            if p == 0.0:
                z = Qa - u
            else:
                z = (Qa - Qf) - (u - Qf) / (1.0 - p)

        prev_mean = self.mean
        self.mean += (z - self.mean) / self.t
        self.M2 += (z - self.mean) * (z - prev_mean)

    def lcb(self) -> float:
        """Compute lower confidence bound (LCB) on uplift.

        Returns
        -------
        float
            Confidence lower bound on mean uplift.
        """
        return self.mean - self._radius()

    def summary(self) -> dict:
        """Get summary statistics.

        Returns
        -------
        dict
            Keys: t (count), mean (uplift estimate), rad (radius), lcb (lower bound).
        """
        return {
            "t": self.t,
            "mean": self.mean,
            "rad": self._radius(),
            "lcb": self.lcb(),
        }


__all__ = [
    "DRCSConfig",
    "DRConfidenceSequence",
]
