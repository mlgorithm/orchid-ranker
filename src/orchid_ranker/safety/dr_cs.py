from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DRCSConfig:
    delta: float = 0.01
    u_max: float = 1.0
    p_min: float = 0.05


class DRConfidenceSequence:
    """
    Doubly-Robust uplift estimator with an empirical-Bernstein/Freedman-style
    confidence sequence. Tracks uplift between an adaptive and baseline policy
    under a logging mixture probability p_t.
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
        if self.t <= 1:
            return self.cfg.delta / 2.0
        return 6.0 * self.cfg.delta / (math.pi**2 * (self.t**2))

    def _radius(self) -> float:
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
        self.t += 1
        u = max(0.0, min(self.cfg.u_max, float(reward)))
        Qa = max(0.0, min(self.cfg.u_max, float(Qa)))
        Qf = max(0.0, min(self.cfg.u_max, float(Qf)))
        p = min(max(float(p_used), self.cfg.p_min), 1.0 - self.cfg.p_min)

        if served_adaptive:
            z = (Qa - Qf) + (u - Qa) / p
        else:
            z = (Qa - Qf) - (u - Qf) / (1.0 - p)

        prev_mean = self.mean
        self.mean += (z - self.mean) / self.t
        self.M2 += (z - self.mean) * (z - prev_mean)

    def lcb(self) -> float:
        return self.mean - self._radius()

    def summary(self) -> dict:
        return {
            "t": self.t,
            "mean": self.mean,
            "rad": self._radius(),
            "lcb": self.lcb(),
        }

