"""Differential-privacy accountants shared across Orchid Ranker components."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from orchid_ranker.agents.simple_dp import SimpleDPAccountant, SimpleDPConfig

try:  # Optional dependency
    from opacus.accountants.analysis import rdp as opacus_rdp
except ImportError:  # pragma: no cover - handled at runtime
    opacus_rdp = None


class _NullAccountant:
    """No-op accountant used when DP is disabled."""

    def step(self, steps: int) -> Tuple[float, float]:
        return 0.0, 0.0

    def fork(self):
        return self


class SimpleAccountantAdapter:
    """Adapter adding fork capability to the legacy SimpleDPAccountant."""

    def __init__(self, cfg: SimpleDPConfig) -> None:
        self._acc = SimpleDPAccountant(q=cfg.sample_rate, sigma=cfg.noise_multiplier, delta=cfg.delta)

    @property
    def _state(self) -> Tuple[int, float]:
        return self._acc.T, self._acc.eps

    @_state.setter
    def _state(self, value: Tuple[int, float]) -> None:
        self._acc.T, self._acc.eps = value

    def step(self, steps: int) -> Tuple[float, float]:
        return self._acc.step(steps)

    def fork(self):
        clone = SimpleAccountantAdapter(SimpleDPConfig(
            enabled=True,
            noise_multiplier=self._acc.sigma,
            max_grad_norm=0.0,
            sample_rate=self._acc.q,
            delta=self._acc.delta,
        ))
        clone._state = self._state
        return clone


class OpacusAccountant:
    """Wrapper around Opacus RDP analysis utilities."""

    def __init__(
        self,
        *,
        sample_rate: float,
        noise_multiplier: float,
        delta: float,
        orders: Optional[Sequence[float]] = None,
    ) -> None:
        if opacus_rdp is None:  # pragma: no cover - import guard
            raise ImportError(
                "opacus is required for OpacusAccountant. Install via `pip install opacus` "
                "or select a different DP engine."
            )
        if not 0.0 < float(delta) < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.sample_rate = float(max(0.0, sample_rate))
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.orders: Tuple[float, ...] = tuple(orders or (1.25, 1.5, 2, 3, 5, 8, 10, 16, 32, 64, 128, 256))
        self.steps = 0
        self._rdp_cache = [0.0 for _ in self.orders]
        self._eps = 0.0

    def _compute_rdp(self, steps: int):
        return opacus_rdp.compute_rdp(
            q=self.sample_rate,
            noise_multiplier=self.noise_multiplier,
            steps=max(0, steps),
            orders=self.orders,
        )

    def _eps_from_rdp(self, rdp_values: Sequence[float]) -> float:
        eps, _ = opacus_rdp.get_privacy_spent(
            orders=self.orders,
            rdp=rdp_values,
            delta=self.delta,
        )
        return float(max(0.0, eps))

    def step(self, steps: int) -> Tuple[float, float]:
        steps = int(max(0, steps))
        if steps == 0 or self.sample_rate <= 0.0 or self.noise_multiplier <= 0.0:
            return 0.0, float(self._eps)

        prev_rdp = self._rdp_cache
        self.steps += steps
        curr_rdp = self._compute_rdp(self.steps)
        self._rdp_cache = list(curr_rdp)

        eps_prev = self._eps_from_rdp(prev_rdp)
        eps_curr = self._eps_from_rdp(curr_rdp)
        self._eps = eps_curr
        return float(max(0.0, eps_curr - eps_prev)), float(eps_curr)

    def fork(self):
        clone = OpacusAccountant(
            sample_rate=self.sample_rate,
            noise_multiplier=self.noise_multiplier,
            delta=self.delta,
            orders=self.orders,
        )
        clone.steps = self.steps
        clone._rdp_cache = list(self._rdp_cache)
        clone._eps = self._eps
        return clone


def build_accountant(engine: str, cfg: SimpleDPConfig):
    """Return an accountant instance for the requested engine."""

    if not cfg.enabled:
        return _NullAccountant()

    engine = engine.lower()
    if engine == "per_sample":
        return SimpleAccountantAdapter(cfg)
    if engine == "opacus":
        return OpacusAccountant(
            sample_rate=cfg.sample_rate,
            noise_multiplier=cfg.noise_multiplier,
            delta=cfg.delta,
        )
    supported = {"per_sample", "opacus"}
    raise ValueError(
        f"Unknown DP engine '{engine}'. Supported engines: {supported}. "
        "A typo here silently disables privacy tracking."
    )


__all__ = ["build_accountant", "OpacusAccountant", "SimpleAccountantAdapter"]
