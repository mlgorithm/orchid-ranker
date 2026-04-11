"""Exploration policies for contextual bandits and Thompson sampling."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


# Debug logging helper
def _d(*args) -> None:
    """Debug logging (respects ORCHID_DEBUG_REC env var)."""
    import os
    if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
        logger.debug("%s", " ".join(str(a) for a in args))


# Global debug flag (imported from parent module)
_DEBUG_REC = False


class LinUCBPolicy:
    """Linear contextual bandit with UCB confidence bounds for arm selection.

    Parameters
    ----------
    d : int
        Feature dimension.
    alpha : float, optional
        Exploration bonus multiplier (default: 1.0).
    l2 : float, optional
        L2 regularization strength (default: 1.0).
    """
    def __init__(self, d: int, alpha: float = 1.0, l2: float = 1.0):
        self.d = int(d)
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.A: Dict[int, np.ndarray] = {}  # arm -> (d,d)
        self.b: Dict[int, np.ndarray] = {}  # arm -> (d,)

    def _ensure(self, arm: int) -> None:
        if arm not in self.A:
            self.A[arm] = self.l2 * np.eye(self.d, dtype=np.float64)
            self.b[arm] = np.zeros(self.d, dtype=np.float64)

    def score(self, x: np.ndarray, base: float = 0.0, i: int = 0) -> float:
        arm = int(i)
        self._ensure(arm)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        A = self.A[arm]
        b = self.b[arm]
        I = np.eye(self.d, dtype=np.float64)

        jitter = 1e-8
        for _ in range(3):
            try:
                A_reg = A + (self.l2 + jitter) * I
                theta = np.linalg.solve(A_reg, b)
                Ax = np.linalg.solve(A_reg, x)
                mean = float(x @ theta)
                conf = float(np.sqrt(max(0.0, x @ Ax)))
                val = float(base) + mean + self.alpha * conf
                if _DEBUG_REC:
                    _d(f"LinUCB score arm={arm} mean={mean:.4f} conf={conf:.4f} -> {val:.4f}")
                return val
            except np.linalg.LinAlgError:
                jitter *= 10.0

        A_pinv = np.linalg.pinv(A + self.l2 * I)
        theta = A_pinv @ b
        mean = float(x @ theta)
        conf = float(np.sqrt(max(0.0, x @ (A_pinv @ x))))
        val = float(base) + mean + self.alpha * conf
        if _DEBUG_REC:
            _d(f"LinUCB score(pinvt) arm={arm} mean={mean:.4f} conf={conf:.4f} -> {val:.4f}")
        return val

    def update(self, i: int, x: np.ndarray, r: float):
        arm = int(i)
        self._ensure(arm)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(r) * x
        if _DEBUG_REC:
            _d(f"LinUCB update arm={arm} r={r:.3f}")


class BootTS:
    """
    Lightweight bootstrapped Thompson Sampling over a linear head.
    Works on interaction feature phi = [u, I, u⊙I] (dim = 3*emb_dim).
    """
    def __init__(self, d: int, heads: int = 10, l2: float = 1.0, rng: int = 42):
        self.d, self.H, self.l2 = int(d), int(heads), float(l2)
        self.rng = np.random.RandomState(rng)
        self.As = [self.l2 * np.eye(self.d, dtype=np.float64) for _ in range(self.H)]
        self.bs = [np.zeros(self.d, dtype=np.float64) for _ in range(self.H)]
        self._thetas_dirty = [True] * self.H  # lazy theta recomputation
        self._thetas = [np.zeros(self.d, dtype=np.float64) for _ in range(self.H)]
        _d(f"BootTS d={d} heads={heads} l2={l2}")

    def _get_theta(self, h: int) -> np.ndarray:
        """Get theta for head h, recomputing only if dirty."""
        if self._thetas_dirty[h]:
            self._thetas[h] = np.linalg.solve(self.As[h] + 1e-8 * np.eye(self.d), self.bs[h])
            self._thetas_dirty[h] = False
        return self._thetas[h]

    def score_vec(self, x: np.ndarray) -> float:
        h = int(self.rng.randint(self.H))
        theta = self._get_theta(h)
        val = float(x @ theta + self.rng.normal(0.0, 0.01))
        if _DEBUG_REC:
            _d(f"BootTS score head={h} -> {val:.4f}")
        return val

    def update(self, x: np.ndarray, r: float, k: int = 2):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        heads = self.rng.choice(self.H, size=min(k, self.H), replace=False)
        for h in heads:
            self.As[h] += np.outer(x, x)
            self.bs[h] += float(r) * x
            self._thetas_dirty[h] = True  # invalidate cached theta
        if _DEBUG_REC:
            _d(f"BootTS update heads={list(map(int,heads))} r={r:.3f}")


__all__ = [
    "LinUCBPolicy",
    "BootTS",
]
