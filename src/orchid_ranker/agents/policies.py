"""Exploration policies for contextual bandits and Thompson sampling."""

from __future__ import annotations

import logging
from typing import Dict, Optional

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
        if self.d <= 0:
            raise ValueError("d must be positive")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not np.isfinite(self.l2) or self.l2 <= 0.0:
            raise ValueError("l2 must be finite and positive")
        self.A: Dict[int, np.ndarray] = {}  # arm -> (d,d)
        self.b: Dict[int, np.ndarray] = {}  # arm -> (d,)

    def _ensure(self, arm: int) -> None:
        if arm not in self.A:
            self.A[arm] = self.l2 * np.eye(self.d, dtype=np.float64)
            self.b[arm] = np.zeros(self.d, dtype=np.float64)

    def _feature(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.shape != (self.d,):
            raise ValueError(f"x must have shape ({self.d},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x must contain only finite values")
        return arr

    def score(self, x: np.ndarray, base: float = 0.0, i: int = 0) -> float:
        arm = int(i)
        self._ensure(arm)
        x = self._feature(x)
        base_value = float(base)
        if not np.isfinite(base_value):
            raise ValueError("base must be finite")
        A = self.A[arm]
        b = self.b[arm]
        jitter = 1e-8
        for _ in range(3):
            try:
                A_reg = A + jitter * np.eye(self.d, dtype=np.float64)
                theta = np.linalg.solve(A_reg, b)
                Ax = np.linalg.solve(A_reg, x)
                mean = float(x @ theta)
                conf = float(np.sqrt(max(0.0, x @ Ax)))
                val = base_value + mean + self.alpha * conf
                if _DEBUG_REC:
                    _d(f"LinUCB score arm={arm} mean={mean:.4f} conf={conf:.4f} -> {val:.4f}")
                return val
            except np.linalg.LinAlgError:
                jitter *= 10.0

        A_pinv = np.linalg.pinv(A)
        theta = A_pinv @ b
        mean = float(x @ theta)
        conf = float(np.sqrt(max(0.0, x @ (A_pinv @ x))))
        val = base_value + mean + self.alpha * conf
        if _DEBUG_REC:
            _d(f"LinUCB score(pinvt) arm={arm} mean={mean:.4f} conf={conf:.4f} -> {val:.4f}")
        return val

    def update(self, i: int, x: np.ndarray, r: float):
        arm = int(i)
        self._ensure(arm)
        x = self._feature(x)
        reward = float(r)
        if not np.isfinite(reward):
            raise ValueError("r must be finite")
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        if _DEBUG_REC:
            _d(f"LinUCB update arm={arm} r={r:.3f}")


class BootTS:
    """
    Lightweight bootstrapped Thompson Sampling over a linear head.
    Works on interaction feature phi = [u, I, u⊙I] (dim = 3*emb_dim).
    """
    def __init__(self, d: int, heads: int = 10, l2: float = 1.0, rng: int = 42):
        self.d, self.H, self.l2 = int(d), int(heads), float(l2)
        if self.d <= 0:
            raise ValueError("d must be positive")
        if self.H <= 0:
            raise ValueError("heads must be positive")
        if not np.isfinite(self.l2) or self.l2 <= 0.0:
            raise ValueError("l2 must be finite and positive")
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

    def _feature(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.shape != (self.d,):
            raise ValueError(f"x must have shape ({self.d},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x must contain only finite values")
        return arr

    def sample_head(self) -> int:
        """Sample one bootstrap head for a coherent slate decision."""
        return int(self.rng.randint(self.H))

    def score_vec(self, x: np.ndarray, *, head: Optional[int] = None, add_noise: bool = True) -> float:
        h = self.sample_head() if head is None else int(head)
        if h < 0 or h >= self.H:
            raise ValueError(f"head must be in [0, {self.H}), got {head}")
        x = self._feature(x)
        theta = self._get_theta(h)
        noise = float(self.rng.normal(0.0, 0.01)) if add_noise else 0.0
        val = float(x @ theta + noise)
        if _DEBUG_REC:
            _d(f"BootTS score head={h} -> {val:.4f}")
        return val

    def score_many(self, xs: np.ndarray, *, head: Optional[int] = None, add_noise: bool = True) -> np.ndarray:
        h = self.sample_head() if head is None else int(head)
        features = np.asarray(xs, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.d:
            raise ValueError(f"xs must have shape (n, {self.d}), got {features.shape}")
        if not np.all(np.isfinite(features)):
            raise ValueError("xs must contain only finite values")
        theta = self._get_theta(h)
        scores = features @ theta
        if add_noise:
            scores = scores + self.rng.normal(0.0, 0.01, size=features.shape[0])
        return np.asarray(scores, dtype=np.float64)

    def update(self, x: np.ndarray, r: float, k: int = 2):
        x = self._feature(x)
        reward = float(r)
        if not np.isfinite(reward):
            raise ValueError("r must be finite")
        heads = self.rng.choice(self.H, size=min(k, self.H), replace=False)
        for h in heads:
            self.As[h] += np.outer(x, x)
            self.bs[h] += reward * x
            self._thetas_dirty[h] = True  # invalidate cached theta
        if _DEBUG_REC:
            _d(f"BootTS update heads={list(map(int,heads))} r={r:.3f}")


__all__ = [
    "LinUCBPolicy",
    "BootTS",
]
