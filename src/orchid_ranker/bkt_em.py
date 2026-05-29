"""Expectation-maximization fitting for Bayesian Knowledge Tracing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .knowledge_tracing import BayesianKnowledgeTracing

__all__ = [
    "BKTFitReport",
    "fit_bkt_em",
]


@dataclass(frozen=True)
class BKTFitReport:
    """Fitted BKT parameters and likelihood diagnostics."""

    p_init: float
    p_transit: float
    p_slip: float
    p_guess: float
    log_likelihood: float
    iterations: int
    n_sequences: int
    n_observations: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_tracer(self, *, mastery_threshold: float = 0.95) -> BayesianKnowledgeTracing:
        return BayesianKnowledgeTracing(
            p_init=self.p_init,
            p_transit=self.p_transit,
            p_slip=self.p_slip,
            p_guess=self.p_guess,
            mastery_threshold=mastery_threshold,
        )


def fit_bkt_em(
    sequences: Sequence[Sequence[int | bool | float]],
    *,
    p_init: float = 0.2,
    p_transit: float = 0.15,
    p_slip: float = 0.1,
    p_guess: float = 0.2,
    max_iter: int = 50,
    tol: float = 1e-5,
    eps: float = 1e-4,
) -> BKTFitReport:
    """Fit BKT parameters with a two-state hidden Markov EM routine."""
    observations = [_coerce_sequence(seq) for seq in sequences if len(seq) > 0]
    if not observations:
        raise ValueError("fit_bkt_em requires at least one non-empty sequence")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    params = np.asarray([p_init, p_transit, p_slip, p_guess], dtype=float)
    if np.any(~np.isfinite(params)) or np.any(params <= 0.0) or np.any(params >= 1.0):
        raise ValueError("initial BKT parameters must be finite and strictly inside (0, 1)")

    previous_ll = -np.inf
    iterations = 0
    total_obs = int(sum(len(seq) for seq in observations))
    for iteration in range(1, int(max_iter) + 1):
        p_init, p_transit, p_slip, p_guess = [float(value) for value in params]
        init_known = 0.0
        trans_num = 0.0
        trans_den = 0.0
        guess_num = 0.0
        guess_den = 0.0
        slip_num = 0.0
        slip_den = 0.0
        total_ll = 0.0

        for obs in observations:
            gamma, xi, log_likelihood = _forward_backward(obs, p_init, p_transit, p_slip, p_guess)
            total_ll += log_likelihood
            init_known += float(gamma[0, 1])
            if len(obs) > 1:
                trans_num += float(np.sum(xi[:, 0, 1]))
                trans_den += float(np.sum(gamma[:-1, 0]))
            unknown_weight = gamma[:, 0]
            known_weight = gamma[:, 1]
            guess_num += float(np.sum(unknown_weight * obs))
            guess_den += float(np.sum(unknown_weight))
            slip_num += float(np.sum(known_weight * (1.0 - obs)))
            slip_den += float(np.sum(known_weight))

        new_params = np.asarray(
            [
                _clip(init_known / len(observations), eps),
                _clip(trans_num / max(trans_den, eps), eps),
                _clip(slip_num / max(slip_den, eps), eps),
                _clip(guess_num / max(guess_den, eps), eps),
            ],
            dtype=float,
        )
        iterations = iteration
        params = new_params
        if abs(total_ll - previous_ll) <= tol:
            previous_ll = total_ll
            break
        previous_ll = total_ll

    p_init, p_transit, p_slip, p_guess = [float(value) for value in params]
    final_ll = sum(_forward_backward(obs, p_init, p_transit, p_slip, p_guess)[2] for obs in observations)
    return BKTFitReport(
        p_init=float(params[0]),
        p_transit=float(params[1]),
        p_slip=float(params[2]),
        p_guess=float(params[3]),
        log_likelihood=float(final_ll),
        iterations=iterations,
        n_sequences=len(observations),
        n_observations=total_obs,
    )


def _forward_backward(
    obs: np.ndarray,
    p_init: float,
    p_transit: float,
    p_slip: float,
    p_guess: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    transition = np.asarray(
        [
            [1.0 - p_transit, p_transit],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    emission = np.column_stack(
        [
            np.where(obs > 0.5, p_guess, 1.0 - p_guess),
            np.where(obs > 0.5, 1.0 - p_slip, p_slip),
        ]
    )
    n = len(obs)
    alpha = np.zeros((n, 2), dtype=float)
    scales = np.zeros((n,), dtype=float)
    alpha[0] = np.asarray([1.0 - p_init, p_init]) * emission[0]
    scales[0] = max(float(np.sum(alpha[0])), 1e-12)
    alpha[0] /= scales[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ transition) * emission[t]
        scales[t] = max(float(np.sum(alpha[t])), 1e-12)
        alpha[t] /= scales[t]

    beta = np.ones((n, 2), dtype=float)
    for t in range(n - 2, -1, -1):
        beta[t] = transition @ (emission[t + 1] * beta[t + 1])
        beta[t] /= scales[t + 1]

    gamma = alpha * beta
    gamma /= np.sum(gamma, axis=1, keepdims=True)
    xi = np.zeros((max(0, n - 1), 2, 2), dtype=float)
    for t in range(n - 1):
        numerator = alpha[t, :, None] * transition * (emission[t + 1] * beta[t + 1])[None, :]
        denom = max(float(np.sum(numerator)), 1e-12)
        xi[t] = numerator / denom
    return gamma, xi, float(np.sum(np.log(scales)))


def _coerce_sequence(seq: Sequence[int | bool | float]) -> np.ndarray:
    values = np.asarray([float(value) for value in seq], dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("each sequence must be one-dimensional and non-empty")
    if np.any(~np.isfinite(values)):
        raise ValueError("BKT observations must be finite")
    return (values >= 0.5).astype(float)


def _clip(value: float, eps: float) -> float:
    return float(np.clip(value, eps, 1.0 - eps))
