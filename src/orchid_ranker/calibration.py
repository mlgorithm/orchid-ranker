"""Probability calibration helpers for adaptive-learning policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

__all__ = [
    "CalibrationReport",
    "IsotonicProbabilityCalibrator",
    "TemperatureScaler",
    "brier_score",
    "expected_calibration_error",
]


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration diagnostics for predicted correctness probabilities."""

    n: int
    brier: float
    ece: float
    positive_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class TemperatureScaler:
    """Grid-search temperature scaling for logits or probabilities."""

    def __init__(self, *, temperatures: Sequence[float] | None = None) -> None:
        self.temperatures = list(temperatures or np.geomspace(0.25, 8.0, 41))
        if not self.temperatures or any(float(t) <= 0.0 for t in self.temperatures):
            raise ValueError("temperatures must contain positive values")
        self.temperature_: float | None = None
        self.from_logits_: bool = False
        self.report_: CalibrationReport | None = None

    @property
    def is_fitted(self) -> bool:
        return self.temperature_ is not None

    def fit(self, scores: Sequence[float], labels: Sequence[int | bool | float], *, from_logits: bool = False) -> "TemperatureScaler":
        raw = _as_float_array(scores, "scores")
        y = _labels(labels)
        if raw.shape[0] != y.shape[0]:
            raise ValueError("scores and labels must have the same length")
        logits = raw if from_logits else _logit(_clip_probs(raw))
        best_temp = min(self.temperatures, key=lambda temp: _log_loss(y, _sigmoid(logits / float(temp))))
        self.temperature_ = float(best_temp)
        self.from_logits_ = bool(from_logits)
        probs = _sigmoid(logits / self.temperature_)
        self.report_ = _report(probs, y)
        return self

    def predict_proba(self, scores: Sequence[float], *, from_logits: bool | None = None) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaler must be fitted before prediction")
        raw = _as_float_array(scores, "scores")
        use_logits = self.from_logits_ if from_logits is None else bool(from_logits)
        logits = raw if use_logits else _logit(_clip_probs(raw))
        return _sigmoid(logits / self.temperature_).astype(np.float64)


class IsotonicProbabilityCalibrator:
    """Monotonic isotonic calibration for predicted correctness probabilities."""

    def __init__(self, *, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds
        self.model_: IsotonicRegression | None = None
        self.report_: CalibrationReport | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model_ is not None

    def fit(self, probabilities: Sequence[float], labels: Sequence[int | bool | float]) -> "IsotonicProbabilityCalibrator":
        probs = _clip_probs(_as_float_array(probabilities, "probabilities"))
        y = _labels(labels)
        if probs.shape[0] != y.shape[0]:
            raise ValueError("probabilities and labels must have the same length")
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=self.out_of_bounds).fit(probs, y)
        calibrated = self.model_.predict(probs)
        self.report_ = _report(calibrated, y)
        return self

    def predict_proba(self, probabilities: Sequence[float]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("IsotonicProbabilityCalibrator must be fitted before prediction")
        probs = _clip_probs(_as_float_array(probabilities, "probabilities"))
        return np.asarray(self.model_.predict(probs), dtype=np.float64)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int | bool | float],
    *,
    n_bins: int = 10,
) -> float:
    """Return equal-width expected calibration error."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    probs = _clip_probs(_as_float_array(probabilities, "probabilities"))
    y = _labels(labels)
    if probs.shape[0] != y.shape[0]:
        raise ValueError("probabilities and labels must have the same length")
    if probs.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not bool(np.any(mask)):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(probs[mask])) - float(np.mean(y[mask])))
    return float(ece)


def brier_score(probabilities: Sequence[float], labels: Sequence[int | bool | float]) -> float:
    probs = _clip_probs(_as_float_array(probabilities, "probabilities"))
    y = _labels(labels)
    if probs.shape[0] != y.shape[0]:
        raise ValueError("probabilities and labels must have the same length")
    if probs.size == 0:
        return 0.0
    return float(np.mean((probs - y) ** 2))


def _report(probabilities: np.ndarray, labels: np.ndarray) -> CalibrationReport:
    return CalibrationReport(
        n=int(labels.size),
        brier=brier_score(probabilities, labels),
        ece=expected_calibration_error(probabilities, labels),
        positive_rate=float(np.mean(labels)) if labels.size else 0.0,
    )


def _log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    clipped = _clip_probs(probs)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = _clip_probs(values)
    return np.log(clipped / (1.0 - clipped))


def _clip_probs(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 1e-6, 1.0 - 1e-6)


def _labels(values: Sequence[int | bool | float]) -> np.ndarray:
    arr = _as_float_array(values, "labels")
    return (arr >= 0.5).astype(float)


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr
