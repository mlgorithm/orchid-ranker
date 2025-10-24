"""Prometheus metrics helpers for Orchid Ranker."""
from __future__ import annotations

from typing import Optional

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, start_http_server

_REGISTRY = CollectorRegistry()

TRAINING_RUNS = Counter(
    "orchid_training_runs_total",
    "Number of training runs executed",
    registry=_REGISTRY,
)

TRAINING_DURATION = Histogram(
    "orchid_training_duration_seconds",
    "Histogram of training duration",
    buckets=(1, 5, 10, 30, 60, 120, 300),
    registry=_REGISTRY,
)

DP_EPSILON = Gauge(
    "orchid_dp_epsilon_cumulative",
    "Latest cumulative epsilon reported by DP accountant",
    registry=_REGISTRY,
)


def metrics_registry() -> CollectorRegistry:
    """Return the shared CollectorRegistry."""

    return _REGISTRY


def start_metrics_server(port: int = 9090, addr: str = "0.0.0.0") -> None:  # pragma: no cover - binds socket
    start_http_server(port, addr=addr, registry=_REGISTRY)


def record_training(duration_seconds: float, epsilon: Optional[float] = None) -> None:
    TRAINING_RUNS.inc()
    TRAINING_DURATION.observe(duration_seconds)
    if epsilon is not None:
        DP_EPSILON.set(epsilon)


def export_metrics() -> bytes:
    return generate_latest(_REGISTRY)


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
