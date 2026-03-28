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
    """Get the shared Prometheus CollectorRegistry for Orchid Ranker.

    Returns
    -------
    prometheus_client.CollectorRegistry
        Registry containing all Orchid metrics.
    """
    return _REGISTRY


def start_metrics_server(port: int = 9090, addr: str = "0.0.0.0") -> None:  # pragma: no cover - binds socket
    """Start an HTTP server exposing Prometheus metrics.

    Parameters
    ----------
    port : int, optional
        Port to bind to (default: 9090).
    addr : str, optional
        Network address to bind to (default: "0.0.0.0").
    """
    start_http_server(port, addr=addr, registry=_REGISTRY)


def record_training(duration_seconds: float, epsilon: Optional[float] = None) -> None:
    """Record metrics from a completed training run.

    Increments training run counter, observes duration, and optionally sets
    the cumulative DP epsilon.

    Parameters
    ----------
    duration_seconds : float
        Training duration in seconds.
    epsilon : float, optional
        Cumulative DP epsilon, if applicable.
    """
    TRAINING_RUNS.inc()
    TRAINING_DURATION.observe(duration_seconds)
    if epsilon is not None:
        DP_EPSILON.set(epsilon)


def export_metrics() -> bytes:
    """Export all collected metrics in Prometheus text format.

    Returns
    -------
    bytes
        Prometheus text-format metrics.
    """
    return generate_latest(_REGISTRY)


def metrics_content_type() -> str:
    """Get the MIME type for Prometheus metrics.

    Returns
    -------
    str
        Content-Type header value.
    """
    return CONTENT_TYPE_LATEST
