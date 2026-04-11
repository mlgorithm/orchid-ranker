"""Prometheus metrics, OpenTelemetry support, and health checks for Orchid Ranker."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)

# ============================================================================
# OpenTelemetry Support (optional)
# ============================================================================
try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

# ============================================================================
# Prometheus Registry & Metrics
# ============================================================================
_REGISTRY = CollectorRegistry()

# Training metrics (existing)
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

# Inference metrics (new)
INFERENCE_LATENCY = Histogram(
    "orchid_inference_latency_seconds",
    "Inference latency in seconds",
    ["strategy"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    registry=_REGISTRY,
)

INFERENCE_REQUESTS = Counter(
    "orchid_inference_requests_total",
    "Total inference requests",
    ["strategy", "outcome"],
    registry=_REGISTRY,
)

INFERENCE_ERRORS = Counter(
    "orchid_inference_errors_total",
    "Total inference errors",
    ["error_type"],
    registry=_REGISTRY,
)

RECOMMENDATION_LIST_SIZE = Histogram(
    "orchid_recommendation_list_size",
    "Items returned per inference request",
    buckets=(1, 5, 10, 20, 50, 100),
    registry=_REGISTRY,
)

ACTIVE_USERS = Gauge(
    "orchid_active_users",
    "Number of users with recent activity",
    registry=_REGISTRY,
)

MODEL_STALENESS = Gauge(
    "orchid_model_staleness_seconds",
    "Seconds since last model update",
    registry=_REGISTRY,
)

DP_BUDGET_REMAINING = Gauge(
    "orchid_dp_budget_remaining",
    "DP budget remaining (1 - current_epsilon / max_epsilon)",
    registry=_REGISTRY,
)


# ============================================================================
# Registry & Server Functions
# ============================================================================

def metrics_registry() -> CollectorRegistry:
    """Get the shared Prometheus CollectorRegistry for Orchid Ranker.

    Returns
    -------
    prometheus_client.CollectorRegistry
        Registry containing all Orchid metrics.
    """
    return _REGISTRY


def start_metrics_server(port: int = 9090, addr: str = "0.0.0.0") -> None:  # pragma: no cover
    """Start an HTTP server exposing Prometheus metrics.

    Parameters
    ----------
    port : int, optional
        Port to bind to (default: 9090).
    addr : str, optional
        Network address to bind to (default: "0.0.0.0").
    """
    start_http_server(port, addr=addr, registry=_REGISTRY)


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


# ============================================================================
# Training Recording
# ============================================================================

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


# ============================================================================
# Inference Recording
# ============================================================================

def record_inference(
    strategy: str,
    duration_seconds: float,
    list_size: int,
    outcome: str = "success",
) -> None:
    """Record metrics from a completed inference call.

    Parameters
    ----------
    strategy : str
        Ranking strategy used (e.g., "bandit", "thompson").
    duration_seconds : float
        Inference latency in seconds.
    list_size : int
        Number of items returned.
    outcome : str, optional
        Result outcome: "success", "error", "timeout" (default: "success").
    """
    INFERENCE_LATENCY.labels(strategy=strategy).observe(duration_seconds)
    INFERENCE_REQUESTS.labels(strategy=strategy, outcome=outcome).inc()
    RECOMMENDATION_LIST_SIZE.observe(list_size)


def record_inference_error(error_type: str) -> None:
    """Record an inference error.

    Parameters
    ----------
    error_type : str
        Type of error encountered (e.g., "model_not_ready", "invalid_input").
    """
    INFERENCE_ERRORS.labels(error_type=error_type).inc()


# ============================================================================
# System State Setters
# ============================================================================

def set_active_users(count: int) -> None:
    """Update the active users gauge.

    Parameters
    ----------
    count : int
        Number of users with recent activity.
    """
    ACTIVE_USERS.set(count)


def set_model_staleness(seconds: float) -> None:
    """Update the model staleness gauge.

    Parameters
    ----------
    seconds : float
        Seconds since last model update.
    """
    MODEL_STALENESS.set(seconds)


def set_dp_budget_remaining(current_epsilon: float, max_epsilon: float) -> None:
    """Update the DP budget remaining gauge.

    Parameters
    ----------
    current_epsilon : float
        Current cumulative epsilon spent.
    max_epsilon : float
        Maximum allowed epsilon.
    """
    remaining = max(0.0, 1.0 - current_epsilon / max_epsilon) if max_epsilon > 0 else 0.0
    DP_BUDGET_REMAINING.set(remaining)


# ============================================================================
# OpenTelemetry Setup (optional)
# ============================================================================

def setup_opentelemetry(service_name: str = "orchid-ranker") -> None:
    """Initialize OpenTelemetry tracing and metrics exporters.

    Requires the optional OTEL dependencies:
        pip install 'orchid-ranker[otel]'

    Uses OTLP exporter by default. Configure endpoint via:
        OTEL_EXPORTER_OTLP_ENDPOINT environment variable

    Parameters
    ----------
    service_name : str, optional
        Service name for resource attributes (default: "orchid-ranker").

    Raises
    ------
    ImportError
        If OpenTelemetry packages are not installed.
    """
    if not _HAS_OTEL:
        raise ImportError(
            "OpenTelemetry not installed. "
            "Install with: pip install 'orchid-ranker[otel]'"
        )

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter as GrpcMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcTraceExporter,
        )

        trace_exporter = GrpcTraceExporter()
        GrpcMetricExporter()
    except ImportError:
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter as HttpMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as HttpTraceExporter,
            )

            trace_exporter = HttpTraceExporter()
            HttpMetricExporter()
        except ImportError as e:
            raise ImportError(
                "OpenTelemetry OTLP exporters not installed. "
                "Install with: pip install 'orchid-ranker[otel]'"
            ) from e

    resource = Resource.create({"service.name": service_name})

    # Setup tracing
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tp)

    # Setup metrics
    mp = MeterProvider(resource=resource)
    otel_metrics.set_meter_provider(mp)


# ============================================================================
# Health Check Endpoints
# ============================================================================

_ready = False
_ready_lock = threading.Lock()


def set_ready(ready: bool = True) -> None:
    """Mark the service as ready to serve traffic.

    Thread-safe: protected by an internal lock.

    Parameters
    ----------
    ready : bool, optional
        Whether service is ready (default: True).
    """
    global _ready
    with _ready_lock:
        _ready = ready


def is_ready() -> bool:
    """Return the current readiness state (thread-safe).

    Returns
    -------
    bool
        True if the service has been marked ready via ``set_ready()``.
    """
    with _ready_lock:
        return _ready


def healthz() -> dict:
    """Liveness probe: returns ok if process is running.

    Returns
    -------
    dict
        Status dict with "status" key.
    """
    return {"status": "ok"}


def readyz() -> dict:
    """Readiness probe: returns ok only if model is loaded.

    Returns
    -------
    dict
        Status dict with "status" key and optional message.
    """
    with _ready_lock:
        ready = _ready
    if ready:
        return {"status": "ok"}
    return {"status": "not_ready", "message": "Model not loaded"}


class _HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for /healthz and /readyz endpoints."""

    def do_GET(self) -> None:  # pragma: no cover
        """Handle GET requests."""
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(healthz()).encode())
        elif self.path == "/readyz":
            status = readyz()
            code = 200 if status["status"] == "ok" else 503
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format_str: str, *args) -> None:  # pragma: no cover
        """Suppress HTTP server logs."""
        pass


def start_health_server(port: int = 8081, addr: str = "0.0.0.0") -> None:  # pragma: no cover
    """Start a minimal HTTP health check server for Kubernetes probes.

    Provides two endpoints:
    - GET /healthz: Liveness probe (always returns 200 if running)
    - GET /readyz: Readiness probe (returns 200 if model loaded, 503 otherwise)

    Parameters
    ----------
    port : int, optional
        Port to bind to (default: 8081).
    addr : str, optional
        Network address to bind to (default: "0.0.0.0").
    """
    server = HTTPServer((addr, port), _HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Metrics
    "TRAINING_RUNS",
    "TRAINING_DURATION",
    "DP_EPSILON",
    "INFERENCE_LATENCY",
    "INFERENCE_REQUESTS",
    "INFERENCE_ERRORS",
    "RECOMMENDATION_LIST_SIZE",
    "ACTIVE_USERS",
    "MODEL_STALENESS",
    "DP_BUDGET_REMAINING",
    # Registry & server
    "metrics_registry",
    "start_metrics_server",
    "export_metrics",
    "metrics_content_type",
    # Recording
    "record_training",
    "record_inference",
    "record_inference_error",
    # State
    "set_active_users",
    "set_model_staleness",
    "set_dp_budget_remaining",
    # OpenTelemetry
    "setup_opentelemetry",
    # Health checks
    "set_ready",
    "is_ready",
    "healthz",
    "readyz",
    "start_health_server",
]
