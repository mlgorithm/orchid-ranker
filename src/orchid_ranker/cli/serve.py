"""Minimal long-running service entry point for container deployments."""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Sequence

from orchid_ranker.observability import (
    healthz,
    readyz,
    set_ready,
    start_health_server,
    start_metrics_server,
)


class _StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # pragma: no cover - exercised through container/runtime smoke
        if self.path in {"/", "/healthz"}:
            self._json(200, healthz())
            return
        if self.path == "/readyz":
            status = readyz()
            self._json(200 if status["status"] == "ok" else 503, status)
            return
        self.send_response(404)
        self.end_headers()

    def _json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format_str: str, *args: object) -> None:  # pragma: no cover
        pass


def _start_status_server(port: int, addr: str) -> HTTPServer:
    server = HTTPServer((addr, port), _StatusHandler)
    thread = threading.Thread(target=server.serve_forever, name="orchid-status", daemon=True)
    thread.start()
    return server


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Orchid Ranker health and metrics endpoints.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--metrics-port", type=int, default=9090)
    parser.add_argument("--health-port", type=int, default=8081)
    parser.add_argument("--no-metrics", action="store_true", help="Do not bind the Prometheus metrics endpoint")
    parser.add_argument(
        "--ready-on-start",
        action="store_true",
        help="Report readiness immediately. Use only for health-endpoint-only deployments.",
    )
    args = parser.parse_args(argv)

    stop = threading.Event()

    def _stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    status_server = _start_status_server(args.port, args.host)
    if not args.no_metrics:
        start_metrics_server(port=args.metrics_port, addr=args.host)
    start_health_server(port=args.health_port, addr=args.host)
    set_ready(bool(args.ready_on_start))
    try:
        while not stop.is_set():
            stop.wait(1.0)
    finally:
        set_ready(False)
        status_server.shutdown()
        time.sleep(0.1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
