"""Logging utilities for enterprise deployments."""

from __future__ import annotations

import logging
from logging import Logger
from typing import Optional


def configure_logging(
    *,
    level: str | int = "INFO",
    logger_name: Optional[str] = "orchid_ranker",
    json: bool = False,
) -> Logger:
    """Configure a library logger with sensible defaults.

    Parameters
    ----------
    level:
        Logging level (string or numeric). Defaults to ``"INFO"``.
    logger_name:
        Name of the logger to configure. ``None`` configures the root logger.
    json:
        When ``True`` the handler emits JSON lines (enterprise-friendly for
        aggregation); otherwise fall back to a concise text formatter.
    """

    lvl = logging.getLevelName(level) if isinstance(level, str) else int(level)
    target_logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()

    # Avoid duplicate handlers when reconfiguring.
    if target_logger.handlers:
        for handler in list(target_logger.handlers):
            target_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    if json:
        try:
            import json as _json

            class _JsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial wrapper
                    payload = {
                        "level": record.levelname,
                        "name": record.name,
                        "message": record.getMessage(),
                    }
                    if record.exc_info:
                        payload["exc_info"] = self.formatException(record.exc_info)
                    return _json.dumps(payload)

            handler.setFormatter(_JsonFormatter())
        except Exception:  # pragma: no cover - fallback for environments without json
            handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    target_logger.addHandler(handler)
    target_logger.setLevel(lvl)
    target_logger.propagate = False
    return target_logger


__all__ = ["configure_logging"]
