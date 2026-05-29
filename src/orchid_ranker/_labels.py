"""Strict parsing for binary outcome labels."""
from __future__ import annotations

import math
from typing import Any


def parse_binary_label(value: Any, *, name: str = "label") -> bool:
    """Return a strict boolean for binary 0/1 outcomes."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1"}:
            return True
        if normalized in {"false", "f", "0"}:
            return False
        raise ValueError(f"{name} must be a strict boolean or 0/1 value")

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a strict boolean or 0/1 value") from exc

    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric == 0.0:
        return False
    if numeric == 1.0:
        return True
    raise ValueError(f"{name} must be a strict boolean or 0/1 value")


def parse_binary_float(value: Any, *, name: str = "label") -> float:
    """Return ``0.0`` or ``1.0`` for a strict binary outcome."""

    return 1.0 if parse_binary_label(value, name=name) else 0.0
