"""Utilities for configuring differential-privacy parameters."""
from __future__ import annotations

from typing import Dict

__all__ = ["DP_PRESETS", "get_dp_config"]

DP_PRESETS: Dict[str, dict] = {
    "off": {"enabled": False},
    "eps_2": {"enabled": True, "noise_multiplier": 0.8, "sample_rate": 0.02, "delta": 1e-5, "max_grad_norm": 1.0},
    "eps_1": {"enabled": True, "noise_multiplier": 1.2, "sample_rate": 0.02, "delta": 1e-5, "max_grad_norm": 1.0},
    "eps_05": {"enabled": True, "noise_multiplier": 2.0, "sample_rate": 0.015, "delta": 1e-5, "max_grad_norm": 1.0},
    "eps_02": {"enabled": True, "noise_multiplier": 2.8, "sample_rate": 0.01, "delta": 1e-5, "max_grad_norm": 1.0},
}


def get_dp_config(preset: str | dict) -> dict:
    """Return a DP configuration dict from a preset name or explicit dict."""
    if isinstance(preset, dict):
        return preset
    key = preset.lower()
    if key not in DP_PRESETS:
        raise KeyError(f"Unknown DP preset '{preset}'. Available: {list(DP_PRESETS)}")
    return dict(DP_PRESETS[key])
