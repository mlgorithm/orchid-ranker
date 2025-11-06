from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DevicePreference = Literal["auto", "cuda", "mps", "cpu"]


@dataclass(frozen=True)
class DeviceChoice:
    name: str
    torch_device: torch.device
    reason: str


def _mps_available() -> bool:
    try:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def select_device(preference: DevicePreference = "auto") -> DeviceChoice:
    pref = preference.lower()
    if pref not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device preference '{preference}'")

    candidates: list[tuple[str, bool, str]] = [
        ("cuda", torch.cuda.is_available(), "CUDA device detected"),
        ("mps", _mps_available(), "Apple MPS backend detected"),
        ("cpu", True, "Defaulting to CPU"),
    ]

    if pref != "auto":
        for name, available, reason in candidates:
            if name == pref:
                if not available:
                    raise RuntimeError(f"Requested device '{pref}' is not available on this system")
                return DeviceChoice(name=name, torch_device=torch.device(name), reason=reason)
        raise RuntimeError(f"Device preference '{pref}' not recognized")

    for name, available, reason in candidates:
        if available:
            return DeviceChoice(name=name, torch_device=torch.device(name), reason=reason)

    raise RuntimeError("Failed to select a torch device (this should be unreachable)")
