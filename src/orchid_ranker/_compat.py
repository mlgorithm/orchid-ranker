"""Optional dependency helpers for orchid-ranker."""
from __future__ import annotations

__all__ = [
    "torch_available",
    "require_torch",
    "get_torch",
]

_torch = None
_torch_checked = False


def torch_available() -> bool:
    """Return True if PyTorch is installed."""
    global _torch, _torch_checked
    if not _torch_checked:
        try:
            import torch as _t
            _torch = _t
        except ImportError:
            _torch = None
        _torch_checked = True
    return _torch is not None


def require_torch(feature: str = "This feature") -> None:
    """Raise ImportError with a helpful message if torch is missing."""
    if not torch_available():
        raise ImportError(
            f"{feature} requires PyTorch. "
            "Install it with: pip install orchid-ranker[ml]  "
            "or: pip install torch>=1.13"
        )


def get_torch():
    """Return the torch module, raising ImportError if unavailable."""
    require_torch()
    return _torch
