"""Optional dependency helpers for orchid-ranker."""
from __future__ import annotations

from types import ModuleType

__all__ = [
    "get_matplotlib_pyplot",
    "torch_available",
    "matplotlib_available",
    "require_matplotlib",
    "require_torch",
    "get_torch",
]

_torch: ModuleType | None = None
_torch_checked = False
_matplotlib_pyplot: ModuleType | None = None
_matplotlib_checked = False


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


def matplotlib_available() -> bool:
    """Return True if Matplotlib is installed."""
    global _matplotlib_pyplot, _matplotlib_checked
    if not _matplotlib_checked:
        try:
            import matplotlib.pyplot as _plt

            _matplotlib_pyplot = _plt
        except ImportError:
            _matplotlib_pyplot = None
        _matplotlib_checked = True
    return _matplotlib_pyplot is not None


def require_torch(feature: str = "This feature") -> None:
    """Raise ImportError with a helpful message if torch is missing."""
    if not torch_available():
        raise ImportError(
            f"{feature} requires PyTorch. "
            "Install it with: pip install orchid-ranker[adaptive]  "
            "or: pip install torch>=2.0"
        )


def require_matplotlib(feature: str = "This feature") -> None:
    """Raise ImportError with a helpful message if Matplotlib is missing."""
    if not matplotlib_available():
        raise ImportError(
            f"{feature} requires Matplotlib. "
            "Install it with: pip install orchid-ranker[viz]  "
            "or: pip install matplotlib>=3.6"
        )


def get_torch():
    """Return the torch module, raising ImportError if unavailable."""
    require_torch()
    return _torch


def get_matplotlib_pyplot(feature: str = "This feature") -> ModuleType:
    """Return matplotlib.pyplot, raising ImportError if unavailable."""
    require_matplotlib(feature)
    assert _matplotlib_pyplot is not None
    return _matplotlib_pyplot
