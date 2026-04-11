from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load as load_extension

_EXTENSION_NAME = "orchid_fast_score"
_SRC_ROOT = Path(__file__).resolve().parent
_FAST_MOD = None
_BUILD_ERROR: Optional[Exception] = None
_LOAD_LOCK = threading.Lock()


def ensure_fast_score(*, verbose: bool = False):
    """
    Attempt to build/load the optional fast scoring extension.
    Returns the loaded module on success, or None if compilation failed.
    """
    global _FAST_MOD, _BUILD_ERROR
    # Fast path without lock (double-checked locking)
    if _FAST_MOD is not None:
        return _FAST_MOD
    if _BUILD_ERROR is not None:
        return None
    with _LOAD_LOCK:
        # Re-check under lock to prevent double compilation
        if _FAST_MOD is not None:
            return _FAST_MOD
        if _BUILD_ERROR is not None:
            return None
        try:
            _FAST_MOD = load_extension(
                name=_EXTENSION_NAME,
                sources=[str(_SRC_ROOT / "fast_score.cpp")],
                extra_cflags=["-O3"],
                verbose=verbose,
            )
        except Exception as exc:  # pragma: no cover - build failures depend on env
            _BUILD_ERROR = exc
            return None
        return _FAST_MOD


def fast_score(user_vec: torch.Tensor, item_matrix: torch.Tensor, *, use_native: bool = True) -> torch.Tensor:
    """
    Fast path for dense user·item scoring. Falls back to torch.matmul if the native
    extension is unavailable or disabled.
    """
    if user_vec.dim() != 2 or item_matrix.dim() != 2:
        raise ValueError("user_vec and item_matrix must be rank-2 tensors")
    if user_vec.size(1) != item_matrix.size(1):
        raise ValueError("Dimension mismatch between user_vec and item_matrix")

    if not use_native:
        return user_vec @ item_matrix.T

    mod = ensure_fast_score()
    if mod is None:
        return user_vec @ item_matrix.T
    return mod.fast_score(user_vec, item_matrix)


__all__ = [
    "ensure_fast_score",
    "fast_score",
]

