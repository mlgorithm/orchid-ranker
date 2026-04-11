"""Privacy-aware JSONL logging for recommendation events."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch as _torch
except ImportError:
    _torch = None


def _d(*args) -> None:
    """Debug logging (respects ORCHID_DEBUG_REC env var)."""
    if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
        logger.debug("%s", " ".join(str(a) for a in args))


class JSONLLogger:
    """Privacy-aware JSONL logger for recommendation events and metrics.

    Parameters
    ----------
    path : Path
        Output file path for JSONL logs.
    max_feature_dump : int, optional
        Maximum features to dump before truncation (default: 256).
    """
    def __init__(self, path: Path, max_feature_dump: int = 256):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_feature_dump = int(max_feature_dump)
        _d(f"JSONLLogger -> {self.path}")

    def _to_jsonable(self, obj):
        if isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        try:
            import numpy as _np
            if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        try:
            if _torch is not None and isinstance(obj, _torch.Tensor):
                return obj.detach().cpu().tolist()
        except Exception:
            pass
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        return str(obj)

    def log(self, record: dict):
        # cap huge user_features (if present)
        if "inputs" in record and "user_features" in record["inputs"]:
            fmap = record["inputs"]["user_features"]
            if isinstance(fmap, dict) and len(fmap) > self.max_feature_dump:
                keys = list(fmap.keys())
                head = {k: fmap[k] for k in keys[: self.max_feature_dump]}
                record["inputs"]["user_features_sample"] = head
                record["inputs"]["user_features_count"] = len(fmap)
                del record["inputs"]["user_features"]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._to_jsonable(record), ensure_ascii=False) + "\n")
        if os.getenv("ORCHID_DEBUG_REC", "").lower() in {"1", "true", "yes", "on"}:
            rtype = record.get("type", "unknown")
            rround = record.get("round", None)
            _d(f"JSONL write type={rtype} round={rround}")


__all__ = [
    "JSONLLogger",
]
