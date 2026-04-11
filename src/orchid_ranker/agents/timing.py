"""Timing recorder for performance profiling."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


class _TimingRecorder:
    def __init__(self, path: Optional[str | Path], max_rounds: int):
        self._max_rounds = max(0, int(max_rounds or 0))
        self._path = Path(path) if path else None
        self.enabled = self._path is not None and self._max_rounds > 0
        if self.enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._current_entry: Optional[dict] = None
        self._written = 0

    def begin(self, round_idx: int) -> None:
        if not self.enabled or self._written >= self._max_rounds:
            self._current_entry = None
            return
        self._current_entry = {"round": int(round_idx), "phases": {}, "start": time.perf_counter()}

    @contextmanager
    def phase(self, name: str):
        if not self._current_entry:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            phases = self._current_entry.setdefault("phases", {})
            phases[name] = phases.get(name, 0.0) + elapsed

    def finish(self, extras: Optional[dict] = None) -> None:
        if not self._current_entry:
            return
        entry = self._current_entry
        entry["total"] = time.perf_counter() - entry.pop("start", time.perf_counter())
        if extras:
            entry["metrics"] = extras
        self._write(entry)
        self._current_entry = None

    def _write(self, entry: dict) -> None:
        if not self.enabled or self._path is None:
            return
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        self._written += 1
        if self._written >= self._max_rounds:
            self.enabled = False


__all__: list[str] = []
