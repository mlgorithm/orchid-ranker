"""Structured audit logging."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request as _urllib_request


@dataclass
class AuditEvent:
    """Representation of a single audit log record."""

    event_type: str
    actor: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        data = asdict(self)
        return json.dumps(data, separators=(",", ":"), default=str)


class AuditLogger:
    """Append-only JSONL audit log writer."""

    def __init__(
        self,
        path: Path,
        *,
        ensure_dir: bool = True,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.path = Path(path)
        if ensure_dir:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = float(timeout)

    def log(self, event: AuditEvent) -> None:
        line = event.to_json()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"{line}\n")
        self._send_remote(line)

    def log_event(self, event_type: str, actor: str, payload: Optional[Dict[str, Any]] = None) -> None:
        evt = AuditEvent(event_type=event_type, actor=actor, payload=payload or {})
        self.log(evt)

    def _send_remote(self, event_json: str) -> None:
        if not self.endpoint:
            return
        data = event_json.encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = _urllib_request.Request(self.endpoint, data=data, headers=headers, method="POST")
        with _urllib_request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            status = getattr(resp, "status", 200)
            if status >= 400:
                raise RuntimeError(f"Audit endpoint returned status {status}")

    @classmethod
    def from_env(cls, path: Path, *, ensure_dir: bool = True) -> "AuditLogger":
        return cls(
            path,
            ensure_dir=ensure_dir,
            endpoint=os.getenv("ORCHID_AUDIT_ENDPOINT"),
            api_key=os.getenv("ORCHID_AUDIT_API_KEY"),
            timeout=float(os.getenv("ORCHID_AUDIT_TIMEOUT", "10.0")),
        )
