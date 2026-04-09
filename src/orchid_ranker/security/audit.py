"""Structured audit logging with HMAC hash chaining and optional encryption."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as _urllib_request

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:
    Fernet = None  # type: ignore
    InvalidToken = None  # type: ignore


@dataclass
class VerificationResult:
    """Result of audit log integrity verification."""

    valid: bool
    lines_checked: int
    first_error_line: Optional[int] = None
    error_message: Optional[str] = None


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
    """Append-only JSONL audit log writer with optional HMAC chaining and encryption."""

    def __init__(
        self,
        path: Path,
        *,
        ensure_dir: bool = True,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        hmac_key: Optional[bytes] = None,
        encryption_key: Optional[bytes] = None,
    ) -> None:
        self.path = Path(path)
        if ensure_dir:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = float(timeout)
        self.hmac_key = hmac_key
        self.encryption_key = encryption_key
        self._prev_hash = "0" * 64  # Genesis hash for hash chaining

    def log(self, event: AuditEvent) -> None:
        line = event.to_json()

        # Add HMAC hash chaining if enabled
        if self.hmac_key:
            line_dict = json.loads(line)
            line_dict["_prev_hash"] = self._prev_hash
            # Compute HMAC of current line (excluding _hash field)
            line_content = json.dumps(line_dict, separators=(",", ":"), default=str)
            current_hash = hmac.new(self.hmac_key, line_content.encode("utf-8"), hashlib.sha256).hexdigest()
            line_dict["_hash"] = current_hash
            line = json.dumps(line_dict, separators=(",", ":"), default=str)
            self._prev_hash = current_hash

        # Encrypt if enabled
        if self.encryption_key:
            if Fernet is None:
                raise RuntimeError("cryptography library required for encryption. Install with: pip install cryptography")
            cipher = Fernet(self.encryption_key)
            line = base64.b64encode(cipher.encrypt(line.encode("utf-8"))).decode("utf-8")

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
        hmac_key = None
        if hmac_key_str := os.getenv("ORCHID_AUDIT_HMAC_KEY"):
            hmac_key = bytes.fromhex(hmac_key_str)

        encryption_key = None
        if enc_key_str := os.getenv("ORCHID_AUDIT_ENCRYPTION_KEY"):
            encryption_key = enc_key_str.encode("utf-8") if isinstance(enc_key_str, str) else enc_key_str

        return cls(
            path,
            ensure_dir=ensure_dir,
            endpoint=os.getenv("ORCHID_AUDIT_ENDPOINT"),
            api_key=os.getenv("ORCHID_AUDIT_API_KEY"),
            timeout=float(os.getenv("ORCHID_AUDIT_TIMEOUT", "10.0")),
            hmac_key=hmac_key,
            encryption_key=encryption_key,
        )


def verify_log_integrity(path: Path, hmac_key: bytes) -> VerificationResult:
    """Verify audit log integrity using HMAC hash chaining.

    Args:
        path: Path to the JSONL audit log file.
        hmac_key: The HMAC key used to sign the log entries.

    Returns:
        VerificationResult with verification status and details.
    """
    if not path.exists():
        return VerificationResult(valid=False, lines_checked=0, error_message="Log file not found")

    prev_hash = "0" * 64
    lines_checked = 0

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line:
                    continue

                lines_checked += 1
                try:
                    line_dict = json.loads(line)
                except json.JSONDecodeError as e:
                    return VerificationResult(
                        valid=False,
                        lines_checked=lines_checked,
                        first_error_line=line_num,
                        error_message=f"JSON parse error: {e}",
                    )

                # Verify _prev_hash matches expected previous hash
                actual_prev_hash = line_dict.get("_prev_hash")
                if actual_prev_hash != prev_hash:
                    return VerificationResult(
                        valid=False,
                        lines_checked=lines_checked,
                        first_error_line=line_num,
                        error_message=f"Hash chain broken at line {line_num}: expected _prev_hash={prev_hash}, got {actual_prev_hash}",
                    )

                # Verify _hash is correct for this line
                expected_hash = line_dict.pop("_hash", None)
                if expected_hash is None:
                    return VerificationResult(
                        valid=False,
                        lines_checked=lines_checked,
                        first_error_line=line_num,
                        error_message=f"Missing _hash field at line {line_num}",
                    )

                line_content = json.dumps(line_dict, separators=(",", ":"), default=str)
                computed_hash = hmac.new(hmac_key, line_content.encode("utf-8"), hashlib.sha256).hexdigest()

                if computed_hash != expected_hash:
                    return VerificationResult(
                        valid=False,
                        lines_checked=lines_checked,
                        first_error_line=line_num,
                        error_message=f"HMAC mismatch at line {line_num}: expected {expected_hash}, computed {computed_hash}",
                    )

                prev_hash = expected_hash

    except Exception as e:
        return VerificationResult(
            valid=False,
            lines_checked=lines_checked,
            error_message=f"Verification error: {e}",
        )

    return VerificationResult(valid=True, lines_checked=lines_checked)


def decrypt_log(path: Path, encryption_key: bytes) -> List[str]:
    """Decrypt an encrypted audit log file.

    Args:
        path: Path to the encrypted JSONL audit log file.
        encryption_key: The Fernet key used to encrypt the log entries.

    Returns:
        List of decrypted JSON strings, one per line.

    Raises:
        RuntimeError: If cryptography library is not installed.
        InvalidToken: If decryption fails (wrong key or corrupted data).
    """
    if Fernet is None:
        raise RuntimeError("cryptography library required for decryption. Install with: pip install cryptography")

    cipher = Fernet(encryption_key)
    decrypted_lines = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            encrypted_data = base64.b64decode(line)
            decrypted = cipher.decrypt(encrypted_data).decode("utf-8")
            decrypted_lines.append(decrypted)

    return decrypted_lines
