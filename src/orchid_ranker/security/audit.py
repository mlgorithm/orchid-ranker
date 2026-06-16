"""Structured audit logging with HMAC hash chaining and optional encryption."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
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


def _env_flag(name: str) -> bool:
    """Return True if env var ``name`` is set to a truthy value."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
        require_hmac: bool = False,
    ) -> None:
        self.path = Path(path)
        if ensure_dir:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        # An audit logger without an HMAC key writes UNSIGNED records: they have
        # no tamper-evidence and the chain cannot be verified later. The default
        # still allows this (backward compatible), but the downgrade must never
        # be silent. Callers can opt into a hard failure via require_hmac=True or
        # the ORCHID_AUDIT_REQUIRE_HMAC env var.
        if hmac_key is None:
            if require_hmac:
                raise ValueError(
                    "AuditLogger requires an HMAC key but none was provided. "
                    "Pass hmac_key=... (or set ORCHID_AUDIT_HMAC_KEY) to enable "
                    "tamper-evident hash chaining."
                )
            import warnings as _warnings

            _warnings.warn(
                "AuditLogger is running WITHOUT an HMAC key: audit records are "
                "unsigned and cannot be integrity-verified. Provide hmac_key "
                "(or ORCHID_AUDIT_HMAC_KEY) to enable tamper-evident chaining, "
                "or set require_hmac=True / ORCHID_AUDIT_REQUIRE_HMAC=1 to enforce it.",
                stacklevel=2,
            )
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "AuditLogger for %s constructed without an HMAC key; "
                "records will be unsigned and unverifiable.",
                self.path,
            )

        # Enforce HTTPS for audit endpoints in production.
        # Set ORCHID_AUDIT_ALLOW_HTTP=1 for local development/testing only.
        if endpoint and not endpoint.startswith("https://"):
            if os.environ.get("ORCHID_AUDIT_ALLOW_HTTP", ""):
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Audit endpoint %r does not use HTTPS. "
                    "Audit logs (including bearer tokens) will be sent unencrypted. "
                    "This is allowed only because ORCHID_AUDIT_ALLOW_HTTP is set.",
                    endpoint,
                )
            else:
                raise ValueError(
                    f"Audit endpoint {endpoint!r} does not use HTTPS. "
                    "Audit logs (including bearer tokens) would be sent unencrypted. "
                    "Use an https:// endpoint, or set ORCHID_AUDIT_ALLOW_HTTP=1 "
                    "for local development/testing."
                )
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = float(timeout)
        self.hmac_key = hmac_key

        # Validate Fernet key format if provided
        if encryption_key is not None:
            if Fernet is None:
                raise RuntimeError(
                    "cryptography library required for encryption. "
                    "Install with: pip install cryptography"
                )
            try:
                Fernet(encryption_key)  # validate key format
            except Exception as exc:
                raise ValueError(
                    "Invalid Fernet encryption key. Must be 32 url-safe "
                    "base64-encoded bytes (44 characters). Generate one with: "
                    "from cryptography.fernet import Fernet; Fernet.generate_key()"
                ) from exc
        self.encryption_key = encryption_key
        if self.hmac_key:
            self._prev_hash, self._next_seq = self._load_previous_hash()
        else:
            self._prev_hash = "0" * 64
            self._next_seq = 0
        # Thread safety: hash-chain state + file append must be atomic
        self._lock = threading.RLock()

    def log(self, event: AuditEvent) -> None:
        line = event.to_json()

        with self._lock:
            # Add HMAC hash chaining if enabled
            if self.hmac_key:
                line_dict = json.loads(line)
                line_dict["_prev_hash"] = self._prev_hash
                # Bind a monotonically increasing sequence number INTO the signed
                # content. Because _seq is covered by the HMAC and the verifier
                # asserts it starts at 0 and increments by exactly 1, deleting
                # records from the tail (or reordering them) is now detectable.
                line_dict["_seq"] = self._next_seq
                # Compute HMAC of current line (excluding _hash field)
                line_content = json.dumps(line_dict, separators=(",", ":"), default=str)
                current_hash = hmac.new(self.hmac_key, line_content.encode("utf-8"), hashlib.sha256).hexdigest()
                line_dict["_hash"] = current_hash
                line = json.dumps(line_dict, separators=(",", ":"), default=str)
                self._prev_hash = current_hash
                self._next_seq += 1

            # Encrypt if enabled
            if self.encryption_key:
                if Fernet is None:
                    raise RuntimeError("cryptography library required for encryption. Install with: pip install cryptography")
                cipher = Fernet(self.encryption_key)
                line = base64.b64encode(cipher.encrypt(line.encode("utf-8"))).decode("utf-8")

            self._append_line(line)

        # Remote send outside lock to avoid blocking
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

    def _append_line(self, line: str) -> None:
        def _secure_opener(path: str, flags: int) -> int:
            return os.open(path, flags, 0o600)

        with open(self.path, "a", encoding="utf-8", opener=_secure_opener) as fh:
            fh.write(f"{line}\n")

    def _load_previous_hash(self) -> tuple[str, int]:
        """Replay an existing log to recover chain state for appending.

        Returns the last record's hash (to seed ``_prev_hash``) and the next
        sequence number to assign (to seed ``_next_seq``). Raises if the
        existing chain is broken so we never append onto a tampered log.
        """
        if not self.path.exists():
            return "0" * 64, 0
        prev_hash = "0" * 64
        next_seq = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    line_dict = json.loads(line)
                except json.JSONDecodeError:
                    if self.encryption_key is None:
                        raise ValueError(
                            f"Existing audit log {self.path} is not plain JSON at line {line_num}; "
                            "pass encryption_key to append to encrypted logs"
                        )
                    decrypted = _decrypt_audit_line(line, self.encryption_key, line_num=line_num)
                    line_dict = json.loads(decrypted)
                actual_prev_hash = line_dict.get("_prev_hash")
                if not isinstance(actual_prev_hash, str) or not hmac.compare_digest(actual_prev_hash, prev_hash):
                    raise ValueError(
                        f"Existing audit log hash chain is broken at line {line_num}: "
                        f"expected _prev_hash={prev_hash}, got {actual_prev_hash}"
                    )
                # Sequence numbers must start at 0 and increment by exactly 1.
                actual_seq = line_dict.get("_seq")
                if actual_seq != next_seq:
                    raise ValueError(
                        f"Existing audit log sequence broken at line {line_num}: "
                        f"expected _seq={next_seq}, got {actual_seq}"
                    )
                current_hash = line_dict.get("_hash")
                if not current_hash:
                    raise ValueError(f"Existing audit log line {line_num} is missing _hash")
                check_dict = dict(line_dict)
                expected_hash = check_dict.pop("_hash")
                line_content = json.dumps(check_dict, separators=(",", ":"), default=str)
                computed_hash = hmac.new(self.hmac_key, line_content.encode("utf-8"), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(computed_hash, str(expected_hash)):
                    raise ValueError(f"Existing audit log HMAC mismatch at line {line_num}")
                prev_hash = str(current_hash)
                next_seq += 1
        return prev_hash, next_seq

    @classmethod
    def from_env(
        cls, path: Path, *, ensure_dir: bool = True, require_hmac: Optional[bool] = None
    ) -> "AuditLogger":
        hmac_key = None
        if hmac_key_str := os.getenv("ORCHID_AUDIT_HMAC_KEY"):
            hmac_key = bytes.fromhex(hmac_key_str)

        encryption_key = None
        if enc_key_str := os.getenv("ORCHID_AUDIT_ENCRYPTION_KEY"):
            encryption_key = enc_key_str.encode("ascii")

        # require_hmac=None means "honor the env var"; an explicit bool wins.
        if require_hmac is None:
            require_hmac = _env_flag("ORCHID_AUDIT_REQUIRE_HMAC")

        return cls(
            path,
            ensure_dir=ensure_dir,
            endpoint=os.getenv("ORCHID_AUDIT_ENDPOINT"),
            api_key=os.getenv("ORCHID_AUDIT_API_KEY"),
            timeout=float(os.getenv("ORCHID_AUDIT_TIMEOUT", "10.0")),
            hmac_key=hmac_key,
            encryption_key=encryption_key,
            require_hmac=require_hmac,
        )


def _decrypt_audit_line(line: str, encryption_key: bytes, *, line_num: int) -> str:
    """Decrypt one base64-encoded Fernet audit log line."""
    if Fernet is None:
        raise RuntimeError(
            "cryptography library required to verify encrypted audit logs. "
            "Install with: pip install cryptography"
        )

    try:
        encrypted_data = base64.b64decode(line, validate=True)
    except Exception as exc:
        raise ValueError(
            f"Encrypted audit log line {line_num} is not valid base64: {exc}"
        ) from exc

    try:
        return Fernet(encryption_key).decrypt(encrypted_data).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError(
            f"Failed to decrypt encrypted audit log at line {line_num}: {exc}"
        ) from exc


def verify_log_integrity(
    path: Path,
    hmac_key: bytes,
    *,
    encryption_key: Optional[bytes] = None,
    expected_records: Optional[int] = None,
) -> VerificationResult:
    """Verify audit log integrity using HMAC hash chaining.

    Every signed record carries a monotonically increasing ``_seq`` (covered by
    the HMAC). Verification always asserts the sequence starts at 0 and
    increments by exactly 1, which makes reordering and interior deletion
    (sequence gaps) detectable.

    Pure tail-truncation to a clean prefix (e.g. keeping records 0,1,2 and
    dropping the rest) leaves a self-consistent chain, so it cannot be detected
    from the file alone. Pass ``expected_records`` (the number of records that
    should be present, tracked out-of-band) to also catch tail-truncation: the
    final sequence must equal ``expected_records``.

    Args:
        path: Path to the JSONL audit log file.
        hmac_key: The HMAC key used to sign the log entries.
        encryption_key: Optional Fernet key for encrypted audit logs. If the
            log was written with encryption enabled, this key is required.
        expected_records: Optional expected number of records. When given,
            verification fails if the log is shorter (tail-truncated) or longer
            than expected.

    Returns:
        VerificationResult with verification status and details.
    """
    if not path.exists():
        return VerificationResult(valid=False, lines_checked=0, error_message="Log file not found")

    prev_hash = "0" * 64
    expected_seq = 0
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
                    if encryption_key is None:
                        try:
                            base64.b64decode(line, validate=True)
                        except Exception:
                            return VerificationResult(
                                valid=False,
                                lines_checked=lines_checked,
                                first_error_line=line_num,
                                error_message=f"JSON parse error: {e}",
                            )
                        return VerificationResult(
                            valid=False,
                            lines_checked=lines_checked,
                            first_error_line=line_num,
                            error_message=(
                                f"Encrypted audit log detected at line {line_num}; "
                                "pass encryption_key to verify_log_integrity()"
                            ),
                        )

                    try:
                        decrypted = _decrypt_audit_line(
                            line, encryption_key, line_num=line_num
                        )
                        line_dict = json.loads(decrypted)
                    except Exception as exc:
                        return VerificationResult(
                            valid=False,
                            lines_checked=lines_checked,
                            first_error_line=line_num,
                            error_message=str(exc),
                        )

                # Verify _prev_hash matches expected previous hash
                actual_prev_hash = line_dict.get("_prev_hash")
                if not isinstance(actual_prev_hash, str) or not hmac.compare_digest(
                    actual_prev_hash, prev_hash
                ):
                    return VerificationResult(
                        valid=False,
                        lines_checked=lines_checked,
                        first_error_line=line_num,
                        error_message=f"Hash chain broken at line {line_num}: expected _prev_hash={prev_hash}, got {actual_prev_hash}",
                    )

                # Verify the sequence number starts at 0 and increments by exactly
                # 1. This binds each record to a position, so deleting records
                # from the tail (truncation) or reordering them is detectable even
                # though each record only carries the *previous* hash.
                actual_seq = line_dict.get("_seq")
                if actual_seq != expected_seq:
                    return VerificationResult(
                        valid=False,
                        lines_checked=lines_checked,
                        first_error_line=line_num,
                        error_message=(
                            f"Sequence broken at line {line_num}: "
                            f"expected _seq={expected_seq}, got {actual_seq}"
                        ),
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

                if not hmac.compare_digest(computed_hash, str(expected_hash)):
                    return VerificationResult(
                        valid=False,
                        lines_checked=lines_checked,
                        first_error_line=line_num,
                        error_message=f"HMAC mismatch at line {line_num}: expected {expected_hash}, computed {computed_hash}",
                    )

                prev_hash = str(expected_hash)
                expected_seq += 1

    except Exception as e:
        return VerificationResult(
            valid=False,
            lines_checked=lines_checked,
            error_message=f"Verification error: {e}",
        )

    # Optional anchor: detect tail-truncation (or unexpected extra records)
    # when the caller knows how many records should be present.
    if expected_records is not None and lines_checked != expected_records:
        return VerificationResult(
            valid=False,
            lines_checked=lines_checked,
            error_message=(
                f"Record count mismatch: expected {expected_records}, "
                f"found {lines_checked} (log may have been tail-truncated)"
            ),
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


__all__ = [
    "VerificationResult",
    "AuditEvent",
    "AuditLogger",
    "verify_log_integrity",
    "decrypt_log",
]
