"""Enterprise security tests for audit logging, JWT auth, RBAC, and SQL injection heuristics.

Covers:
- AuditLogger HTTPS enforcement and env-var override
- HMAC hash-chain integrity and tamper detection
- Audit thread safety under concurrent writes
- Fernet encryption key validation and roundtrip
- Fernet key encoding via from_env classmethod
- JWT algorithm confusion prevention
- JWT empty algorithms rejection
- RBAC wildcard permission warnings
- SnowflakeConnector SQL injection heuristic logging
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import pytest

from orchid_ranker.security.audit import (
    AuditEvent,
    AuditLogger,
    decrypt_log,
    verify_log_integrity,
)
from orchid_ranker.security.auth import JWTAuthenticator
from orchid_ranker.security.access import AccessControl
from orchid_ranker.connectors.snowflake import SnowflakeConnector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    from cryptography.fernet import Fernet

    _HAS_FERNET = True
except ImportError:
    _HAS_FERNET = False

skip_no_fernet = pytest.mark.skipif(
    not _HAS_FERNET, reason="cryptography library not installed"
)

try:
    import jwt as _jwt  # noqa: F401

    _HAS_JWT = True
except ImportError:
    _HAS_JWT = False

skip_no_jwt = pytest.mark.skipif(
    not _HAS_JWT, reason="PyJWT library not installed"
)

HMAC_KEY = bytes.fromhex("deadbeef" * 8)  # 32-byte HMAC key for tests


# ═══════════════════════════════════════════════════════════════════════════
# 1. Audit HTTPS enforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditHTTPSEnforcement:
    """AuditLogger must reject plain-HTTP endpoints unless explicitly allowed."""

    def test_http_endpoint_raises_by_default(self, tmp_path: Path, monkeypatch):
        """An http:// audit endpoint must raise ValueError when env var is not set."""
        monkeypatch.delenv("ORCHID_AUDIT_ALLOW_HTTP", raising=False)
        with pytest.raises(ValueError, match="does not use HTTPS"):
            AuditLogger(
                tmp_path / "audit.jsonl",
                endpoint="http://insecure.example.com/logs",
            )

    def test_https_endpoint_accepted(self, tmp_path: Path, monkeypatch):
        """An https:// endpoint should be accepted without error."""
        monkeypatch.delenv("ORCHID_AUDIT_ALLOW_HTTP", raising=False)
        logger = AuditLogger(
            tmp_path / "audit.jsonl",
            endpoint="https://secure.example.com/logs",
        )
        assert logger.endpoint == "https://secure.example.com/logs"

    def test_http_allowed_with_env_var(self, tmp_path: Path, monkeypatch):
        """When ORCHID_AUDIT_ALLOW_HTTP=1, http:// endpoints are permitted."""
        monkeypatch.setenv("ORCHID_AUDIT_ALLOW_HTTP", "1")
        logger = AuditLogger(
            tmp_path / "audit.jsonl",
            endpoint="http://localhost:9200/logs",
        )
        assert logger.endpoint == "http://localhost:9200/logs"

    def test_http_allowed_env_var_logs_warning(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """When http is force-allowed, a warning must be logged."""
        monkeypatch.setenv("ORCHID_AUDIT_ALLOW_HTTP", "1")
        with caplog.at_level(logging.WARNING):
            AuditLogger(
                tmp_path / "audit.jsonl",
                endpoint="http://localhost:9200/logs",
            )
        assert "does not use HTTPS" in caplog.text


# ═══════════════════════════════════════════════════════════════════════════
# 2. Audit HMAC hash-chain integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditHashChain:
    """HMAC hash-chain verification must detect any tampering."""

    def _write_events(self, log_path: Path, n: int = 4) -> AuditLogger:
        logger = AuditLogger(log_path, hmac_key=HMAC_KEY)
        for i in range(n):
            logger.log(
                AuditEvent(
                    event_type="test.action",
                    actor=f"user_{i}",
                    payload={"seq": i},
                )
            )
        return logger

    def test_integrity_passes_on_clean_log(self, tmp_path: Path):
        """verify_log_integrity returns valid=True for an untampered log."""
        log_path = tmp_path / "audit.jsonl"
        self._write_events(log_path, n=4)
        result = verify_log_integrity(log_path, HMAC_KEY)
        assert result.valid is True
        assert result.lines_checked == 4

    def test_integrity_detects_content_tamper(self, tmp_path: Path):
        """Changing event content must cause verification to fail."""
        log_path = tmp_path / "audit.jsonl"
        self._write_events(log_path, n=4)

        lines = log_path.read_text().splitlines()
        # Tamper with the second line's actor field
        record = json.loads(lines[1])
        record["actor"] = "EVIL_ACTOR"
        lines[1] = json.dumps(record, separators=(",", ":"), default=str)
        log_path.write_text("\n".join(lines) + "\n")

        result = verify_log_integrity(log_path, HMAC_KEY)
        assert result.valid is False
        assert result.first_error_line is not None

    def test_integrity_detects_deleted_line(self, tmp_path: Path):
        """Removing a line must break the hash chain."""
        log_path = tmp_path / "audit.jsonl"
        self._write_events(log_path, n=4)

        lines = log_path.read_text().splitlines()
        # Remove the second line so line 3's _prev_hash is wrong
        del lines[1]
        log_path.write_text("\n".join(lines) + "\n")

        result = verify_log_integrity(log_path, HMAC_KEY)
        assert result.valid is False

    def test_integrity_detects_wrong_key(self, tmp_path: Path):
        """Verifying with a different HMAC key must fail."""
        log_path = tmp_path / "audit.jsonl"
        self._write_events(log_path, n=3)

        wrong_key = bytes.fromhex("cafebabe" * 8)
        result = verify_log_integrity(log_path, wrong_key)
        assert result.valid is False


class TestEncryptedAuditIntegrity:
    """Encrypted audit logs must verify with the decryption key."""

    @skip_no_fernet
    def test_integrity_requires_encryption_key(self, tmp_path: Path):
        """An encrypted log without the key must fail explicitly."""
        key = Fernet.generate_key()
        log_path = tmp_path / "audit_enc.jsonl"
        logger = AuditLogger(log_path, hmac_key=HMAC_KEY, encryption_key=key)
        logger.log(
            AuditEvent(
                event_type="encrypted.test",
                actor="alice",
                payload={"seq": 1},
            )
        )

        result = verify_log_integrity(log_path, HMAC_KEY)
        assert result.valid is False
        assert result.error_message is not None
        assert "encrypted audit log" in result.error_message.lower()

    @skip_no_fernet
    def test_integrity_verifies_encrypted_log_with_key(self, tmp_path: Path):
        """Providing the encryption key must allow full integrity verification."""
        key = Fernet.generate_key()
        log_path = tmp_path / "audit_enc.jsonl"
        logger = AuditLogger(log_path, hmac_key=HMAC_KEY, encryption_key=key)
        for i in range(3):
            logger.log(
                AuditEvent(
                    event_type="encrypted.test",
                    actor=f"user_{i}",
                    payload={"seq": i},
                )
            )

        result = verify_log_integrity(
            log_path,
            HMAC_KEY,
            encryption_key=key,
        )
        assert result.valid is True
        assert result.lines_checked == 3


# ═══════════════════════════════════════════════════════════════════════════
# 3. Audit thread safety
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditThreadSafety:
    """Concurrent writes must not corrupt the audit log or the hash chain."""

    def test_concurrent_writes_produce_valid_chain(self, tmp_path: Path):
        """Two threads writing events concurrently must still yield a valid chain."""
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path, hmac_key=HMAC_KEY)
        events_per_thread = 20

        def writer(thread_id: int):
            for i in range(events_per_thread):
                logger.log(
                    AuditEvent(
                        event_type="concurrent.write",
                        actor=f"thread_{thread_id}",
                        payload={"seq": i},
                    )
                )

        t1 = threading.Thread(target=writer, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # All events must be present
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == events_per_thread * 2

        # Hash chain must still verify
        result = verify_log_integrity(log_path, HMAC_KEY)
        assert result.valid is True
        assert result.lines_checked == events_per_thread * 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. Fernet encryption key validation
# ═══════════════════════════════════════════════════════════════════════════


class TestFernetKeyValidation:
    """AuditLogger must reject bad Fernet keys early."""

    @skip_no_fernet
    def test_invalid_fernet_key_raises_valueerror(self, tmp_path: Path):
        """A garbage string as encryption_key must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Fernet encryption key"):
            AuditLogger(
                tmp_path / "audit.jsonl",
                encryption_key=b"not-a-valid-fernet-key",
            )

    @skip_no_fernet
    def test_valid_fernet_key_accepted(self, tmp_path: Path):
        """A properly generated Fernet key must be accepted."""
        key = Fernet.generate_key()
        logger = AuditLogger(tmp_path / "audit.jsonl", encryption_key=key)
        assert logger.encryption_key == key

    @skip_no_fernet
    def test_empty_fernet_key_raises(self, tmp_path: Path):
        """An empty bytes object is not a valid Fernet key."""
        with pytest.raises(ValueError, match="Invalid Fernet encryption key"):
            AuditLogger(tmp_path / "audit.jsonl", encryption_key=b"")


# ═══════════════════════════════════════════════════════════════════════════
# 5. JWT algorithm confusion
# ═══════════════════════════════════════════════════════════════════════════


class TestJWTAlgorithmConfusion:
    """JWTAuthenticator must reject mixed symmetric/asymmetric algorithms."""

    @skip_no_jwt
    def test_mixing_hs256_and_rs256_raises(self):
        """Specifying both HS256 and RS256 must raise ValueError."""
        with pytest.raises(ValueError, match="Cannot mix symmetric"):
            JWTAuthenticator(
                issuer="https://example.com",
                audience="test-api",
                algorithms=["HS256", "RS256"],
            )

    @skip_no_jwt
    def test_mixing_hs512_and_es256_raises(self):
        """Specifying both HS512 and ES256 must raise ValueError."""
        with pytest.raises(ValueError, match="Cannot mix symmetric"):
            JWTAuthenticator(
                issuer="https://example.com",
                audience="test-api",
                algorithms=["HS512", "ES256"],
            )

    @skip_no_jwt
    def test_single_symmetric_algorithm_accepted(self):
        """A single symmetric algorithm should be accepted."""
        # The constructor will try to build a JWKSCache; that's OK for this
        # test -- we only care that the algorithms check passes.
        auth = JWTAuthenticator(
            issuer="https://example.com",
            audience="test-api",
            algorithms=["HS256"],
        )
        assert auth.algorithms == ["HS256"]

    @skip_no_jwt
    def test_multiple_asymmetric_algorithms_accepted(self):
        """Multiple asymmetric algorithms (e.g. RS256 + ES256) should be accepted."""
        auth = JWTAuthenticator(
            issuer="https://example.com",
            audience="test-api",
            algorithms=["RS256", "ES256"],
        )
        assert set(auth.algorithms) == {"RS256", "ES256"}


# ═══════════════════════════════════════════════════════════════════════════
# 6. JWT empty algorithms
# ═══════════════════════════════════════════════════════════════════════════


class TestJWTEmptyAlgorithms:
    """JWTAuthenticator must reject an empty algorithms list."""

    @skip_no_jwt
    def test_empty_algorithms_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            JWTAuthenticator(
                issuer="https://example.com",
                audience="test-api",
                algorithms=[],
            )


# ═══════════════════════════════════════════════════════════════════════════
# 7. JWT HTTPS transport enforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestJWTHttpsTransport:
    """JWTAuthenticator must reject insecure issuer and JWKS transport."""

    @skip_no_jwt
    def test_http_issuer_rejected(self):
        """An http:// issuer must raise ValueError."""
        with pytest.raises(ValueError, match="issuer must use https://"):
            JWTAuthenticator(
                issuer="http://example.com",
                audience="test-api",
            )

    @skip_no_jwt
    def test_http_jwks_uri_rejected(self):
        """An http:// jwks_uri must raise ValueError."""
        with pytest.raises(ValueError, match="jwks_uri must use https://"):
            JWTAuthenticator(
                issuer="https://example.com",
                audience="test-api",
                jwks_uri="http://example.com/.well-known/jwks.json",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 8. RBAC wildcard warning
# ═══════════════════════════════════════════════════════════════════════════


class TestRBACWildcard:
    """AccessControl.can() must warn when a role has wildcard ('*') permission."""

    def test_wildcard_permission_logs_warning(self, caplog):
        """A role with '*' permission should trigger a warning log."""
        ac = AccessControl(policy={"superuser": {"*"}})
        with caplog.at_level(logging.WARNING):
            result = ac.can("superuser", "anything")
        assert result is True
        assert "wildcard" in caplog.text.lower()

    def test_wildcard_grants_any_action(self):
        """Wildcard must grant access to arbitrary actions."""
        ac = AccessControl(policy={"superuser": {"*"}})
        assert ac.can("superuser", "preprocess") is True
        assert ac.can("superuser", "launch_missiles") is True

    def test_non_wildcard_role_no_warning(self, caplog):
        """A role without wildcard should not trigger a wildcard warning."""
        ac = AccessControl(policy={"viewer": {"read_logs"}})
        with caplog.at_level(logging.WARNING):
            ac.can("viewer", "read_logs")
        assert "wildcard" not in caplog.text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 9. SQL injection heuristic
# ═══════════════════════════════════════════════════════════════════════════


class TestSQLInjectionHeuristic:
    """SnowflakeConnector._check_sql_injection must warn on suspicious SQL."""

    def test_drop_table_pattern(self, caplog):
        """Semicolon + DROP TABLE pattern should trigger a warning."""
        with caplog.at_level(logging.WARNING):
            SnowflakeConnector._check_sql_injection("SELECT 1; DROP TABLE users;--")
        assert "injection" in caplog.text.lower()

    def test_union_select_pattern(self, caplog):
        """UNION SELECT pattern should trigger a warning."""
        with caplog.at_level(logging.WARNING):
            SnowflakeConnector._check_sql_injection(
                "SELECT id FROM users WHERE name='x' UNION SELECT password FROM secrets"
            )
        assert "injection" in caplog.text.lower()

    def test_tautology_pattern(self, caplog):
        """1=1 tautology pattern should trigger a warning."""
        with caplog.at_level(logging.WARNING):
            SnowflakeConnector._check_sql_injection(
                "SELECT * FROM users WHERE 1=1"
            )
        assert "injection" in caplog.text.lower()

    def test_comment_injection_pattern(self, caplog):
        """SQL comment (--) pattern should trigger a warning."""
        with caplog.at_level(logging.WARNING):
            SnowflakeConnector._check_sql_injection(
                "SELECT * FROM users WHERE name='admin'--'"
            )
        assert "injection" in caplog.text.lower()

    def test_clean_query_no_warning(self, caplog):
        """A normal parameterized-style query should not trigger a warning."""
        with caplog.at_level(logging.WARNING):
            SnowflakeConnector._check_sql_injection(
                "SELECT id, name FROM users WHERE org_id = %s"
            )
        assert "injection" not in caplog.text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 10. Encryption / decryption roundtrip
# ═══════════════════════════════════════════════════════════════════════════


class TestEncryptionRoundtrip:
    """Encrypted audit logs must be decryptable back to their original content."""

    @skip_no_fernet
    def test_encrypt_then_decrypt(self, tmp_path: Path):
        """Events written with encryption must survive a full encrypt-decrypt cycle."""
        key = Fernet.generate_key()
        log_path = tmp_path / "audit_enc.jsonl"
        logger = AuditLogger(log_path, encryption_key=key)

        events = [
            AuditEvent(event_type="login", actor="alice", payload={"ip": "10.0.0.1"}),
            AuditEvent(event_type="export", actor="bob", payload={"rows": 42}),
            AuditEvent(event_type="logout", actor="alice"),
        ]
        for evt in events:
            logger.log(evt)

        decrypted_lines = decrypt_log(log_path, key)
        assert len(decrypted_lines) == 3

        for i, line in enumerate(decrypted_lines):
            record = json.loads(line)
            assert record["event_type"] == events[i].event_type
            assert record["actor"] == events[i].actor

    @skip_no_fernet
    def test_decrypt_with_wrong_key_fails(self, tmp_path: Path):
        """Decrypting with the wrong Fernet key must raise an error."""
        key = Fernet.generate_key()
        wrong_key = Fernet.generate_key()
        log_path = tmp_path / "audit_enc.jsonl"

        logger = AuditLogger(log_path, encryption_key=key)
        logger.log(AuditEvent(event_type="test", actor="user"))

        from cryptography.fernet import InvalidToken

        with pytest.raises(InvalidToken):
            decrypt_log(log_path, wrong_key)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Fernet key encoding via from_env
# ═══════════════════════════════════════════════════════════════════════════


class TestFernetKeyFromEnv:
    """AuditLogger.from_env must encode the encryption key with ascii."""

    @skip_no_fernet
    def test_from_env_uses_ascii_encoding(self, tmp_path: Path, monkeypatch):
        """The from_env classmethod must encode ORCHID_AUDIT_ENCRYPTION_KEY as ascii."""
        key = Fernet.generate_key()  # bytes
        key_str = key.decode("ascii")

        monkeypatch.setenv("ORCHID_AUDIT_ENCRYPTION_KEY", key_str)
        # Clear other env vars so they don't interfere
        monkeypatch.delenv("ORCHID_AUDIT_HMAC_KEY", raising=False)
        monkeypatch.delenv("ORCHID_AUDIT_ENDPOINT", raising=False)
        monkeypatch.delenv("ORCHID_AUDIT_API_KEY", raising=False)

        logger = AuditLogger.from_env(tmp_path / "audit.jsonl")
        assert logger.encryption_key == key_str.encode("ascii")
        assert logger.encryption_key == key

    @skip_no_fernet
    def test_from_env_roundtrip_with_encryption(self, tmp_path: Path, monkeypatch):
        """Events written via from_env with encryption can be decrypted."""
        key = Fernet.generate_key()
        key_str = key.decode("ascii")
        log_path = tmp_path / "audit.jsonl"

        monkeypatch.setenv("ORCHID_AUDIT_ENCRYPTION_KEY", key_str)
        monkeypatch.delenv("ORCHID_AUDIT_HMAC_KEY", raising=False)
        monkeypatch.delenv("ORCHID_AUDIT_ENDPOINT", raising=False)
        monkeypatch.delenv("ORCHID_AUDIT_API_KEY", raising=False)

        logger = AuditLogger.from_env(log_path)
        logger.log(AuditEvent(event_type="env_test", actor="ci"))

        decrypted = decrypt_log(log_path, key)
        assert len(decrypted) == 1
        record = json.loads(decrypted[0])
        assert record["event_type"] == "env_test"
        assert record["actor"] == "ci"
