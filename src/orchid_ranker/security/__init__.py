"""Security utilities: access-control policies, audit logging, and authentication."""

from .access import AccessControl, DEFAULT_POLICY
from .audit import (
    AuditLogger,
    AuditEvent,
    VerificationResult,
    verify_log_integrity,
    decrypt_log,
)
from .auth import (
    JWTAuthenticator,
    TokenPayload,
    AuthenticationError,
    TokenExpiredError,
    InvalidTokenError,
    InsufficientPermissionError,
)

__all__ = [
    "AccessControl",
    "DEFAULT_POLICY",
    "AuditLogger",
    "AuditEvent",
    "VerificationResult",
    "verify_log_integrity",
    "decrypt_log",
    "JWTAuthenticator",
    "TokenPayload",
    "AuthenticationError",
    "TokenExpiredError",
    "InvalidTokenError",
    "InsufficientPermissionError",
]
