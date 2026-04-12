"""JWT/OIDC authentication middleware for optional provider-agnostic auth."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urljoin, urlsplit

try:
    import jwt
    from jwt import PyJWKClient
    _HAS_JWT = True
except ImportError:
    _HAS_JWT = False

from .access import AccessControl


class AuthenticationError(Exception):
    """Base authentication error."""


class TokenExpiredError(AuthenticationError):
    """JWT token has expired."""


class InvalidTokenError(AuthenticationError):
    """JWT token is invalid (format, signature, claims)."""


class InsufficientPermissionError(AuthenticationError):
    """Authenticated principal lacks required permission."""


@dataclass
class TokenPayload:
    """Decoded JWT token payload with extracted metadata.

    Attributes:
        sub: Subject claim (usually user ID).
        role: Role claim extracted from token.
        claims: Full decoded JWT payload.
        exp: Token expiration timestamp (Unix seconds), or None if not present.
        iss: Issuer claim.
    """
    sub: str
    role: str
    claims: Dict[str, Any]
    exp: Optional[float]
    iss: str


class _JWKSCache:
    """Thread-safe JWKS key cache with TTL expiration."""

    def __init__(self, jwks_uri: str, cache_ttl: int = 3600):
        """Initialize cache.

        Args:
            jwks_uri: URI to fetch JWKS from.
            cache_ttl: Cache time-to-live in seconds.
        """
        self.jwks_uri = jwks_uri
        self.cache_ttl = cache_ttl
        self._client = PyJWKClient(self.jwks_uri)
        self._lock = threading.Lock()
        self._last_fetch: float = 0.0

    def get_signing_key(self, kid: str) -> Any:
        """Fetch and cache signing key by key ID.

        Args:
            kid: Key ID from JWT header.

        Returns:
            Signing key from JWKS.

        Raises:
            InvalidTokenError: If key not found in JWKS.
        """
        with self._lock:
            now = time.time()
            if now - self._last_fetch > self.cache_ttl:
                # Cache expired; refresh
                self._client.fetch_data()
                self._last_fetch = now

        try:
            return self._client.get_signing_key(kid)
        except Exception as e:
            raise InvalidTokenError(f"Could not fetch signing key '{kid}': {e}")


class JWTAuthenticator:
    """Provider-agnostic JWT/OIDC token authenticator.

    Decodes and validates JWT tokens from any OIDC-compliant provider
    (Okta, Auth0, Azure AD, Keycloak, etc.). Integrates with AccessControl
    to extract and enforce role-based permissions.

    Example:
        >>> auth = JWTAuthenticator(
        ...     issuer="https://example.okta.com",
        ...     audience="my-api"
        ... )
        >>> payload = auth.authenticate(token)
        >>> role = payload.role
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        role_claim: str = "role",
        jwks_uri: Optional[str] = None,
        algorithms: Sequence[str] = ("RS256",),
        cache_ttl: int = 3600,
    ):
        """Initialize authenticator with OIDC/JWT configuration.

        Args:
            issuer: Token issuer URL (e.g., https://example.okta.com).
            audience: Intended audience claim value.
            role_claim: JWT claim name containing user role (default: "role").
            jwks_uri: URI to fetch JWKS from. If None, derived from issuer
                      as issuer + "/.well-known/jwks.json" (OIDC standard).
            algorithms: Allowed token algorithms (default: ("RS256",)).
            cache_ttl: JWKS cache time-to-live in seconds (default: 3600).

        Raises:
            ImportError: If PyJWT is not installed.
        """
        if not _HAS_JWT:
            raise ImportError(
                "PyJWT is required for JWT authentication. "
                "Install it with: pip install 'orchid-ranker[auth]'"
            )

        self._require_https_uri(issuer, "issuer")
        self.issuer = issuer
        self.audience = audience
        self.role_claim = role_claim
        # Prevent algorithm confusion attacks: reject mixing symmetric
        # (HS*) with asymmetric (RS*/ES*/PS*) algorithms.
        _symmetric = {"HS256", "HS384", "HS512"}
        _asymmetric = {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512"}
        algs = set(algorithms)
        if not algs:
            raise ValueError("algorithms list must not be empty")
        if algs & _symmetric and algs & _asymmetric:
            raise ValueError(
                "Cannot mix symmetric (HS*) and asymmetric (RS*/ES*/PS*) "
                "algorithms — this enables algorithm confusion attacks. "
                f"Got: {list(algorithms)}"
            )
        self.algorithms = list(algorithms)

        # Derive JWKS URI from issuer if not provided (OIDC standard)
        if jwks_uri is None:
            jwks_uri = urljoin(issuer.rstrip("/") + "/", ".well-known/jwks.json")
        else:
            self._require_https_uri(jwks_uri, "jwks_uri")

        self._jwks_cache = _JWKSCache(jwks_uri, cache_ttl)

    @staticmethod
    def _require_https_uri(uri: str, label: str) -> None:
        """Reject non-HTTPS transport for issuer and JWKS discovery."""
        parsed = urlsplit(uri)
        if parsed.scheme.lower() != "https":
            raise ValueError(f"{label} must use https:// transport, got {uri!r}")

    def authenticate(self, token: str) -> TokenPayload:
        """Decode and validate JWT token.

        Validates issuer, audience, and signature using JWKS.

        Args:
            token: JWT token string.

        Returns:
            TokenPayload with extracted claims and metadata.

        Raises:
            TokenExpiredError: If token has expired.
            InvalidTokenError: If token is malformed or signature is invalid.
        """
        try:
            # Get kid from header to fetch the right key
            unverified = jwt.get_unverified_header(token)
            kid = unverified.get("kid")
            if not kid:
                raise InvalidTokenError("Token header missing 'kid' (key ID)")

            signing_key = self._jwks_cache.get_signing_key(kid)

            # Decode and validate
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
            )

        except jwt.ExpiredSignatureError as e:
            raise TokenExpiredError(f"Token has expired: {e}")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {e}")
        except Exception as e:
            raise InvalidTokenError(f"Failed to decode token: {e}")

        # Extract role claim
        role = payload.get(self.role_claim)
        if not role:
            raise InvalidTokenError(
                f"Token missing required claim '{self.role_claim}'"
            )

        # Extract subject and issuer
        sub = payload.get("sub")
        if not sub:
            raise InvalidTokenError("Token missing 'sub' claim")

        iss = payload.get("iss", self.issuer)

        return TokenPayload(
            sub=sub,
            role=str(role),
            claims=payload,
            exp=payload.get("exp"),
            iss=iss,
        )

    def get_role(self, token: str) -> str:
        """Decode token and return the role claim.

        Convenience method for when only the role is needed.

        Args:
            token: JWT token string.

        Returns:
            Role claim value.

        Raises:
            TokenExpiredError: If token has expired.
            InvalidTokenError: If token is invalid.
        """
        payload = self.authenticate(token)
        return payload.role

    def require(
        self, token: str, action: str, access_control: AccessControl
    ) -> TokenPayload:
        """Authenticate, extract role, and enforce permission.

        Decodes token, checks that the extracted role is permitted to
        perform the given action via AccessControl.

        Args:
            token: JWT token string.
            action: Action to authorize.
            access_control: AccessControl instance to check permissions.

        Returns:
            TokenPayload on successful authentication and authorization.

        Raises:
            TokenExpiredError: If token has expired.
            InvalidTokenError: If token is invalid.
            InsufficientPermissionError: If role lacks the required action.
        """
        payload = self.authenticate(token)
        try:
            access_control.require(payload.role, action)
        except PermissionError as e:
            raise InsufficientPermissionError(str(e))
        return payload


__all__ = [
    "JWTAuthenticator",
    "TokenPayload",
    "AuthenticationError",
    "TokenExpiredError",
    "InvalidTokenError",
    "InsufficientPermissionError",
]
