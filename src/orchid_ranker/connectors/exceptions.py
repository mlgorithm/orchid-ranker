"""Connector exception hierarchy."""


class ConnectorError(Exception):
    """Base exception for all connector errors."""
    pass


class ConnectionTimeoutError(ConnectorError):
    """Raised when a connection attempt times out."""
    pass


class RetryExhaustedError(ConnectorError):
    """Raised when all retry attempts are exhausted."""
    pass


__all__ = [
    "ConnectorError",
    "ConnectionTimeoutError",
    "RetryExhaustedError",
]
