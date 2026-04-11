"""S3 streaming connector for ingest/egress."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from .exceptions import ConnectorError, RetryExhaustedError


try:  # pragma: no cover - optional import
    import boto3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    boto3 = None


logger = logging.getLogger(__name__)


@dataclass
class S3StreamConnector:
    """Connector for streaming data from Amazon S3 with retry support.

    Provides methods to list and stream objects from S3 buckets with automatic
    exponential backoff retry logic for transient errors.
    Requires boto3 library (optional dependency).

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str, optional
        S3 key prefix to filter objects (default: "").
    profile : str, optional
        AWS profile name from credentials. If None, uses default profile.
    region : str, optional
        AWS region name. If None, uses default region from credentials.
    max_retries : int, optional
        Maximum number of retry attempts for transient errors (default: 3).
    timeout : int, optional
        Operation timeout in seconds (default: 30, not currently enforced).

    Attributes
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        S3 key prefix.
    profile : str, optional
        AWS profile name.
    region : str, optional
        AWS region name.
    max_retries : int
        Max retry attempts.
    timeout : int
        Operation timeout.

    Examples
    --------
    Load from environment variables:
        >>> conn = S3StreamConnector.from_env()
    """

    bucket: str
    prefix: str = ""
    profile: Optional[str] = None
    region: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30

    @classmethod
    def from_env(cls, prefix: str = "ORCHID_S3") -> S3StreamConnector:
        """Create an S3StreamConnector from environment variables.

        Reads configuration from environment variables with the given prefix.
        Only the bucket is required; other variables are optional.

        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix (default: "ORCHID_S3").
            For example, with prefix="ORCHID_S3", expects:
            - ORCHID_S3_BUCKET (required)
            - ORCHID_S3_PREFIX (optional, default: "")
            - ORCHID_S3_PROFILE (optional)
            - ORCHID_S3_REGION (optional)

        Returns
        -------
        S3StreamConnector
            Configured connector instance.

        Raises
        ------
        EnvironmentError
            If ORCHID_S3_BUCKET is not set.
        """
        bucket = os.environ.get(f"{prefix}_BUCKET")
        if not bucket:
            raise EnvironmentError(f"Missing required environment variable: {prefix}_BUCKET")

        prefix_value = os.environ.get(f"{prefix}_PREFIX", "")
        profile = os.environ.get(f"{prefix}_PROFILE")
        region = os.environ.get(f"{prefix}_REGION")

        return cls(
            bucket=bucket,
            prefix=prefix_value,
            profile=profile,
            region=region,
        )

    def _client(self):
        """Get or create S3 client.

        Returns
        -------
        boto3.client
            S3 client with configured profile and region.

        Raises
        ------
        ImportError
            If boto3 is not installed.
        """
        if boto3 is None:  # pragma: no cover
            raise ImportError(
                "boto3 is required. Install via `pip install orchid-ranker[connectors]`"
            )
        session_kwargs = {}
        if self.profile:
            session_kwargs["profile_name"] = self.profile
        session = boto3.session.Session(**session_kwargs)
        return session.client("s3", region_name=self.region)

    def __enter__(self):
        """Context manager entry."""
        self._client_instance = self._client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the S3 client."""
        if hasattr(self, '_client_instance') and self._client_instance:
            try:
                self._client_instance.close()
                logger.info("S3 client closed")
            except Exception as e:
                logger.warning(f"Error closing S3 client: {e}")

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic for transient errors.

        Retries only on transient S3 errors (timeout, throttling, service unavailable).
        Non-transient errors are raised immediately.

        Parameters
        ----------
        func : callable
            Function to execute.
        *args
            Positional arguments for func.
        **kwargs
            Keyword arguments for func.

        Returns
        -------
        Any
            Result from func.

        Raises
        ------
        ConnectorError
            On non-transient S3 errors.
        RetryExhaustedError
            If all retry attempts are exhausted for transient errors.
        """
        delays = [1, 2, 4]
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                # Only retry on transient errors
                error_msg = str(e).lower()
                if any(x in error_msg for x in ['timeout', 'throttling', 'serviceunava']):
                    if attempt < self.max_retries - 1:
                        delay = delays[attempt]
                        logger.warning(
                            f"Transient S3 error (attempt {attempt + 1}), retrying in {delay}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {self.max_retries} retry attempts exhausted")
                else:
                    # Non-transient error, raise immediately
                    raise ConnectorError(f"S3 error: {e}") from e

        raise RetryExhaustedError(
            f"Failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def list_objects(self) -> Iterable[str]:
        """List all S3 object keys matching the bucket and prefix with retry support.

        Returns
        -------
        Iterable[str]
            Generator of S3 object keys.

        Raises
        ------
        ImportError
            If boto3 is not installed.
        ConnectorError
            On non-transient S3 errors.
        RetryExhaustedError
            If listing fails after all retry attempts.
        """
        # Check dependencies before entering retry loop
        self._client()

        def _list():
            client = self._client()
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                for item in page.get("Contents", []):
                    yield item["Key"]

        return self._retry_with_backoff(_list)

    def stream_object(self, key: str):
        """Stream lines from an S3 object with retry support.

        Parameters
        ----------
        key : str
            S3 object key to stream.

        Returns
        -------
        Iterable[bytes]
            Line-by-line stream from object body.

        Raises
        ------
        ImportError
            If boto3 is not installed.
        ConnectorError
            On non-transient S3 errors.
        RetryExhaustedError
            If streaming fails after all retry attempts.
        """
        # Check dependencies before entering retry loop
        self._client()

        def _stream():
            client = self._client()
            obj = client.get_object(Bucket=self.bucket, Key=key)
            return obj["Body"].iter_lines()

        return self._retry_with_backoff(_stream)


__all__ = ["S3StreamConnector"]
