"""Snowflake connector utilities (optional dependency)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .exceptions import ConnectorError, ConnectionTimeoutError, RetryExhaustedError


try:  # pragma: no cover - optional import
    import snowflake.connector  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    snowflake = None
else:  # pragma: no cover - optional dependency
    snowflake = snowflake.connector


logger = logging.getLogger(__name__)


@dataclass
class SnowflakeConnector:
    """Connector for Snowflake data warehouse with Pandas integration and retry support.

    Provides methods to execute queries and fetch results as DataFrames with
    automatic exponential backoff retry logic.
    Requires snowflake-connector-python library (optional dependency).

    Parameters
    ----------
    account : str
        Snowflake account name (e.g., "xy12345.us-east-1").
    user : str
        Snowflake username.
    password : str
        Snowflake password.
    warehouse : str, optional
        Warehouse name. If None, uses default warehouse.
    database : str, optional
        Database name. If None, uses default database.
    schema : str, optional
        Schema name. If None, uses default schema.
    max_retries : int, optional
        Maximum number of retry attempts (default: 3).
    timeout : int, optional
        Login timeout in seconds (default: 30).

    Attributes
    ----------
    account : str
        Snowflake account identifier.
    user : str
        Snowflake username.
    password : str
        Snowflake password (should not be logged).
    warehouse : str, optional
        Warehouse name.
    database : str, optional
        Database name.
    schema : str, optional
        Schema name.
    max_retries : int
        Max retry attempts.
    timeout : int
        Login timeout.
    """

    account: str
    user: str
    password: str
    warehouse: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30

    def _require_lib(self):
        """Verify snowflake-connector-python is installed.

        Raises
        ------
        ImportError
            If snowflake-connector-python is not installed.
        """
        if snowflake is None:  # pragma: no cover - runtime path
            raise ImportError(
                "snowflake-connector-python is required. Install via `pip install orchid-ranker[connectors]`"
            )

    def _connect(self):
        """Create a Snowflake connection.

        Returns
        -------
        snowflake.connector.SnowflakeConnection
            Authenticated Snowflake connection.

        Raises
        ------
        ImportError
            If snowflake-connector-python is not installed.
        ConnectorError
            If connection fails.
        """
        self._require_lib()
        try:
            return snowflake.connect(  # type: ignore[union-attr]
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                login_timeout=self.timeout,
            )
        except Exception as e:
            raise ConnectorError(f"Failed to connect to Snowflake: {e}") from e

    def __enter__(self):
        """Context manager entry."""
        self._conn = self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if hasattr(self, '_conn') and self._conn:
            self.close()

    def close(self):
        """Close the Snowflake connection."""
        if hasattr(self, '_conn') and self._conn:
            try:
                self._conn.close()
                logger.info("Snowflake connection closed")
            except Exception as e:
                logger.warning(f"Error closing Snowflake connection: {e}")

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic.

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
        RetryExhaustedError
            If all retry attempts are exhausted.
        """
        delays = [1, 2, 4]
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = delays[attempt]
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} retry attempts exhausted")

        raise RetryExhaustedError(
            f"Failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def fetch_dataframe(self, query: str):
        """Execute a query and return results as a DataFrame with retry support.

        Parameters
        ----------
        query : str
            SQL query to execute.

        Returns
        -------
        pd.DataFrame
            Query results with columns matching Snowflake result set.

        Raises
        ------
        ImportError
            If snowflake-connector-python is not installed.
        RetryExhaustedError
            If query fails after all retry attempts.
        """
        import pandas as pd  # type: ignore

        self._require_lib()

        def _fetch():
            with self._connect() as conn:
                cur = conn.cursor()
                try:
                    cur.execute(query)
                    data = cur.fetchall()
                    columns = [col[0] for col in cur.description]
                    return pd.DataFrame(data, columns=columns)
                finally:
                    cur.close()

        return self._retry_with_backoff(_fetch)

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Execute a query without returning results with retry support.

        Useful for DDL/DML operations like CREATE TABLE, INSERT, etc.

        Parameters
        ----------
        query : str
            SQL query to execute.
        params : dict, optional
            Query parameters for parameterized queries.

        Raises
        ------
        ImportError
            If snowflake-connector-python is not installed.
        RetryExhaustedError
            If query fails after all retry attempts.
        """
        self._require_lib()

        def _execute():
            with self._connect() as conn:
                cur = conn.cursor()
                try:
                    cur.execute(query, params or {})
                finally:
                    cur.close()

        self._retry_with_backoff(_execute)
