"""Snowflake connector utilities (optional dependency)."""
from __future__ import annotations

import logging
import os
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

    Secure credential management: Use `from_env()` or `from_vault()` classmethods
    to load credentials from environment variables or secret vaults rather than
    passing plaintext passwords. This prevents credentials from being exposed in
    logs, tracebacks, or source code.

    Parameters
    ----------
    account : str
        Snowflake account name (e.g., "xy12345.us-east-1").
    user : str
        Snowflake username.
    password : str
        Snowflake password. Prefer loading via `from_env()` or `from_vault()`.
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
        Snowflake password (masked in __repr__).
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

    Examples
    --------
    Load from environment variables:
        >>> conn = SnowflakeConnector.from_env()

    Load from a vault (HashiCorp Vault or AWS Secrets Manager):
        >>> conn = SnowflakeConnector.from_vault(vault_client, "snowflake/prod")
    """

    account: str
    user: str
    password: str
    warehouse: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30

    def __repr__(self) -> str:
        """Return a string representation with password masked for security.

        Returns
        -------
        str
            Representation with password shown as '****'.
        """
        return (
            f"SnowflakeConnector(account={self.account!r}, user={self.user!r}, "
            f"password='****', warehouse={self.warehouse!r}, database={self.database!r}, "
            f"schema={self.schema!r}, max_retries={self.max_retries}, timeout={self.timeout})"
        )

    @classmethod
    def from_env(cls, prefix: str = "ORCHID_SNOWFLAKE") -> SnowflakeConnector:
        """Create a SnowflakeConnector from environment variables.

        Reads credentials and configuration from environment variables with the
        given prefix. Required variables: {prefix}_ACCOUNT, {prefix}_USER,
        {prefix}_PASSWORD.

        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix (default: "ORCHID_SNOWFLAKE").
            For example, with prefix="ORCHID_SNOWFLAKE", expects:
            - ORCHID_SNOWFLAKE_ACCOUNT
            - ORCHID_SNOWFLAKE_USER
            - ORCHID_SNOWFLAKE_PASSWORD
            - ORCHID_SNOWFLAKE_WAREHOUSE (optional)
            - ORCHID_SNOWFLAKE_DATABASE (optional)
            - ORCHID_SNOWFLAKE_SCHEMA (optional)

        Returns
        -------
        SnowflakeConnector
            Configured connector instance.

        Raises
        ------
        EnvironmentError
            If required environment variables are missing.
        """
        account = os.environ.get(f"{prefix}_ACCOUNT")
        user = os.environ.get(f"{prefix}_USER")
        password = os.environ.get(f"{prefix}_PASSWORD")
        warehouse = os.environ.get(f"{prefix}_WAREHOUSE")
        database = os.environ.get(f"{prefix}_DATABASE")
        schema = os.environ.get(f"{prefix}_SCHEMA")

        missing = []
        if not account:
            missing.append(f"{prefix}_ACCOUNT")
        if not user:
            missing.append(f"{prefix}_USER")
        if not password:
            missing.append(f"{prefix}_PASSWORD")

        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return cls(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema,
        )

    @classmethod
    def from_vault(
        cls,
        vault_client: Any,
        secret_path: str,
        *,
        account_key: str = "account",
        user_key: str = "user",
        password_key: str = "password",
        warehouse_key: str = "warehouse",
        database_key: str = "database",
        schema_key: str = "schema",
    ) -> SnowflakeConnector:
        """Create a SnowflakeConnector from a vault client.

        Supports multiple vault backends:
        - HashiCorp Vault: calls `vault_client.read_secret(secret_path)`
        - AWS Secrets Manager: calls `vault_client.get_secret_value(SecretId=secret_path)`

        Parameters
        ----------
        vault_client : Any
            Vault client instance (e.g., hvac.Client or boto3 SecretsManager client).
        secret_path : str
            Path or SecretId for the secret in the vault.
        account_key : str, optional
            Key for account field in secret (default: "account").
        user_key : str, optional
            Key for user field in secret (default: "user").
        password_key : str, optional
            Key for password field in secret (default: "password").
        warehouse_key : str, optional
            Key for warehouse field in secret (default: "warehouse").
        database_key : str, optional
            Key for database field in secret (default: "database").
        schema_key : str, optional
            Key for schema field in secret (default: "schema").

        Returns
        -------
        SnowflakeConnector
            Configured connector instance.

        Raises
        ------
        ValueError
            If secret cannot be read or required keys are missing.

        Examples
        --------
        With HashiCorp Vault:
            >>> import hvac
            >>> vault = hvac.Client(url="http://vault:8200", token="...")
            >>> conn = SnowflakeConnector.from_vault(vault, "secret/snowflake/prod")

        With AWS Secrets Manager:
            >>> import boto3
            >>> sm = boto3.client("secretsmanager")
            >>> conn = SnowflakeConnector.from_vault(sm, "snowflake/prod")
        """
        try:
            # Try HashiCorp Vault pattern first
            if hasattr(vault_client, "read_secret"):
                secret_data = vault_client.read_secret(secret_path)
                if isinstance(secret_data, dict) and "data" in secret_data:
                    # Standard Vault response structure
                    secret_dict = secret_data.get("data", {})
                else:
                    secret_dict = secret_data
            # Fall back to AWS Secrets Manager pattern
            elif hasattr(vault_client, "get_secret_value"):
                response = vault_client.get_secret_value(SecretId=secret_path)
                if "SecretString" in response:
                    import json

                    secret_dict = json.loads(response["SecretString"])
                else:
                    secret_dict = response
            else:
                raise ValueError(
                    "vault_client must implement read_secret() (Vault) or "
                    "get_secret_value() (AWS Secrets Manager)"
                )
        except Exception as e:
            raise ValueError(f"Failed to read secret from vault: {e}") from e

        # Extract keys with defaults for optional fields
        account = secret_dict.get(account_key)
        user = secret_dict.get(user_key)
        password = secret_dict.get(password_key)
        warehouse = secret_dict.get(warehouse_key)
        database = secret_dict.get(database_key)
        schema = secret_dict.get(schema_key)

        # Check required fields
        missing = []
        if not account:
            missing.append(account_key)
        if not user:
            missing.append(user_key)
        if not password:
            missing.append(password_key)

        if missing:
            raise ValueError(
                f"Missing required keys in vault secret: {', '.join(missing)}"
            )

        return cls(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema,
        )

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


__all__ = ["SnowflakeConnector"]
