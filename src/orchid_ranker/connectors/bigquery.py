"""BigQuery connector utilities (optional dependency)."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from .exceptions import ConnectorError, RetryExhaustedError


try:  # pragma: no cover - optional import
    from google.cloud import bigquery  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bigquery = None


logger = logging.getLogger(__name__)


@dataclass
class BigQueryConnector:
    """Connector for Google BigQuery data operations with retry support.

    Enables querying from and loading data to BigQuery tables with automatic
    exponential backoff retry logic for resilience.
    Requires google-cloud-bigquery library (optional dependency).

    Parameters
    ----------
    project : str, optional
        GCP project ID. If None, uses default project from credentials.
    dataset : str, optional
        Default dataset for load operations.
    max_retries : int, optional
        Maximum number of retry attempts (default: 3).
    timeout : int, optional
        Query/load timeout in seconds (default: 30).

    Attributes
    ----------
    project : str, optional
        GCP project ID.
    dataset : str, optional
        Default dataset.
    max_retries : int
        Max retry attempts.
    timeout : int
        Operation timeout.

    Examples
    --------
    Load from environment variables:
        >>> conn = BigQueryConnector.from_env()
    """

    project: Optional[str] = None
    dataset: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30

    @classmethod
    def from_env(cls, prefix: str = "ORCHID_BIGQUERY") -> BigQueryConnector:
        """Create a BigQueryConnector from environment variables.

        Reads configuration from environment variables with the given prefix.
        All variables are optional and can be None.

        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix (default: "ORCHID_BIGQUERY").
            For example, with prefix="ORCHID_BIGQUERY", expects:
            - ORCHID_BIGQUERY_PROJECT (optional)
            - ORCHID_BIGQUERY_DATASET (optional)

        Returns
        -------
        BigQueryConnector
            Configured connector instance.
        """
        project = os.environ.get(f"{prefix}_PROJECT")
        dataset = os.environ.get(f"{prefix}_DATASET")

        return cls(
            project=project,
            dataset=dataset,
        )

    def _client(self):
        """Get or create BigQuery client.

        Returns
        -------
        google.cloud.bigquery.Client
            Authenticated BigQuery client.

        Raises
        ------
        ImportError
            If google-cloud-bigquery is not installed.
        """
        if bigquery is None:  # pragma: no cover
            raise ImportError(
                "google-cloud-bigquery is required. Install via `pip install orchid-ranker[connectors]`"
            )
        return bigquery.Client(project=self.project)

    def __enter__(self):
        """Context manager entry."""
        self._client_instance = self._client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the client connection."""
        if hasattr(self, '_client_instance') and self._client_instance:
            try:
                self._client_instance.close()
                logger.info("BigQuery client closed")
            except Exception as e:
                logger.warning(f"Error closing BigQuery client: {e}")

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

    def query_dataframe(self, sql: str):
        """Execute a SQL query and return results as a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query string.

        Returns
        -------
        pd.DataFrame
            Query results with automatic retry on transient failures.

        Raises
        ------
        ImportError
            If google-cloud-bigquery is not installed.
        RetryExhaustedError
            If query fails after all retry attempts.
        """
        import pandas as pd  # type: ignore

        # Check dependencies before entering retry loop
        self._client()

        def _query():
            client = self._client()
            job = client.query(sql, timeout=self.timeout)
            return job.result().to_dataframe(create_bqstorage_client=False)

        return self._retry_with_backoff(_query)

    def load_dataframe(self, table: str, dataframe):
        """Load a DataFrame into a BigQuery table.

        Parameters
        ----------
        table : str
            Target table name (or "dataset.table" if dataset not set).
        dataframe : pd.DataFrame
            Data to load.

        Returns
        -------
        google.cloud.bigquery.LoadJob.Result
            Load job result with automatic retry on transient failures.

        Raises
        ------
        ImportError
            If google-cloud-bigquery is not installed.
        RetryExhaustedError
            If load fails after all retry attempts.
        """
        # Check dependencies before entering retry loop
        self._client()

        def _load():
            client = self._client()
            destination = f"{self.dataset}.{table}" if self.dataset else table
            job = client.load_table_from_dataframe(dataframe, destination)
            return job.result(timeout=self.timeout)

        return self._retry_with_backoff(_load)
