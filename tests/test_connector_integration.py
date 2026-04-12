"""Connector integration tests with mocked external services.

Tests BigQuery and Snowflake connectors for:
- Client/connection caching and lifecycle
- Retry with exponential backoff and jitter
- Connection reuse across retry attempts
- SQL injection warning detection
- Non-transient error fast-fail (no retries)
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest

from orchid_ranker.connectors.bigquery import BigQueryConnector
from orchid_ranker.connectors.snowflake import SnowflakeConnector
from orchid_ranker.connectors.exceptions import RetryExhaustedError


# ---------------------------------------------------------------------------
# BigQuery: client caching
# ---------------------------------------------------------------------------

class TestBigQueryClientCaching:
    """Verify _client() returns the same object on repeated calls and close() resets it."""

    @patch("orchid_ranker.connectors.bigquery.bigquery")
    def test_client_returns_same_instance(self, mock_bq_module) -> None:
        """Repeated _client() calls should return the same cached object."""
        mock_client_instance = MagicMock()
        mock_bq_module.Client.return_value = mock_client_instance

        conn = BigQueryConnector(project="test-project")
        first = conn._client()
        second = conn._client()

        assert first is second, "_client() should return the same cached instance"
        # The Client constructor should have been called exactly once
        mock_bq_module.Client.assert_called_once_with(project="test-project")

    @patch("orchid_ranker.connectors.bigquery.bigquery")
    def test_close_resets_cached_client(self, mock_bq_module) -> None:
        """close() should set _cached_client to None, forcing re-creation."""
        mock_client_instance = MagicMock()
        mock_bq_module.Client.return_value = mock_client_instance

        conn = BigQueryConnector(project="test-project")
        conn._client()
        assert conn._cached_client is not None

        conn.close()
        assert conn._cached_client is None
        mock_client_instance.close.assert_called_once()


# ---------------------------------------------------------------------------
# BigQuery: retry with jitter
# ---------------------------------------------------------------------------

class TestBigQueryRetryWithJitter:
    """Verify retry logic: fail twice, succeed on third attempt, with jittered delays."""

    @patch("orchid_ranker.connectors.bigquery.time.sleep")
    @patch("orchid_ranker.connectors.bigquery.bigquery")
    def test_retry_succeeds_on_third_attempt(
        self, mock_bq_module, mock_sleep
    ) -> None:
        """Query fails twice with a transient error, then succeeds on attempt 3."""
        mock_client = MagicMock()
        mock_bq_module.Client.return_value = mock_client

        # Set up the query job mock to fail twice then succeed
        mock_job = MagicMock()
        mock_df = MagicMock()
        mock_job.result.return_value.to_dataframe.return_value = mock_df

        call_count = 0

        def _query_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Service temporarily unavailable")
            return mock_job

        mock_client.query.side_effect = _query_side_effect

        conn = BigQueryConnector(project="test-project", max_retries=3)
        result = conn.query_dataframe("SELECT 1")

        assert result is mock_df
        assert call_count == 3, f"Expected 3 attempts, got {call_count}"
        # Two retries means two sleep calls with jittered delays
        assert mock_sleep.call_count == 2
        # Verify delays include jitter (base delays are 1 and 2)
        for i, call in enumerate(mock_sleep.call_args_list):
            delay = call[0][0]
            base = [1, 2][i]
            assert base <= delay <= base + 0.5, (
                f"Delay {delay} not in expected jitter range [{base}, {base + 0.5}]"
            )


# ---------------------------------------------------------------------------
# BigQuery: non-transient error (no retries)
# ---------------------------------------------------------------------------

class TestBigQueryNonTransientError:
    """Non-transient errors (e.g. TypeError) should raise immediately, no retries."""

    @patch("orchid_ranker.connectors.bigquery.time.sleep")
    @patch("orchid_ranker.connectors.bigquery.bigquery")
    def test_type_error_raises_immediately(
        self, mock_bq_module, mock_sleep
    ) -> None:
        """A TypeError from the query function should not be retried."""
        mock_client = MagicMock()
        mock_bq_module.Client.return_value = mock_client
        mock_client.query.side_effect = TypeError("bad argument type")

        conn = BigQueryConnector(project="test-project", max_retries=3)

        with pytest.raises(TypeError, match="bad argument type"):
            conn.query_dataframe("SELECT 1")

        # No sleep calls since error is non-transient
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Snowflake: connection reuse across retries
# ---------------------------------------------------------------------------

class TestSnowflakeConnectionReuse:
    """Verify that fetch_dataframe creates a single connection for all retry attempts."""

    @patch("orchid_ranker.connectors.snowflake.time.sleep")
    @patch("orchid_ranker.connectors.snowflake.snowflake")
    def test_single_connection_for_retries(
        self, mock_sf_module, mock_sleep
    ) -> None:
        """fetch_dataframe should open one connection and reuse it across retries."""
        mock_conn = MagicMock()
        mock_sf_module.connect.return_value = mock_conn

        # Cursor that fails once then succeeds
        call_count = 0

        def _execute_side_effect(query):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("Transient network error")
            # Success path: set up fetchall and description
            return None

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = _execute_side_effect
        mock_cursor.fetchall.return_value = [("row1",)]
        mock_cursor.description = [("col1", None, None, None, None, None, None)]
        mock_conn.cursor.return_value = mock_cursor

        connector = SnowflakeConnector(
            account="test-account",
            user="test-user",
            password="test-pass",
            max_retries=3,
        )

        # pandas is imported locally inside fetch_dataframe; patch it at the
        # module level so the local import picks up the mock.
        mock_pd = MagicMock()
        with patch.dict("sys.modules", {"pandas": mock_pd}):
            connector.fetch_dataframe("SELECT 1")

        # Only one connection should have been created
        mock_sf_module.connect.assert_called_once()
        # Connection should be closed after the operation
        mock_conn.close.assert_called()


# ---------------------------------------------------------------------------
# Snowflake: SQL injection warning
# ---------------------------------------------------------------------------

class TestSnowflakeSQLInjectionWarning:
    """Verify that suspicious queries trigger log warnings."""

    @patch("orchid_ranker.connectors.snowflake.snowflake")
    def test_sql_injection_pattern_logs_warning(
        self, mock_sf_module, caplog
    ) -> None:
        """A query containing '1=1' should trigger a SQL injection warning."""
        mock_conn = MagicMock()
        mock_sf_module.connect.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("row1",)]
        mock_cursor.description = [("col1", None, None, None, None, None, None)]
        mock_conn.cursor.return_value = mock_cursor

        connector = SnowflakeConnector(
            account="test-account",
            user="test-user",
            password="test-pass",
        )

        mock_pd = MagicMock()
        with patch.dict("sys.modules", {"pandas": mock_pd}):
            with caplog.at_level(logging.WARNING, logger="orchid_ranker.connectors.snowflake"):
                connector.fetch_dataframe("SELECT * FROM users WHERE 1=1")

        assert any(
            "SQL injection" in record.message for record in caplog.records
        ), (
            f"Expected a SQL injection warning in logs, got: "
            f"{[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# Snowflake: non-transient error (no retries)
# ---------------------------------------------------------------------------

class TestSnowflakeNonTransientError:
    """Non-transient errors (e.g. TypeError) should raise immediately, no retries."""

    @patch("orchid_ranker.connectors.snowflake.time.sleep")
    @patch("orchid_ranker.connectors.snowflake.snowflake")
    def test_type_error_raises_immediately(
        self, mock_sf_module, mock_sleep
    ) -> None:
        """A TypeError from the cursor should not be retried."""
        mock_conn = MagicMock()
        mock_sf_module.connect.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = TypeError("unsupported operand type")
        mock_conn.cursor.return_value = mock_cursor

        connector = SnowflakeConnector(
            account="test-account",
            user="test-user",
            password="test-pass",
            max_retries=3,
        )

        with pytest.raises(TypeError, match="unsupported operand type"):
            connector.fetch_dataframe("SELECT 1")

        # No sleep calls since error is non-transient
        mock_sleep.assert_not_called()
