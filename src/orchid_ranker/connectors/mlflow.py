"""MLflow tracking helper."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .exceptions import ConnectorError, RetryExhaustedError

try:  # pragma: no cover - optional import
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None


logger = logging.getLogger(__name__)


@dataclass
class MLflowTracker:
    """Tracker for logging metrics and parameters to MLflow with retry support.

    Provides a simple interface to log training metrics and hyperparameters
    to an MLflow tracking server with automatic exponential backoff retry logic.
    Requires mlflow library (optional dependency).

    Parameters
    ----------
    experiment : str, optional
        MLflow experiment name. If None, uses default experiment.
    tracking_uri : str, optional
        MLflow tracking server URI. If None, uses default.
    max_retries : int, optional
        Maximum number of retry attempts (default: 3).
    timeout : int, optional
        Operation timeout in seconds (default: 30, not currently enforced).

    Attributes
    ----------
    experiment : str, optional
        MLflow experiment name.
    tracking_uri : str, optional
        MLflow tracking URI.
    max_retries : int
        Max retry attempts.
    timeout : int
        Operation timeout.

    Examples
    --------
    Load from environment variables:
        >>> tracker = MLflowTracker.from_env()
    """

    experiment: Optional[str] = None
    tracking_uri: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30

    @classmethod
    def from_env(cls, prefix: str = "ORCHID_MLFLOW") -> MLflowTracker:
        """Create an MLflowTracker from environment variables.

        Reads configuration from environment variables with the given prefix.
        All variables are optional.

        Parameters
        ----------
        prefix : str, optional
            Environment variable prefix (default: "ORCHID_MLFLOW").
            For example, with prefix="ORCHID_MLFLOW", expects:
            - ORCHID_MLFLOW_EXPERIMENT (optional)
            - ORCHID_MLFLOW_TRACKING_URI (optional)

        Returns
        -------
        MLflowTracker
            Configured tracker instance.
        """
        experiment = os.environ.get(f"{prefix}_EXPERIMENT")
        tracking_uri = os.environ.get(f"{prefix}_TRACKING_URI")

        return cls(
            experiment=experiment,
            tracking_uri=tracking_uri,
        )

    def _client(self):
        """Get or initialize MLflow client.

        Sets tracking URI and experiment if configured.

        Returns
        -------
        mlflow
            MLflow module with tracking context.

        Raises
        ------
        ImportError
            If mlflow is not installed.
        """
        if mlflow is None:  # pragma: no cover
            raise ImportError("mlflow is required. Install via `pip install orchid-ranker[connectors]`")
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment:
            mlflow.set_experiment(self.experiment)
        return mlflow

    def __enter__(self):
        """Context manager entry."""
        self._run = self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()

    def start_run(self):
        """Start an MLflow run.

        Raises
        ------
        ConnectorError
            If run startup fails.
        """
        try:
            client = self._client()
            client.start_run()
            logger.info("MLflow run started")
        except Exception as e:
            raise ConnectorError(f"Failed to start MLflow run: {e}") from e

    def end_run(self):
        """End the current MLflow run.

        Logs a warning if run termination fails but does not raise.
        """
        try:
            client = self._client()
            client.end_run()
            logger.info("MLflow run ended")
        except Exception as e:
            logger.warning(f"Error ending MLflow run: {e}")

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

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow with automatic retry.

        Parameters
        ----------
        metrics : dict
            Mapping from metric name to float value.
        step : int, optional
            Training step/epoch number.

        Raises
        ------
        ImportError
            If mlflow is not installed.
        RetryExhaustedError
            If logging fails after all retry attempts.
        """
        # Check dependencies before entering retry loop
        self._client()

        def _log():
            client = self._client()
            client.log_metrics(metrics, step=step)

        self._retry_with_backoff(_log)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to MLflow with automatic retry.

        Parameters
        ----------
        params : dict
            Mapping from parameter name to value (string or scalar).

        Raises
        ------
        ImportError
            If mlflow is not installed.
        RetryExhaustedError
            If logging fails after all retry attempts.
        """
        # Check dependencies before entering retry loop
        self._client()

        def _log():
            client = self._client()
            client.log_params(params)

        self._retry_with_backoff(_log)


__all__ = ["MLflowTracker"]
