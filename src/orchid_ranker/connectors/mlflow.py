"""MLflow tracking helper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional import
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None


@dataclass
class MLflowTracker:
    experiment: Optional[str] = None
    tracking_uri: Optional[str] = None

    def _client(self):
        if mlflow is None:  # pragma: no cover
            raise ImportError("mlflow is required. Install via `pip install orchid-ranker[connectors]`")
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment:
            mlflow.set_experiment(self.experiment)
        return mlflow

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        client = self._client()
        client.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        client = self._client()
        client.log_params(params)
