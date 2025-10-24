"""BigQuery connector utilities (optional dependency)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


try:  # pragma: no cover - optional import
    from google.cloud import bigquery  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bigquery = None


@dataclass
class BigQueryConnector:
    project: Optional[str] = None
    dataset: Optional[str] = None

    def _client(self):
        if bigquery is None:  # pragma: no cover
            raise ImportError(
                "google-cloud-bigquery is required. Install via `pip install orchid-ranker[connectors]`"
            )
        return bigquery.Client(project=self.project)

    def query_dataframe(self, sql: str):
        import pandas as pd  # type: ignore

        job = self._client().query(sql)
        return job.result().to_dataframe(create_bqstorage_client=False)

    def load_dataframe(self, table: str, dataframe):
        client = self._client()
        destination = f"{self.dataset}.{table}" if self.dataset else table
        job = client.load_table_from_dataframe(dataframe, destination)
        return job.result()
