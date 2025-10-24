"""Snowflake connector utilities (optional dependency)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


try:  # pragma: no cover - optional import
    import snowflake.connector  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    snowflake = None
else:  # pragma: no cover - optional dependency
    snowflake = snowflake.connector


@dataclass
class SnowflakeConnector:
    """Thin wrapper around `snowflake.connector` with Pandas integration."""

    account: str
    user: str
    password: str
    warehouse: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None

    def _require_lib(self):
        if snowflake is None:  # pragma: no cover - runtime path
            raise ImportError(
                "snowflake-connector-python is required. Install via `pip install orchid-ranker[connectors]`"
            )

    def _connect(self):
        self._require_lib()
        return snowflake.connect(  # type: ignore[union-attr]
            account=self.account,
            user=self.user,
            password=self.password,
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema,
        )

    def fetch_dataframe(self, query: str):
        import pandas as pd  # type: ignore

        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(query)
                data = cur.fetchall()
                columns = [col[0] for col in cur.description]
                return pd.DataFrame(data, columns=columns)
            finally:
                cur.close()

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(query, params or {})
            finally:
                cur.close()
