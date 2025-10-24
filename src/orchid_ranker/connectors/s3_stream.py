"""S3 streaming connector for ingest/egress."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


try:  # pragma: no cover - optional import
    import boto3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    boto3 = None


@dataclass
class S3StreamConnector:
    bucket: str
    prefix: str = ""
    profile: Optional[str] = None
    region: Optional[str] = None

    def _client(self):
        if boto3 is None:  # pragma: no cover
            raise ImportError(
                "boto3 is required. Install via `pip install orchid-ranker[connectors]`"
            )
        session_kwargs = {}
        if self.profile:
            session_kwargs["profile_name"] = self.profile
        session = boto3.session.Session(**session_kwargs)
        return session.client("s3", region_name=self.region)

    def list_objects(self) -> Iterable[str]:
        client = self._client()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for item in page.get("Contents", []):
                yield item["Key"]

    def stream_object(self, key: str):
        client = self._client()
        obj = client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].iter_lines()
