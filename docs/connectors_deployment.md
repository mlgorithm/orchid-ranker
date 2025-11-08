# Connectors & Deployment Guide

## 1. Data connectors

### Snowflake
```python
from orchid_ranker.connectors.snowflake import SnowflakeFetcher
fetcher = SnowflakeFetcher(conn_str="...", role="ANALYST")
df = fetcher.query("SELECT user_id, item_id, label FROM interactions")
```

### BigQuery
```python
from orchid_ranker.connectors.bigquery import BigQueryFetcher
fetcher = BigQueryFetcher(project_id="my-proj")
df = fetcher.query("SELECT * FROM dataset.interactions")
```

### S3 streaming
Use `orchid_ranker.connectors.s3_stream` to stream interaction logs into the agentic service.

## 2. Experiment logging to MLflow

```python
from orchid_ranker.connectors.mlflow import MLflowLogger
logger = MLflowLogger(experiment="orchid")
logger.log_params({...})
```

## 3. Deployment options

- **Docker**: `docker build -t orchid-ranker .` and run `docker run -p 8000:8000 orchid-ranker`.
- **Helm**: `helm install orchid ./deploy/helm/orchid-ranker` (configure values for secrets, DP flags, connectors).
- **Terraform**: reference modules under `deploy/terraform/` to provision infrastructure.

## 4. Environment variables

| Variable | Description |
| --- | --- |
| `ORCHID_DP_ENABLED` | Toggle DP globally |
| `ORCHID_SAFE_EB` | Enable SafeSwitch gate |
| `ORCHID_METRICS_PORT` | Port for Prometheus exporter |
| `ORCHID_CONNECTOR_*` | Credentials for Snowflake/BigQuery/S3 |

## 5. Checklist

- [ ] Build container image and push to registry.
- [ ] Configure secrets (Snowflake, BigQuery) via Kubernetes secrets or cloud KMS.
- [ ] Enable metrics endpoint and SIEM log shipping.
- [ ] Run `ci_safe_smoke.sh` against staging before promoting to prod.
