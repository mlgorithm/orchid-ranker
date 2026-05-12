# Connectors & Deployment Guide

## 1. Data connectors

### Snowflake
```python
from orchid_ranker.connectors.snowflake import SnowflakeConnector

connector = SnowflakeConnector.from_env()
df = connector.fetch_dataframe("SELECT user_id, item_id, label FROM interactions")
```

### BigQuery
```python
from orchid_ranker.connectors.bigquery import BigQueryConnector

connector = BigQueryConnector(project="my-proj")
df = connector.query_dataframe("SELECT * FROM dataset.interactions")
```

### S3 streaming
Use `orchid_ranker.connectors.s3_stream` to stream interaction logs into the agentic service.

## 2. Experiment logging to MLflow

```python
from orchid_ranker.connectors.mlflow import MLflowTracker

tracker = MLflowTracker(experiment="orchid")
tracker.log_params({...})
```

## 3. Deployment options

- **Docker**: `docker build -t orchid-ranker .` and run `docker run -p 8000:8000 -p 8081:8081 -p 9090:9090 orchid-ranker`.
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
