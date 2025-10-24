# Deployment & Observability Guide

## Docker

```bash
docker build -t orchid-ranker .
docker run --rm -p 8000:8000 orchid-ranker
```

The Dockerfile installs Orchid Ranker with agentic, viz, preprocess, and observability extras and exposes port 8000 by default.

## Helm

Use the chart under `deploy/helm/orchid-ranker`:

```bash
helm install orchid ./deploy/helm/orchid-ranker \
  --set image.repository=ghcr.io/your-org/orchid-ranker \
  --set image.tag=v0.2.0 \
  --set env.ORCHID_AUDIT_ENDPOINT=https://siem.example/v1/events
```

Metrics are published on the `metrics` port (default 9090) and audit secrets are sourced from the generated secret template.

## Terraform

The `deploy/terraform` directory contains a reference module snippet illustrating how to wrap the Helm chart from Terraform Helm releases. Integrate it with your existing infrastructure-as-code stack to manage namespaces, secrets, and ingress.

## Observability

- Start the Prometheus exporter within your service using `orchid_ranker.start_metrics_server()`.
- Record metrics for training jobs via `orchid_ranker.record_training(duration_seconds, epsilon=…)`.
- Fetch metrics payloads for custom exporters with `orchid_ranker.export_metrics()`.
- Structured logs can be routed through `orchid_ranker.logging.configure_logging(json=True)`.

## Data Connectors & Tracking

Optional connectors live under `orchid_ranker.connectors` and require extra dependencies:

- Snowflake (`SnowflakeConnector`): install `snowflake-connector-python`.
- BigQuery (`BigQueryConnector`): install `google-cloud-bigquery`.
- S3 streaming (`S3StreamConnector`): install `boto3`.
- Experiment tracking (`MLflowTracker`): install `mlflow`.

Install all with `pip install orchid-ranker[connectors]`.
