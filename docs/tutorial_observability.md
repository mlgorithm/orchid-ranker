# Observability & Monitoring

## 1. Timing logs

Run benchmarks with `--timing-log runs/foo/timing.jsonl --timing-rounds 5`. Each entry contains durations for `candidate_sampling`, `tower_infer`, `decide`, `student_interact`, `train_step`, `warmup_sync`.

## 2. Prometheus exporter

```python
from orchid_ranker.observability import MetricsServer

server = MetricsServer(port=9000)
server.register_round_metrics()
server.start()

# Later
server.stop()
```

Add the endpoint (`/metrics`) to your Prometheus scrape config.

## 3. Sample Grafana dashboard

- Panel 1: mix probability `safe_gate.p` vs rounds.
- Panel 2: latency per phase (stacked bar from timing logs).
- Panel 3: DP epsilon vs time.

## 4. Alerts

- `safe_gate.p` stuck at 0 for > N rounds → potential regression.
- `tower_infer` latency > budget → scale resources / enable native scoring.

## 5. Log shipping

Forward JSONL logs via `fluent-bit` or a lightweight Python shipper:

```python
from orchid_ranker.logging import stream_jsonl
stream_jsonl(\"runs/ml100k-safe/adaptive.jsonl\", target=\"https://siem.example/input\")
```
