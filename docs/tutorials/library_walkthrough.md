# Orchid Ranker End-to-End Tutorial

This walkthrough expands on the quickstart guide and touches the major features of the library: fitting recommenders, evaluating, applying differential privacy, tracking metrics, and simulating adaptive policies.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install orchid-ranker[agentic,viz,observability]
```

The extras install plotting, agentic simulators, and Prometheus helpers (optional but recommended for this tutorial).

## 2. Prepare Sample Data

```python
import pandas as pd

interactions = pd.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 3, 3, 4],
    "item_id": [11, 12, 13, 11, 14, 12, 15, 16],
    "label":   [1, 0, 1, 1, 0, 1, 1, 0],
})
train = interactions
test = interactions.sample(frac=0.4, random_state=42)
```

## 3. Train Multiple Strategies

```python
from orchid_ranker import OrchidRecommender

strategies = [
    ("als", {"epochs": 5}),
    ("neural_mf", {"epochs": 5, "emb_dim": 16}),
    ("user_knn", {"k": 3}),
]

recs = {}
for name, params in strategies:
    rec = OrchidRecommender(strategy=name, **params)
    rec.fit(train, rating_col="label")
    recs[name] = rec
    print(name, rec.recommend(user_id=1, top_k=3))
```

## 4. Evaluate with CLI

```bash
python examples/quickstart.py  # creates CSVs in examples/data/

orchid-evaluate \
  --train examples/data/quickstart_train.csv \
  --test examples/data/quickstart_test.csv \
  --strategy "als,epochs=3" \
  --strategy "user_knn,k=10" \
  --top-k 5
```

## 5. Differential Privacy Training

```python
dp_cfg = {
    "enabled": True,
    "engine": "opacus",  # or "per_sample"
    "noise_multiplier": 1.0,
    "sample_rate": 0.05,
    "delta": 1e-5,
}

dp_rec = OrchidRecommender(strategy="als", epochs=3, dp_cfg=dp_cfg)
dp_rec.fit(train, rating_col="label")
print(dp_rec.recommend(user_id=2, top_k=3))
```

## 6. Metrics & Observability

```python
from orchid_ranker import start_metrics_server, record_training, export_metrics

start_metrics_server(port=9090)
record_training(duration_seconds=3.2, epsilon=0.9)
print(export_metrics().decode("utf-8")[:200])
```

Fetch `http://localhost:9090/metrics` with curl/Prometheus.

## 7. Audit Logging

```python
from orchid_ranker import AuditLogger

logger = AuditLogger.from_env(path="tutorial_audit.jsonl")  # uses env vars if set
logger.log_event("training_run", actor="tutorial", payload={"strategy": "als", "epsilon": 0.9})
```

## 8. Connectors (Optional)

```python
from orchid_ranker import SnowflakeConnector

# Raises ImportError until snowflake connector is installed
try:
    sf = SnowflakeConnector(account="acct", user="user", password="pwd")
    df = sf.fetch_dataframe("SELECT CURRENT_ROLE()")
except ImportError:
    print("Install extras: pip install orchid-ranker[connectors]")
```

## 9. Agentic Simulation Snapshot

```python
from orchid_ranker.agents import StudentAgent, MultiConfig, MultiUserOrchestrator
from orchid_ranker.agents.recommender_agent import TwoTowerRecommender
import torch

num_users, num_items = 5, 10
model = TwoTowerRecommender(
    num_users=num_users,
    num_items=num_items,
    user_dim=4,
    item_dim=4,
    dp_cfg={"enabled": False},
)
students = [StudentAgent(user_id=i, seed=42 + i) for i in range(num_users)]
users = []
for idx, student in enumerate(students):
    users.append(
        MultiUserOrchestrator.UserCtx(  # type: ignore[attr-defined]
            user_id=idx,
            user_idx=idx,
            student=student,
            user_vec=torch.rand(1, 4),
        )
    )

config = MultiConfig(rounds=3, top_k_base=2, log_path="runs/tutorial.jsonl", console=False)
orch = MultiUserOrchestrator(
    rec=model,
    users=users,
    item_matrix_normal=torch.rand(num_items, 4),
    item_matrix_sanitized=None,
    item_ids_pos=torch.arange(num_items),
    pos2id=list(range(num_items)),
    id2pos={i: i for i in range(num_items)},
    item_meta_by_id={},
    cfg=config,
    device=torch.device("cpu"),
    mode_label="tutorial",
)
orch.run()
print("Simulation complete; see runs/tutorial.jsonl")
```

## 10. Next Steps
- Extend the notebook in `examples/notebooks/pilot_quickstart.ipynb` for your data.
- Register Prometheus dashboards or SIEM forwarding using environment variables (`ORCHID_AUDIT_ENDPOINT`).
- Engage with the design partner pilot process if you need structured evaluation support.
