# Guide 3: Operate safely in production

An adaptive ranker that keeps users clicking but stops them progressing should
be halted. Generic recommender stacks gate on CTR or revenue; progression
domains need guardrails tied to competence, category coverage, and stretch
zone fit. This guide adds monitoring, a domain-specific circuit breaker, and a
Prometheus + Grafana observability layer.

## Add a rolling progression monitor

```python
from orchid_ranker.live_metrics import RollingProgressionMonitor

monitor = RollingProgressionMonitor(
    window_size=500,
    total_categories={"algebra", "fractions", "geometry"},  # your category universe
    success_threshold=0.7,
    stretch_width=0.25,
)
```

The monitor maintains a bounded window of recent outcomes and recomputes four
metrics on every event: `progression_gain`, `category_coverage`,
`sequence_adherence`, and `stretch_fit`. Values are automatically pushed to
Prometheus gauges.

Record each served item after the learner outcome arrives. With the primary
adaptive-learning API, capture competence before and after `observe()`:

```python
ranked = rec.rank(user_id=42, candidate_item_ids=candidates, top_k=10)
chosen = ranked[0]
concept = chosen.concept_id or "__default__"
pre = rec.competence_for(user_id=42, concept=concept)

rec.observe(user_id=42, item_id=chosen.item_id, correct=True)
post = rec.competence_for(user_id=42, concept=concept)

monitor.record(
    user_id=42,
    item_id=chosen.item_id,
    correct=True,
    pre_competence=pre,
    post_competence=post,
    category=str(concept),
    difficulty=chosen.difficulty,
)
```

## Add a progression guardrail

The guardrail watches the monitor and halts the adaptive policy when metrics
drop below configured floors.

```python
from orchid_ranker.live_metrics import ProgressionGuardrail, GuardrailConfig

cfg = GuardrailConfig(
    min_progression_gain=0.0,
    min_accept_rate=0.3,
    min_sequence_adherence=0.8,
    warmup_samples=50,
    consecutive_violations=3,
)
guardrail = ProgressionGuardrail(monitor, cfg)
```

## The fallback pattern

Before every rank call, ask the guardrail whether the adaptive policy is still
safe. If it has tripped, fall back to a reviewed non-adaptive learning policy:
for example prerequisite order plus difficulty fit. Keep that fallback in a
separate object so the adaptive learner and reviewed policy have clear
responsibilities.

```python
if guardrail.evaluate():
    # Adaptive path -- KT, progression reward, and live competence updates
    top = rec.rank(user_id=42, candidate_item_ids=candidates, top_k=10)
else:
    # Guardrail fired -- serve the reviewed prerequisite/difficulty policy
    top = reviewed_policy.rank(user_id=42, candidate_item_ids=candidates, top_k=10)
```

Pass the same candidate pool to both paths when you need the fallback to rank
the same eligible learning items.

## Start the Prometheus metrics server

```python
from orchid_ranker.observability import start_metrics_server

start_metrics_server(port=9090)  # exposes /metrics on 0.0.0.0:9090
```

## Prometheus scrape config

Add this job to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: orchid-ranker
    scrape_interval: 15s
    static_configs:
      - targets: ["orchid-ranker:9090"]
```

Key metrics to watch:

| Metric | Description |
|---|---|
| `orchid_progression_gain` | Rolling competence gain per event |
| `orchid_proficiency_coverage` | Fraction of categories completed |
| `orchid_difficulty_appropriateness` | Fraction of items in the stretch zone |
| `orchid_rolling_accept_rate` | Rolling acceptance / correctness rate |
| `orchid_progression_guardrail_halted` | 1 when guardrail has halted adaptive policy |

## Grafana dashboard

Import the Orchid dashboard from the metrics above. A minimal panel set:

1. **Progression gain** -- timeseries of `orchid_progression_gain{policy="adaptive"}`.
2. **Category coverage** -- gauge showing `orchid_proficiency_coverage`.
3. **Stretch fit** -- timeseries of `orchid_difficulty_appropriateness`.
4. **Guardrail status** -- stat panel on `orchid_progression_guardrail_halted` with thresholds (0 = green, 1 = red).
5. **Accept rate** -- timeseries comparing `orchid_rolling_accept_rate{policy="adaptive"}` vs `{policy="baseline"}`.

Each panel filters on `{policy="adaptive"}` so you can overlay a baseline
control group if you run one.

---

Your recommender is now adaptive, live, and safe. If any metric drops below
your configured floor, the system automatically falls back to the frozen
reviewed learning policy.
