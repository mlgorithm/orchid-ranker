# Guide 2: Serve a streaming recommender

A batch recommender re-ranks on stale data. In a progression domain --
education, rehab, onboarding -- a user's competence changes with every
interaction. This guide promotes the batch model from Guide 1 into a live,
adaptive ranker that updates per-user residuals and competence estimates on
every observed outcome.

## Prerequisites

A fitted `OrchidRecommender` with a neural strategy. The streaming bridge
requires a neural tower, so use `neural_mf` from the high-level API. The
lower-level `TwoTowerRecommender` is available for advanced custom training, but
it is not a `strategy=` value for `OrchidRecommender`.

```python
import pandas as pd
from orchid_ranker import OrchidRecommender

df = pd.read_csv("interactions.csv")
rec = OrchidRecommender.from_interactions(df, strategy="neural_mf",
                                          user_col="user_id",
                                          item_col="item_id",
                                          rating_col="rating")
```

## Create a streaming ranker

```python
streamer = rec.as_streaming(lr=0.05, l2=1e-3)
```

`as_streaming()` freezes the base tower and attaches a per-user online adapter
plus a Bayesian Knowledge Tracing state provider. No tensors to manage -- the
bridge materialises everything internally and accepts the original `user_id` and
`item_id` values from the fitted interactions DataFrame.

## Observe and rank

```python
# A user answers item 7 correctly -- update competence + residual
streamer.observe(user_id=42, item_id=7, correct=True)

# Re-rank candidates with the just-updated state
top = streamer.rank(user_id=42, candidate_item_ids=[1, 2, 3, 7, 99], top_k=5)
for item_id, score in top:
    print(item_id, score)
```

Each `observe` runs one SGD step on the user's residual embedding and updates
their competence estimate via outcome tracing. The next `rank` reflects that
update immediately -- no batch delay. Use `category="fractions"` on `observe`
to track competence per category rather than a single global estimate.

## Hook up a Kafka bus

For a production service, interactions arrive from a message bus rather than
inline calls. `StreamingIngestor` ties a bus to the ranker in a background
thread.

```python
from orchid_ranker.streaming_bus import KafkaEventBus, StreamingIngestor

bus = KafkaEventBus(brokers="kafka:9092", topic="interactions")
ingestor = StreamingIngestor(bus, streamer)
ingestor.start()  # background daemon thread
```

Events are JSON with the schema `{"user_id": int, "item_id": int,
"correct": 0|1}`. Optional fields: `"skill"` (legacy category label; use
`"category"` when calling `observe` directly) and `"timestamp"` (epoch seconds).

## Minimal docker-compose for Kafka

```yaml
version: "3.8"
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on: [zookeeper]
    ports: ["9092:9092"]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

Install the Kafka driver alongside Orchid:

```bash
pip install 'orchid-ranker[ml]' confluent-kafka
```

---

You can stop here and have a live, adaptive recommender. For production safety
and monitoring, continue to [Guide 3: Operate safely in production](03-operate-safely.md).
