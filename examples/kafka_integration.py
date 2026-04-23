#!/usr/bin/env python3
"""Streaming integration demo -- InMemoryEventBus + StreamingIngestor.

Shows the full observe/rank loop without requiring a Kafka broker.
Swap InMemoryEventBus for KafkaEventBus in production (same interface).

Run with: python examples/kafka_integration.py
Requires: pip install orchid-ranker[ml]
"""
import numpy as np
import pandas as pd

from orchid_ranker import OrchidRecommender
from orchid_ranker.streaming_bus import InMemoryEventBus, InteractionEvent, StreamingIngestor

# --- Train a neural-MF recommender (needed for streaming tower) ---
rng = np.random.default_rng(7)
n_users, n_items = 50, 80
interactions = pd.DataFrame({
    "user_id": rng.integers(0, n_users, 2000),
    "item_id": rng.integers(0, n_items, 2000),
})
rec = OrchidRecommender.from_interactions(interactions, strategy="neural_mf")
ranker = rec.as_streaming(lr=0.05, l2=1e-3)

# --- Wire up the event bus and ingestor ---
bus = InMemoryEventBus()
ingestor = StreamingIngestor(bus, ranker)

# --- Simulate live interactions arriving on the bus ---
for uid, iid, correct in [(1, 7, True), (1, 12, False), (1, 3, True)]:
    bus.publish(InteractionEvent(user_id=uid, item_id=iid, correct=correct))

# Drain events into the ranker (synchronous; in production use ingestor.start())
applied = ingestor.drain(max_batches=1)
print(f"Applied {applied} events from the bus")

# --- Rank after adaptation ---
candidates = list(range(n_items))
top5 = ranker.rank(user_id=1, candidate_item_ids=candidates, top_k=5)
print("\nTop-5 for user 1 (post-adaptation):")
for item_id, score in top5:
    print(f"  item {item_id}: score {score:.4f}")

print(f"\nUser 1 competence: {ranker.competence(1):.3f}")
print(f"Ingestor metrics: {ingestor.metrics.as_dict()}")
