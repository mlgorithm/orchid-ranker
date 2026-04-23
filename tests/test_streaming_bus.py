"""Tests for the interaction event bus and streaming ingestor.

These tests cover the contract that lets the ranker be driven by a Kafka
topic in production: events on the bus must land in the ranker, errors must
not crash the loop, and shutdown must be prompt.

All tests use :class:`InMemoryEventBus`; the Kafka integration is covered
by a smoke test that asserts the driver import path and error message.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
import torch

from orchid_ranker.agents.two_tower import TwoTowerRecommender
from orchid_ranker.streaming import StreamingAdaptiveRanker
from orchid_ranker.streaming_bus import (
    InMemoryEventBus,
    InteractionEvent,
    StreamingIngestor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
NUM_USERS = 12
NUM_ITEMS = 20
FEAT_DIM = 4


@pytest.fixture
def ranker():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    uf = torch.tensor(rng.normal(size=(NUM_USERS, FEAT_DIM)).astype(np.float32))
    ifeat = torch.tensor(rng.normal(size=(NUM_ITEMS, FEAT_DIM)).astype(np.float32))
    tower = TwoTowerRecommender(
        num_users=NUM_USERS, num_items=NUM_ITEMS,
        user_dim=FEAT_DIM, item_dim=FEAT_DIM,
        hidden=8, emb_dim=8, state_dim=4,
        device="cpu", dp_cfg={"enabled": False},
    ).eval()
    return StreamingAdaptiveRanker(tower, uf, ifeat)


# ---------------------------------------------------------------------------
# InteractionEvent
# ---------------------------------------------------------------------------
class TestInteractionEvent:
    def test_from_mapping_minimal(self):
        ev = InteractionEvent.from_mapping(
            {"user_id": 3, "item_id": 5, "correct": 1}
        )
        assert ev.user_id == 3 and ev.item_id == 5
        assert ev.correct is True
        assert ev.skill is None and ev.timestamp is None

    def test_from_mapping_full(self):
        ev = InteractionEvent.from_mapping({
            "user_id": "7", "item_id": 2, "correct": 0,
            "skill": "frac", "timestamp": 12345.6,
        })
        assert ev.user_id == 7 and ev.correct is False
        assert ev.skill == "frac" and ev.timestamp == pytest.approx(12345.6)

    def test_from_json(self):
        raw = json.dumps({"user_id": 1, "item_id": 2, "correct": 1})
        assert InteractionEvent.from_json(raw).user_id == 1
        assert InteractionEvent.from_json(raw.encode()).user_id == 1

    def test_rejects_missing_field(self):
        with pytest.raises(ValueError):
            InteractionEvent.from_mapping({"user_id": 1, "correct": 1})

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError):
            InteractionEvent.from_mapping(
                {"user_id": "not-an-int", "item_id": 2, "correct": 1}
            )


# ---------------------------------------------------------------------------
# InMemoryEventBus
# ---------------------------------------------------------------------------
class TestInMemoryEventBus:
    def test_publish_and_poll(self):
        bus = InMemoryEventBus()
        bus.publish(InteractionEvent(1, 2, True))
        bus.publish({"user_id": 3, "item_id": 4, "correct": 0})
        got = bus.poll(max_events=10, timeout_s=0.1)
        assert [(e.user_id, e.item_id) for e in got] == [(1, 2), (3, 4)]

    def test_poll_timeout_returns_empty(self):
        bus = InMemoryEventBus()
        t0 = time.monotonic()
        got = bus.poll(max_events=5, timeout_s=0.05)
        elapsed = time.monotonic() - t0
        assert got == []
        assert elapsed >= 0.04  # respected the timeout (within scheduler slop)

    def test_respects_max_events(self):
        bus = InMemoryEventBus()
        for k in range(20):
            bus.publish(InteractionEvent(k, k, True))
        got = bus.poll(max_events=5, timeout_s=0.1)
        assert len(got) == 5
        # remainder stays queued
        assert len(bus.poll(max_events=100, timeout_s=0.1)) == 15

    def test_close_unblocks_poll(self):
        bus = InMemoryEventBus()
        result = {}

        def waiter():
            result["events"] = bus.poll(max_events=1, timeout_s=5.0)

        th = threading.Thread(target=waiter)
        th.start()
        time.sleep(0.05)
        bus.close()
        th.join(timeout=1.0)
        assert not th.is_alive(), "close() must unblock poll()"
        assert result["events"] == []

    def test_publish_after_close_raises(self):
        bus = InMemoryEventBus()
        bus.close()
        with pytest.raises(RuntimeError):
            bus.publish(InteractionEvent(0, 0, True))


# ---------------------------------------------------------------------------
# StreamingIngestor
# ---------------------------------------------------------------------------
class TestStreamingIngestor:
    def test_drain_applies_events_to_ranker(self, ranker):
        bus = InMemoryEventBus()
        ingestor = StreamingIngestor(bus, ranker, poll_timeout_s=0.05)
        for k in range(5):
            bus.publish(InteractionEvent(user_id=k, item_id=k, correct=True))
        applied = ingestor.drain(max_batches=1)
        assert applied == 5
        assert ingestor.metrics.events_applied == 5
        assert ingestor.metrics.events_consumed == 5
        # ranker state must reflect the events
        assert ranker.updates_for(0) == 1 and ranker.updates_for(4) == 1

    def test_apply_error_counted_not_raised(self, ranker):
        bus = InMemoryEventBus()
        ingestor = StreamingIngestor(bus, ranker, poll_timeout_s=0.05)
        # Out-of-range user triggers IndexError deep inside the adapter;
        # ingestor must log it and continue.
        bus.publish(InteractionEvent(user_id=999, item_id=0, correct=True))
        bus.publish(InteractionEvent(user_id=1, item_id=0, correct=True))
        applied = ingestor.drain(max_batches=1)
        assert applied == 2  # we tried both
        assert ingestor.metrics.apply_errors == 1
        assert ingestor.metrics.events_applied == 1
        assert ranker.updates_for(1) == 1

    def test_background_thread_lifecycle(self, ranker):
        bus = InMemoryEventBus()
        ingestor = StreamingIngestor(bus, ranker, poll_timeout_s=0.02)
        ingestor.start()
        try:
            for k in range(3):
                bus.publish(InteractionEvent(user_id=k, item_id=k, correct=True))
            # give the thread a moment to pick up the events
            deadline = time.monotonic() + 1.0
            while ingestor.metrics.events_applied < 3 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ingestor.metrics.events_applied == 3
        finally:
            ingestor.stop(timeout_s=1.0)
        # stop() must close the bus and join the thread
        assert ingestor._thread is None  # type: ignore[attr-defined]

    def test_on_event_hook_called(self, ranker):
        bus = InMemoryEventBus()
        seen = []
        ingestor = StreamingIngestor(
            bus, ranker, poll_timeout_s=0.02,
            on_event=lambda e: seen.append(e.user_id),
        )
        bus.publish(InteractionEvent(2, 3, True))
        bus.publish(InteractionEvent(5, 6, False))
        ingestor.drain()
        assert seen == [2, 5]


# ---------------------------------------------------------------------------
# Kafka adapter — smoke test only (no broker dependency)
# ---------------------------------------------------------------------------
class TestKafkaEventBus:
    def test_missing_driver_message(self, monkeypatch):
        """If confluent-kafka is not importable, we fail fast with guidance."""
        import builtins
        from orchid_ranker import streaming_bus as sb

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "confluent_kafka" or name.startswith("confluent_kafka."):
                raise ImportError("mock: confluent-kafka not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="confluent-kafka"):
            sb.KafkaEventBus(brokers="localhost:9092", topic="t")
