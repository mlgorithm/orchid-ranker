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
    InteractionEventBus,
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
        device="cpu",
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

    def test_rejects_non_binary_correct_value(self):
        with pytest.raises(ValueError, match="correct"):
            InteractionEvent.from_mapping(
                {"user_id": 1, "item_id": 2, "correct": 2}
            )

    def test_bus_token_excluded_from_equality(self):
        # FIX 5: the internal bus_token must not change value semantics, so two
        # otherwise-identical interactions still compare (and hash) equal.
        import dataclasses

        a = InteractionEvent(1, 2, True)
        b = dataclasses.replace(a, bus_token=42)
        c = dataclasses.replace(a, bus_token=99)
        assert a == b == c
        assert hash(a) == hash(b) == hash(c)
        assert b.bus_token == 42 and c.bus_token == 99
        # Default is None for events created from the wire schema.
        assert InteractionEvent.from_mapping(
            {"user_id": 1, "item_id": 2, "correct": 1}
        ).bus_token is None


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
        assert applied == 1
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

    def test_single_error_does_not_trip_degraded(self, ranker):
        # FIX 4: one apply error must not flip the degraded health signal.
        bus = InMemoryEventBus()
        ingestor = StreamingIngestor(
            bus, ranker, poll_timeout_s=0.02,
            max_consecutive_apply_errors=3,
        )
        bus.publish(InteractionEvent(user_id=999, item_id=0, correct=True))  # bad
        bus.publish(InteractionEvent(user_id=1, item_id=0, correct=True))    # good
        ingestor.drain()
        assert ingestor.metrics.apply_errors == 1
        assert ingestor.metrics.degraded is False
        # The good event reset the streak.
        assert ingestor.metrics.consecutive_apply_errors == 0

    def test_consecutive_errors_trip_degraded(self, ranker):
        # FIX 4: a deterministically-failing observe() must trip the health
        # signal after the configured threshold and fire on_degraded.
        bus = InMemoryEventBus()
        degraded_calls: list = []
        ingestor = StreamingIngestor(
            bus, ranker, poll_timeout_s=0.02,
            max_consecutive_apply_errors=3,
            on_degraded=lambda ing: degraded_calls.append(ing),
        )
        for _ in range(5):
            bus.publish(InteractionEvent(user_id=999, item_id=0, correct=True))
        ingestor.drain()
        assert ingestor.metrics.degraded is True
        assert ingestor.metrics.consecutive_apply_errors >= 3
        # on_degraded fires exactly once on the transition.
        assert len(degraded_calls) == 1
        assert degraded_calls[0] is ingestor

    def test_degraded_reraise_propagates(self, ranker):
        # FIX 4: with reraise_on_degraded, drain raises IngestorDegraded so a
        # supervisor can restart the worker.
        from orchid_ranker.streaming_bus import IngestorDegraded

        bus = InMemoryEventBus()
        ingestor = StreamingIngestor(
            bus, ranker, poll_timeout_s=0.02,
            max_consecutive_apply_errors=2,
            reraise_on_degraded=True,
            on_degraded=lambda ing: None,  # avoid touching global readiness
        )
        for _ in range(3):
            bus.publish(InteractionEvent(user_id=999, item_id=0, correct=True))
        with pytest.raises(IngestorDegraded):
            ingestor.drain()

    def test_consecutive_counter_resets_on_success(self, ranker):
        bus = InMemoryEventBus()
        ingestor = StreamingIngestor(
            bus, ranker, poll_timeout_s=0.02,
            max_consecutive_apply_errors=10,
        )
        # bad, bad, good -> streak must be 0 after the good one.
        bus.publish(InteractionEvent(user_id=999, item_id=0, correct=True))
        bus.publish(InteractionEvent(user_id=998, item_id=0, correct=True))
        bus.publish(InteractionEvent(user_id=2, item_id=0, correct=True))
        ingestor.drain()
        assert ingestor.metrics.consecutive_apply_errors == 0
        assert ingestor.metrics.apply_errors == 2
        assert not ingestor.metrics.degraded


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


# ---------------------------------------------------------------------------
# Pending-message bookkeeping (FIX 5): stable token + prune on failure
# ---------------------------------------------------------------------------
class _PendingBus(InteractionEventBus):
    """Minimal bus that mimics Kafka's per-message pending bookkeeping using the
    stable ``bus_token`` carried on the event. ``ack`` and ``fail`` both prune
    the pending entry; nothing leaks on either path."""

    stop_on_apply_error = False

    def __init__(self, events):
        self._events = list(events)
        self._seq = 0
        self.pending = {}
        self.acked = []
        self.failed = []

    def poll(self, max_events=100, timeout_s=0.5):
        import dataclasses

        out = []
        while self._events and len(out) < max_events:
            ev = self._events.pop(0)
            token = self._seq
            self._seq += 1
            self.pending[token] = ev  # stand-in for the broker message
            out.append(dataclasses.replace(ev, bus_token=token))
        return out

    def ack(self, event):
        if event.bus_token is not None:
            self.pending.pop(event.bus_token, None)
            self.acked.append(event.bus_token)

    def fail(self, event):
        if event.bus_token is not None:
            self.pending.pop(event.bus_token, None)
            self.failed.append(event.bus_token)


class TestPendingBookkeeping:
    def test_pending_pruned_on_success_and_failure(self, ranker):
        # One good event (acked) and one bad event (failed) -- after draining,
        # nothing must remain pending, regardless of outcome.
        bus = _PendingBus([
            InteractionEvent(user_id=1, item_id=0, correct=True),    # ok -> ack
            InteractionEvent(user_id=999, item_id=0, correct=True),  # bad -> fail
        ])
        ingestor = StreamingIngestor(bus, ranker, poll_timeout_s=0.02)
        ingestor.drain()
        assert bus.pending == {}, "no pending entry may leak on ack or fail"
        assert len(bus.acked) == 1
        assert len(bus.failed) == 1

    def test_distinct_tokens_per_message(self, ranker):
        evs = [InteractionEvent(user_id=1, item_id=0, correct=True) for _ in range(5)]
        bus = _PendingBus(evs)
        polled = bus.poll(max_events=10)
        tokens = [e.bus_token for e in polled]
        assert len(set(tokens)) == len(tokens), "tokens must be unique per message"
        # Identical-valued events still get distinct tokens (id() reuse would not
        # guarantee this).
        assert tokens == sorted(tokens)
