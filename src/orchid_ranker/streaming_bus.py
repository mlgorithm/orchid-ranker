"""Interaction event bus adapters.

Bridges a message bus (Kafka/Redis/in-memory queue) to the streaming adaptive
ranker. Interactions arrive as events and are fed into
:meth:`StreamingAdaptiveRanker.observe` with backpressure handling, error
isolation, and metrics.

Design:

* :class:`InteractionEvent` — a frozen dataclass; the wire schema.
* :class:`InteractionEventBus` — abstract base class. One method: ``poll``.
* :class:`InMemoryEventBus` — for tests and single-process deployments. Tiny.
* :class:`KafkaEventBus` — real deployment. Optional import; raises a clear
  error at construction time if ``confluent-kafka`` is not installed.
* :class:`StreamingIngestor` — ties a bus to a ranker, runs the poll loop in
  the calling thread or a background thread, and exposes metrics.

This module is intentionally single-file and framework-free: no Kafka
dependency at import time, no asyncio, no schema-registry coupling. If an
organisation prefers Pulsar/Kinesis/Redis Streams, they implement a
:class:`InteractionEventBus` subclass in ~30 lines.

JSON wire format example::

    {
      "user_id": 42,
      "item_id": 7,
      "correct": 1,
      "skill": "fractions",            // optional
      "timestamp": 1712345678.9         // optional; seconds since epoch
    }
"""
from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional

from orchid_ranker._labels import parse_binary_label
from orchid_ranker.streaming import StreamingAdaptiveRanker

logger = logging.getLogger(__name__)

__all__ = [
    "InteractionEvent",
    "InteractionEventBus",
    "InMemoryEventBus",
    "KafkaEventBus",
    "StreamingIngestor",
    "IngestorMetrics",
    "IngestorDegraded",
]


class IngestorDegraded(RuntimeError):
    """Raised from the poll loop when consecutive apply errors trip the health
    threshold and ``reraise_on_degraded`` is enabled, so a supervisor can
    restart the worker."""


def _parse_correct(value: Any) -> bool:
    return parse_binary_label(value, name="correct")


# ---------------------------------------------------------------------------
# Wire schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InteractionEvent:
    """One interaction from the bus.

    The schema is intentionally narrow — anything richer belongs in a
    side-channel feature store, not the hot path.
    """
    user_id: int
    item_id: int
    correct: bool
    skill: Optional[str] = None
    timestamp: Optional[float] = None
    #: Bus-internal, stable per-message token used to correlate this event with
    #: the underlying broker message for acknowledgement. Set by the bus at poll
    #: time (e.g. :class:`KafkaEventBus`); ``None`` for events created from the
    #: wire schema. Excluded from equality / repr so the public value semantics
    #: are unchanged and identical interactions still compare equal.
    bus_token: Optional[int] = field(default=None, compare=False, repr=False)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "InteractionEvent":
        """Parse from a dict (typical JSON deserialisation result).

        Raises :class:`ValueError` on missing or malformed required fields —
        the ingestor catches this and records a parse failure rather than
        crashing.
        """
        try:
            user_id = int(payload["user_id"])
            item_id = int(payload["item_id"])
            correct = _parse_correct(payload["correct"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid interaction event ({exc})") from exc
        skill_raw = payload.get("skill")
        skill = str(skill_raw) if skill_raw is not None else None
        ts_raw = payload.get("timestamp")
        timestamp = float(ts_raw) if ts_raw is not None else None
        return cls(user_id=user_id, item_id=item_id, correct=correct,
                   skill=skill, timestamp=timestamp)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "InteractionEvent":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return cls.from_mapping(json.loads(raw))


# ---------------------------------------------------------------------------
# Bus abstraction
# ---------------------------------------------------------------------------
class InteractionEventBus(ABC):
    """Abstract event source.

    Implementations need only ``poll`` and ``close``. ``poll`` should return
    quickly (<= ~a few seconds) — the ingestor loop calls it repeatedly.
    """
    stop_on_apply_error = False

    @abstractmethod
    def poll(self, max_events: int = 100, timeout_s: float = 0.5) -> List[InteractionEvent]:
        """Fetch up to ``max_events`` events, waiting at most ``timeout_s`` seconds."""

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release any underlying resources. Safe to call multiple times."""

    def ack(self, event: InteractionEvent) -> None:  # pragma: no cover - trivial default
        """Acknowledge successful application of an event, if the bus supports it."""

    def fail(self, event: InteractionEvent) -> None:  # pragma: no cover - trivial default
        """Signal terminal failure to apply an event, if the bus supports it.

        Lets the bus release any per-message bookkeeping (e.g. a pending offset)
        so it cannot leak on the failure path. Default is a no-op.
        """


class InMemoryEventBus(InteractionEventBus):
    """Thread-safe in-memory queue. Tests and single-process deployments.

    ``publish`` is what tests / producers call; ``poll`` is what the ingestor
    calls. A simple condition variable provides blocking-with-timeout semantics.
    """

    def __init__(self) -> None:
        self._queue: Deque[InteractionEvent] = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._closed = False

    def publish(self, event: InteractionEvent | Mapping[str, Any]) -> None:
        if not isinstance(event, InteractionEvent):
            event = InteractionEvent.from_mapping(event)
        with self._cond:
            if self._closed:
                raise RuntimeError("bus is closed")
            self._queue.append(event)
            self._cond.notify()

    def publish_many(self, events: Iterable[InteractionEvent | Mapping[str, Any]]) -> None:
        for e in events:
            self.publish(e)

    def poll(self, max_events: int = 100, timeout_s: float = 0.5) -> List[InteractionEvent]:
        deadline = time.monotonic() + float(timeout_s)
        with self._cond:
            while not self._queue and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cond.wait(timeout=remaining)
            out: List[InteractionEvent] = []
            while self._queue and len(out) < max_events:
                out.append(self._queue.popleft())
            return out

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._queue)


class KafkaEventBus(InteractionEventBus):
    """Kafka-backed bus. Requires ``confluent-kafka``.

    The dependency is imported lazily so ``orchid_ranker.streaming_bus`` can
    be imported on machines that have never heard of Kafka. Construction
    fails with a clear error if the driver is missing.

    Parameters
    ----------
    brokers : str
        Kafka bootstrap servers, e.g. ``"kafka-1:9092,kafka-2:9092"``.
    topic : str
        Topic to subscribe to.
    group_id : str
        Consumer group. Orchid treats each group as one consumer cohort —
        events are load-balanced across members of the group.
    config : dict, optional
        Extra consumer config, merged after the defaults. Use for TLS, SASL,
        auto-commit settings, etc.
    """
    stop_on_apply_error = True

    def __init__(
        self,
        brokers: str,
        topic: str,
        *,
        group_id: str = "orchid-ranker",
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            from confluent_kafka import Consumer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise ImportError(
                "KafkaEventBus requires the `confluent-kafka` package. "
                "Install with: pip install 'orchid-ranker[streaming]' "
                "or pip install confluent-kafka"
            ) from exc

        cfg = {
            "bootstrap.servers": brokers,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "latest",
        }
        if config:
            cfg.update(dict(config))
        self._consumer = Consumer(cfg)
        self._consumer.subscribe([topic])
        self._topic = topic
        self._closed = False
        # Keyed by a stable, monotonically-increasing token (NOT id(event)):
        # id() can be reused after GC, which would let ack() commit the wrong
        # message; and id()-keyed entries leaked on any non-ack path. The token
        # is carried on the event so ack()/fail() can find the exact message.
        self._pending: Dict[int, Any] = {}
        self._seq = 0
        self._pending_lock = threading.Lock()
        self.parse_errors = 0

    def poll(self, max_events: int = 100, timeout_s: float = 0.5) -> List[InteractionEvent]:  # pragma: no cover - requires broker
        if self._closed:
            return []
        out: List[InteractionEvent] = []
        deadline = time.monotonic() + float(timeout_s)
        while len(out) < max_events:
            remaining = max(0.0, deadline - time.monotonic())
            msg = self._consumer.poll(timeout=min(remaining, 0.25))
            if msg is None:
                if time.monotonic() >= deadline:
                    break
                continue
            if msg.error():
                logger.warning("kafka error: %s", msg.error())
                continue
            try:
                parsed = InteractionEvent.from_json(msg.value())
                with self._pending_lock:
                    token = self._seq
                    self._seq += 1
                    self._pending[token] = msg
                # Carry the token on the event so ack()/fail() can correlate it
                # back to this exact message without relying on id().
                event = replace(parsed, bus_token=token)
                out.append(event)
            except ValueError as exc:
                self.parse_errors += 1
                logger.warning("dropping malformed event: %s", exc)
                try:
                    self._consumer.commit(message=msg, asynchronous=False)
                except Exception:  # noqa: BLE001
                    logger.exception("failed to commit malformed kafka message")
        return out

    def ack(self, event: InteractionEvent) -> None:  # pragma: no cover - requires broker
        token = event.bus_token
        if token is None:
            return
        with self._pending_lock:
            msg = self._pending.pop(token, None)
        if msg is None or self._closed:
            return
        self._consumer.commit(message=msg, asynchronous=False)

    def fail(self, event: InteractionEvent) -> None:  # pragma: no cover - requires broker
        """Release a pending message after a terminal apply failure.

        The offset is intentionally NOT committed, so the broker can redeliver
        the message; we only drop the bookkeeping entry so ``_pending`` cannot
        leak on the failure path (it was previously only pruned on ack).
        """
        token = event.bus_token
        if token is None:
            return
        with self._pending_lock:
            self._pending.pop(token, None)

    def close(self) -> None:  # pragma: no cover - requires broker
        if not self._closed:
            self._closed = True
            try:
                self._consumer.close()
            except Exception:  # noqa: BLE001
                logger.exception("error closing kafka consumer")


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------
@dataclass
class IngestorMetrics:
    """Snapshot of ingestor counters.

    All counters except ``consecutive_apply_errors`` and ``degraded`` are
    monotonic. ``consecutive_apply_errors`` resets to 0 on any successful apply;
    ``degraded`` flips to True once the consecutive-error threshold trips and the
    ingestor surfaces an unhealthy signal.
    """
    events_consumed: int = 0
    events_applied: int = 0
    parse_errors: int = 0
    apply_errors: int = 0
    last_event_ts: float = 0.0
    polls: int = 0
    consecutive_apply_errors: int = 0
    degraded: bool = False

    def as_dict(self) -> dict:
        return {
            "events_consumed": self.events_consumed,
            "events_applied": self.events_applied,
            "parse_errors": self.parse_errors,
            "apply_errors": self.apply_errors,
            "last_event_ts": self.last_event_ts,
            "polls": self.polls,
            "consecutive_apply_errors": self.consecutive_apply_errors,
            "degraded": self.degraded,
        }


class StreamingIngestor:
    """Pulls events from an :class:`InteractionEventBus` into a ranker.

    The ingestor is single-threaded by design — one bus, one ranker, one
    poll loop. Horizontal scale comes from running multiple ingestor
    instances in the same consumer group; Kafka handles partition assignment.

    Two modes:

    * ``run_forever()`` — blocks the calling thread. Use from a worker process.
    * ``start()`` / ``stop()`` — spawns a daemon thread. Use from a service
      that also serves rank requests from the same ranker.

    Errors in ``ranker.observe`` are caught, logged, and counted; a single
    bad event never kills the loop. However, a *deterministically*-failing
    observe() (e.g. a model that is wedged) would otherwise silently stop the
    model from adapting while polls keep incrementing. To catch this, the
    ingestor tracks the number of *consecutive* apply errors (reset on any
    success) and, once it reaches ``max_consecutive_apply_errors``, surfaces an
    unhealthy signal: it marks readiness ``False`` (so k8s/readyz takes the pod
    out of rotation) and, if ``reraise_on_degraded`` is set, re-raises so the
    poll loop exits and orchestration restarts the worker.

    Parameters
    ----------
    bus : InteractionEventBus
    ranker : StreamingAdaptiveRanker
    batch_size : int
        Max events per poll.
    poll_timeout_s : float
        Upper bound on a single poll call's wait time. Lower values shorten
        the shutdown latency at the cost of slightly more CPU when idle.
    on_event : callable, optional
        Called with each applied event. Useful for custom telemetry hooks.
    max_consecutive_apply_errors : int
        Number of back-to-back observe() failures tolerated before the ingestor
        trips its health signal. ``0`` disables the check (legacy behaviour:
        errors are only counted). Defaults to 50.
    on_degraded : callable, optional
        Invoked once (with this ingestor) the first time the consecutive-error
        threshold trips. Defaults to flipping ``observability.set_ready(False)``.
        Pass a no-op to opt out of the default readiness side effect.
    reraise_on_degraded : bool
        When True, after tripping the health signal the worker re-raises an
        :class:`IngestorDegraded` from the drain loop so a supervisor can
        restart it. Defaults to False (stay alive, but marked unready).
    """

    def __init__(
        self,
        bus: InteractionEventBus,
        ranker: StreamingAdaptiveRanker,
        *,
        batch_size: int = 64,
        poll_timeout_s: float = 0.5,
        on_event: Optional[Callable[[InteractionEvent], None]] = None,
        max_consecutive_apply_errors: int = 50,
        on_degraded: Optional[Callable[["StreamingIngestor"], None]] = None,
        reraise_on_degraded: bool = False,
    ) -> None:
        self.bus = bus
        self.ranker = ranker
        self.batch_size = int(batch_size)
        self.poll_timeout_s = float(poll_timeout_s)
        self._on_event = on_event
        self.max_consecutive_apply_errors = int(max_consecutive_apply_errors)
        self._on_degraded = on_degraded
        self.reraise_on_degraded = bool(reraise_on_degraded)
        self.metrics = IngestorMetrics()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- internal ----
    def _trip_degraded(self) -> None:
        """Surface an unhealthy signal once the consecutive-error threshold trips.

        Idempotent: only fires the side effects on the first transition into the
        degraded state. The default action marks the service unready so health
        probes pull the worker out of rotation; callers may override via
        ``on_degraded``.
        """
        if self.metrics.degraded:
            return
        self.metrics.degraded = True
        logger.error(
            "StreamingIngestor degraded: %d consecutive apply errors "
            "(threshold=%d); observe() appears wedged, marking unhealthy",
            self.metrics.consecutive_apply_errors,
            self.max_consecutive_apply_errors,
        )
        if self._on_degraded is not None:
            try:
                self._on_degraded(self)
            except Exception:  # noqa: BLE001
                logger.exception("on_degraded hook raised")
        else:
            # Default: take the worker out of rotation via the readiness probe.
            try:
                from orchid_ranker.observability import set_ready

                set_ready(False)
            except Exception:  # noqa: BLE001
                logger.exception("failed to set readiness False on degraded ingestor")

    def _apply(self, event: InteractionEvent) -> bool:
        try:
            self.ranker.observe(
                user_id=event.user_id,
                item_id=event.item_id,
                correct=event.correct,
                skill=event.skill,
                timestamp=event.timestamp,
            )
            self.metrics.events_applied += 1
            self.metrics.last_event_ts = time.time()
            # A successful apply clears the consecutive-error streak.
            self.metrics.consecutive_apply_errors = 0
            if self._on_event is not None:
                try:
                    self._on_event(event)
                except Exception:  # noqa: BLE001
                    logger.exception("on_event hook raised")
            try:
                self.bus.ack(event)
            except Exception:  # noqa: BLE001
                logger.exception("failed to acknowledge event: %r", event)
            return True
        except Exception:  # noqa: BLE001
            self.metrics.apply_errors += 1
            self.metrics.consecutive_apply_errors += 1
            logger.exception("failed to apply event: %r", event)
            # Release any per-message bookkeeping on the bus so it cannot leak on
            # the failure path (Kafka: drops the pending entry without committing,
            # so the message is redelivered).
            try:
                self.bus.fail(event)
            except Exception:  # noqa: BLE001
                logger.exception("bus.fail hook raised for event: %r", event)
            if (
                self.max_consecutive_apply_errors > 0
                and self.metrics.consecutive_apply_errors >= self.max_consecutive_apply_errors
            ):
                self._trip_degraded()
            return False

    def _drain_once(self) -> int:
        """Poll once and apply whatever arrived. Returns events applied."""
        self.metrics.polls += 1
        parse_errors_before = int(getattr(self.bus, "parse_errors", 0))
        batch = self.bus.poll(max_events=self.batch_size, timeout_s=self.poll_timeout_s)
        parse_errors_after = int(getattr(self.bus, "parse_errors", parse_errors_before))
        if parse_errors_after > parse_errors_before:
            self.metrics.parse_errors += parse_errors_after - parse_errors_before
        self.metrics.events_consumed += len(batch)
        applied = 0
        for ev in batch:
            if self._apply(ev):
                applied += 1
                continue
            # Apply failed. Stop draining this batch if the bus wants ordered
            # delivery, or if we've just tripped the degraded health signal.
            if self.metrics.degraded or getattr(self.bus, "stop_on_apply_error", False):
                break
        # If we degraded and the caller wants a hard restart, propagate so the
        # poll loop / supervisor can act.
        if self.metrics.degraded and self.reraise_on_degraded:
            raise IngestorDegraded(
                f"{self.metrics.consecutive_apply_errors} consecutive apply errors"
            )
        return applied

    # ---- public API ----
    def drain(self, max_batches: int = 1) -> int:
        """Synchronously drain up to ``max_batches`` polls. Returns events applied.

        Useful in tests and for operators who want a short, bounded replay
        without starting a background thread.
        """
        total = 0
        for _ in range(max(1, int(max_batches))):
            total += self._drain_once()
        return total

    def run_forever(self) -> None:
        """Block the calling thread, draining the bus until :meth:`stop` is called."""
        self._stop.clear()
        while not self._stop.is_set():
            try:
                self._drain_once()
            except IngestorDegraded:
                # Health threshold tripped and reraise_on_degraded is set: exit
                # the loop so a supervisor restarts the worker. Readiness has
                # already been flipped to False by _trip_degraded().
                logger.error("ingestor poll loop exiting: degraded health signal")
                raise
            except Exception:  # noqa: BLE001
                # Bus-level errors: log and back off so we don't tight-loop.
                logger.exception("ingestor poll loop error; backing off")
                self._stop.wait(timeout=1.0)

    def start(self) -> None:
        """Start a daemon thread running :meth:`run_forever`."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_forever, name="orchid-ingestor", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Signal the poll loop to exit and optionally join the thread."""
        self._stop.set()
        # Closing the bus unblocks any in-flight poll immediately.
        try:
            self.bus.close()
        except Exception:  # noqa: BLE001
            logger.exception("error closing bus")
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None
