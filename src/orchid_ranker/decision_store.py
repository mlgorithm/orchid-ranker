"""Small durable stores for Orchid serving decisions and delayed outcomes.

The store is deliberately separate from the ranker model state.  It provides
an audit/replay boundary for immutable serving decisions and their one delayed
outcome, while applications remain free to choose how they persist learner
state and model artifacts.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Protocol

from .adaptive_schema import DecisionOutcome, LoggedDecision

__all__ = [
    "DecisionOutcomeStore",
    "InMemoryDecisionStore",
    "SQLiteDecisionStore",
]


class DecisionOutcomeStore(Protocol):
    """Persistence contract for immutable decisions and delayed outcomes.

    A successful retry returns the previously stored value with ``created`` set
    to ``False``.  A request that reuses an identifier for different immutable
    content raises :class:`ValueError` rather than overwriting audit evidence.
    """

    def get_decision(self, decision_id: str) -> Optional[LoggedDecision]: ...

    def get_outcome(self, decision_id: str) -> Optional[DecisionOutcome]: ...

    def get_outcome_by_event_id(self, outcome_event_id: str) -> Optional[DecisionOutcome]: ...

    def create_decision(self, decision: LoggedDecision) -> tuple[LoggedDecision, bool]: ...

    def attach_outcome(self, outcome: DecisionOutcome) -> tuple[DecisionOutcome, bool]: ...

    def is_outcome_applied(self, decision_id: str) -> bool: ...

    def mark_outcome_applied(self, decision_id: str) -> bool: ...

    def pending_outcomes(self) -> list[DecisionOutcome]: ...

    def decisions(self) -> list[LoggedDecision]: ...

    def outcomes(self) -> list[DecisionOutcome]: ...


class InMemoryDecisionStore:
    """Thread-safe default store retaining records only for this process."""

    def __init__(self) -> None:
        self._decisions: dict[str, LoggedDecision] = {}
        self._outcomes: dict[str, DecisionOutcome] = {}
        self._outcomes_by_event_id: dict[str, DecisionOutcome] = {}
        self._applied_outcome_decision_ids: set[str] = set()
        self._lock = threading.RLock()

    def get_decision(self, decision_id: str) -> Optional[LoggedDecision]:
        with self._lock:
            return self._decisions.get(str(decision_id))

    def get_outcome(self, decision_id: str) -> Optional[DecisionOutcome]:
        with self._lock:
            return self._outcomes.get(str(decision_id))

    def get_outcome_by_event_id(self, outcome_event_id: str) -> Optional[DecisionOutcome]:
        with self._lock:
            return self._outcomes_by_event_id.get(str(outcome_event_id))

    def create_decision(self, decision: LoggedDecision) -> tuple[LoggedDecision, bool]:
        with self._lock:
            existing = self._decisions.get(decision.decision_id)
            if existing is None:
                stored = _copy_decision(decision)
                self._decisions[stored.decision_id] = stored
                return stored, True
            _require_same_decision(existing, decision)
            return existing, False

    def attach_outcome(self, outcome: DecisionOutcome) -> tuple[DecisionOutcome, bool]:
        with self._lock:
            if outcome.decision_id not in self._decisions:
                raise KeyError(f"unknown decision_id: {outcome.decision_id}")
            existing = self._outcomes.get(outcome.decision_id)
            if existing is not None:
                _require_same_outcome(existing, outcome)
                return existing, False
            event_id = outcome.outcome_event_id
            if event_id is not None:
                linked = self._outcomes_by_event_id.get(event_id)
                if linked is not None:
                    raise ValueError(
                        "outcome_event_id already belongs to a different immutable outcome: "
                        f"{event_id}"
                    )
            stored = _copy_outcome(outcome)
            self._outcomes[stored.decision_id] = stored
            if stored.outcome_event_id is not None:
                self._outcomes_by_event_id[stored.outcome_event_id] = stored
            return stored, True

    def is_outcome_applied(self, decision_id: str) -> bool:
        with self._lock:
            return str(decision_id) in self._applied_outcome_decision_ids

    def mark_outcome_applied(self, decision_id: str) -> bool:
        with self._lock:
            resolved_decision_id = str(decision_id)
            if resolved_decision_id not in self._outcomes:
                raise KeyError(f"unknown outcome for decision_id: {resolved_decision_id}")
            if resolved_decision_id in self._applied_outcome_decision_ids:
                return False
            self._applied_outcome_decision_ids.add(resolved_decision_id)
            return True

    def pending_outcomes(self) -> list[DecisionOutcome]:
        with self._lock:
            return [
                outcome
                for decision_id, outcome in self._outcomes.items()
                if decision_id not in self._applied_outcome_decision_ids
            ]

    def decisions(self) -> list[LoggedDecision]:
        with self._lock:
            return list(self._decisions.values())

    def outcomes(self) -> list[DecisionOutcome]:
        with self._lock:
            return list(self._outcomes.values())


class SQLiteDecisionStore:
    """SQLite-backed decision store suitable for a single-host Orchid service.

    SQLite is part of Python's standard library, so this provides durable,
    transactional storage without a new runtime dependency.  It is safe for
    concurrent threads and processes that share the database file.  Each
    decision and outcome is saved as canonical JSON to retain the complete
    immutable audit record.
    """

    def __init__(self, database: str | Path) -> None:
        self.database = str(Path(database))
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database, check_same_thread=False)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orchid_decisions (
                decision_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orchid_outcomes (
                decision_id TEXT PRIMARY KEY,
                outcome_event_id TEXT UNIQUE,
                payload TEXT NOT NULL,
                FOREIGN KEY (decision_id) REFERENCES orchid_decisions(decision_id)
            )
            """
        )
        self._ensure_outcome_event_index()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orchid_outcome_applications (
                decision_id TEXT PRIMARY KEY,
                FOREIGN KEY (decision_id) REFERENCES orchid_outcomes(decision_id)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteDecisionStore":
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()

    def get_decision(self, decision_id: str) -> Optional[LoggedDecision]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM orchid_decisions WHERE decision_id = ?", (str(decision_id),)
            ).fetchone()
        return None if row is None else _decision_from_payload(row[0])

    def get_outcome(self, decision_id: str) -> Optional[DecisionOutcome]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM orchid_outcomes WHERE decision_id = ?", (str(decision_id),)
            ).fetchone()
        return None if row is None else _outcome_from_payload(row[0])

    def get_outcome_by_event_id(self, outcome_event_id: str) -> Optional[DecisionOutcome]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM orchid_outcomes WHERE outcome_event_id = ?", (str(outcome_event_id),)
            ).fetchone()
        return None if row is None else _outcome_from_payload(row[0])

    def create_decision(self, decision: LoggedDecision) -> tuple[LoggedDecision, bool]:
        payload = _canonical_payload(decision.to_dict())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT payload FROM orchid_decisions WHERE decision_id = ?", (decision.decision_id,)
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO orchid_decisions (decision_id, payload) VALUES (?, ?)",
                        (decision.decision_id, payload),
                    )
                    self._connection.commit()
                    return _copy_decision(decision), True
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        existing = _decision_from_payload(row[0])
        _require_same_decision(existing, decision)
        return existing, False

    def attach_outcome(self, outcome: DecisionOutcome) -> tuple[DecisionOutcome, bool]:
        payload = _canonical_payload(outcome.to_dict())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                decision = self._connection.execute(
                    "SELECT 1 FROM orchid_decisions WHERE decision_id = ?", (outcome.decision_id,)
                ).fetchone()
                if decision is None:
                    raise KeyError(f"unknown decision_id: {outcome.decision_id}")
                row = self._connection.execute(
                    "SELECT payload FROM orchid_outcomes WHERE decision_id = ?", (outcome.decision_id,)
                ).fetchone()
                if row is not None:
                    self._connection.commit()
                else:
                    if outcome.outcome_event_id is not None:
                        linked = self._connection.execute(
                            "SELECT decision_id FROM orchid_outcomes WHERE outcome_event_id = ?",
                            (outcome.outcome_event_id,),
                        ).fetchone()
                        if linked is not None:
                            raise ValueError(
                                "outcome_event_id already belongs to a different immutable outcome: "
                                f"{outcome.outcome_event_id}"
                            )
                    self._connection.execute(
                        "INSERT INTO orchid_outcomes (decision_id, outcome_event_id, payload) VALUES (?, ?, ?)",
                        (outcome.decision_id, outcome.outcome_event_id, payload),
                    )
                    self._connection.commit()
                    return _copy_outcome(outcome), True
            except BaseException:
                self._connection.rollback()
                raise
        existing = _outcome_from_payload(row[0])
        _require_same_outcome(existing, outcome)
        return existing, False

    def is_outcome_applied(self, decision_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM orchid_outcome_applications WHERE decision_id = ?", (str(decision_id),)
            ).fetchone()
        return row is not None

    def mark_outcome_applied(self, decision_id: str) -> bool:
        resolved_decision_id = str(decision_id)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                outcome = self._connection.execute(
                    "SELECT 1 FROM orchid_outcomes WHERE decision_id = ?", (resolved_decision_id,)
                ).fetchone()
                if outcome is None:
                    raise KeyError(f"unknown outcome for decision_id: {resolved_decision_id}")
                existing = self._connection.execute(
                    "SELECT 1 FROM orchid_outcome_applications WHERE decision_id = ?", (resolved_decision_id,)
                ).fetchone()
                if existing is not None:
                    self._connection.commit()
                    return False
                self._connection.execute(
                    "INSERT INTO orchid_outcome_applications (decision_id) VALUES (?)", (resolved_decision_id,)
                )
                self._connection.commit()
                return True
            except BaseException:
                self._connection.rollback()
                raise

    def pending_outcomes(self) -> list[DecisionOutcome]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT outcomes.payload
                FROM orchid_outcomes AS outcomes
                LEFT JOIN orchid_outcome_applications AS applications
                    ON outcomes.decision_id = applications.decision_id
                WHERE applications.decision_id IS NULL
                ORDER BY outcomes.rowid
                """
            ).fetchall()
        return [_outcome_from_payload(row[0]) for row in rows]

    def decisions(self) -> list[LoggedDecision]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM orchid_decisions ORDER BY rowid").fetchall()
        return [_decision_from_payload(row[0]) for row in rows]

    def outcomes(self) -> list[DecisionOutcome]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM orchid_outcomes ORDER BY rowid").fetchall()
        return [_outcome_from_payload(row[0]) for row in rows]

    def _ensure_outcome_event_index(self) -> None:
        """Migrate stores created before outcome-event uniqueness existed."""
        columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(orchid_outcomes)").fetchall()
        }
        if "outcome_event_id" not in columns:
            self._connection.execute("ALTER TABLE orchid_outcomes ADD COLUMN outcome_event_id TEXT")
        rows = self._connection.execute(
            "SELECT decision_id, payload FROM orchid_outcomes WHERE outcome_event_id IS NULL"
        ).fetchall()
        for decision_id, payload in rows:
            event_id = _outcome_from_payload(payload).outcome_event_id
            if event_id is not None:
                self._connection.execute(
                    "UPDATE orchid_outcomes SET outcome_event_id = ? WHERE decision_id = ?",
                    (event_id, decision_id),
                )
        duplicate = self._connection.execute(
            """
            SELECT outcome_event_id
            FROM orchid_outcomes
            WHERE outcome_event_id IS NOT NULL
            GROUP BY outcome_event_id
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise ValueError(f"stored outcomes reuse outcome_event_id: {duplicate[0]}")
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS orchid_outcomes_event_id_unique "
            "ON orchid_outcomes(outcome_event_id) WHERE outcome_event_id IS NOT NULL"
        )


def _copy_decision(decision: LoggedDecision) -> LoggedDecision:
    return LoggedDecision(**decision.to_dict())


def _copy_outcome(outcome: DecisionOutcome) -> DecisionOutcome:
    return DecisionOutcome(**outcome.to_dict())


def _canonical_payload(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def _decision_from_payload(payload: str) -> LoggedDecision:
    return LoggedDecision(**json.loads(payload))


def _outcome_from_payload(payload: str) -> DecisionOutcome:
    return DecisionOutcome(**json.loads(payload))


def _require_same_decision(existing: LoggedDecision, incoming: LoggedDecision) -> None:
    if _canonical_payload(existing.to_dict()) != _canonical_payload(incoming.to_dict()):
        raise ValueError(f"decision_id already exists with different immutable content: {incoming.decision_id}")


def _require_same_outcome(existing: DecisionOutcome, incoming: DecisionOutcome) -> None:
    if _canonical_payload(existing.to_dict()) != _canonical_payload(incoming.to_dict()):
        raise ValueError(f"decision_id already has an outcome: {incoming.decision_id}")
