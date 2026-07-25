"""Typed event and logged-decision contracts for adaptive recommendation."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence, cast

import numpy as np
import pandas as pd

__all__ = [
    "DecisionOutcome",
    "LoggedDecision",
    "UserEvent",
    "decision_outcomes_to_frame",
    "hash_identifier",
    "logged_decisions_to_frame",
    "parse_candidate_list",
    "stable_context_hash",
    "validate_decision_outcomes",
    "validate_logged_decisions",
    "validate_user_events",
    "user_events_to_frame",
]


@dataclass(frozen=True)
class UserEvent:
    """One outcome event used for adaptive training or live updates."""

    user_id: Any
    timestamp: int
    item_id: Any
    outcome: Optional[int]
    category_id: Optional[Any] = None
    latency_ms: Optional[int] = None
    session_id: Optional[str] = None
    item_text: Optional[str] = None
    item_meta: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class LoggedDecision:
    """One logged serving decision with candidate set and propensity."""

    user_id: Any
    timestamp: int
    candidate_item_ids: Sequence[Any]
    chosen_item_id: Any
    propensity: float
    policy_name: str
    policy_version: str
    scores: Sequence[float]
    context_hash: str
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    action_probabilities: Optional[Sequence[float]] = None
    predicted_outcomes: Optional[Sequence[float]] = None
    exploration_rate: float = 0.0
    was_exploration: bool = False
    exploration_bonus: Optional[Sequence[float]] = None
    policy_metadata: Optional[Mapping[str, Any]] = None
    reward: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class DecisionOutcome:
    """One delayed outcome linked immutably to a serving decision."""

    decision_id: str
    user_id: Any
    item_id: Any
    outcome_timestamp: int
    outcome: Optional[int] = None
    reward: Optional[float] = None
    category_id: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def user_events_to_frame(events: Iterable[UserEvent]) -> pd.DataFrame:
    """Convert user event dataclasses into a DataFrame and validate it."""
    frame = pd.DataFrame([event.to_dict() for event in events])
    return validate_user_events(frame)


def logged_decisions_to_frame(decisions: Iterable[LoggedDecision]) -> pd.DataFrame:
    """Convert logged decision dataclasses into a DataFrame and validate it."""
    frame = pd.DataFrame([decision.to_dict() for decision in decisions])
    reward_col = "reward" if "reward" in frame.columns and frame["reward"].notna().all() else None
    return validate_logged_decisions(frame, reward_col=reward_col)


def decision_outcomes_to_frame(outcomes: Iterable[DecisionOutcome]) -> pd.DataFrame:
    """Convert decision outcomes into a validated DataFrame."""
    return validate_decision_outcomes(pd.DataFrame([outcome.to_dict() for outcome in outcomes]))


def hash_identifier(value: Any, *, salt: Optional[str] = None) -> str:
    """Return a stable salted SHA-256 hash for privacy-preserving IDs."""
    if salt is None:
        salt = os.environ.get("ORCHID_HASH_SALT")
    if not salt:
        raise ValueError("hash_identifier requires a secret salt or ORCHID_HASH_SALT")
    payload = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_context_hash(*parts: Any, salt: str = "orchid-context") -> str:
    """Build a deterministic context hash from JSON-serializable parts."""
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hash_identifier(payload, salt=salt)


def validate_user_events(
    events: pd.DataFrame,
    *,
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
    item_col: str = "item_id",
    outcome_col: str = "outcome",
    require_timestamp: bool = True,
) -> pd.DataFrame:
    """Validate adaptive user events and return a defensive copy."""
    required = {user_col, item_col, outcome_col}
    if require_timestamp:
        required.add(timestamp_col)
    _require_columns(events, *sorted(required), frame_name="events")
    if events.empty:
        raise ValueError("events DataFrame is empty")

    work = events.copy()
    if require_timestamp:
        timestamps = _numeric(work[timestamp_col], timestamp_col)
        if np.any(timestamps < 0):
            raise ValueError(f"{timestamp_col} must be non-negative")
    labels = work[outcome_col].dropna()
    if not labels.empty:
        values = pd.to_numeric(labels, errors="raise").to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{outcome_col} values must be in [0, 1] or missing")
    return work


def validate_logged_decisions(
    decisions: pd.DataFrame,
    *,
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
    candidate_col: str = "candidate_item_ids",
    chosen_col: str = "chosen_item_id",
    propensity_col: str = "propensity",
    policy_name_col: str = "policy_name",
    policy_version_col: str = "policy_version",
    scores_col: str = "scores",
    context_hash_col: str = "context_hash",
    reward_col: Optional[str] = None,
) -> pd.DataFrame:
    """Validate logged decisions for OPE and offline policy learning.

    Every row must contain the served candidate set, chosen action, chosen
    probability, policy identity, scores, and context hash. If ``reward_col`` is
    supplied, rewards must be finite.
    """
    required = [
        user_col,
        timestamp_col,
        candidate_col,
        chosen_col,
        propensity_col,
        policy_name_col,
        policy_version_col,
        scores_col,
        context_hash_col,
    ]
    if reward_col is not None:
        required.append(reward_col)
    _require_columns(decisions, *required, frame_name="decisions")
    if decisions.empty:
        raise ValueError("decisions DataFrame is empty")

    work = decisions.copy()
    if "decision_id" in work.columns:
        identifiers = work["decision_id"].astype(str)
        if (identifiers.str.len() == 0).any():
            raise ValueError("decision_id must be non-empty")
        if identifiers.duplicated().any():
            raise ValueError("decision_id must be unique")
    propensities = _numeric(work[propensity_col], propensity_col)
    if np.any((propensities <= 0.0) | (propensities > 1.0)):
        raise ValueError(f"{propensity_col} values must be in (0, 1]")
    if reward_col is not None:
        _numeric(work[reward_col], reward_col)

    for row_id, row in work.iterrows():
        candidates = parse_candidate_list(row[candidate_col])
        if not candidates:
            raise ValueError(f"{candidate_col} must be non-empty at row {row_id}")
        chosen = row[chosen_col]
        if chosen not in candidates:
            raise ValueError(f"{chosen_col} must appear in {candidate_col} at row {row_id}")
        scores = _parse_float_list(row[scores_col])
        if len(scores) != len(candidates):
            raise ValueError(f"{scores_col} length must match {candidate_col} at row {row_id}")
        if "action_probabilities" in work.columns and not _is_missing_scalar(row["action_probabilities"]):
            probabilities = _parse_float_list(row["action_probabilities"])
            if len(probabilities) != len(candidates):
                raise ValueError(f"action_probabilities length must match {candidate_col} at row {row_id}")
            if any(value < 0.0 or value > 1.0 for value in probabilities):
                raise ValueError(f"action_probabilities values must be in [0, 1] at row {row_id}")
            if not np.isclose(sum(probabilities), 1.0, atol=1e-8):
                raise ValueError(f"action_probabilities must sum to 1 at row {row_id}")
            chosen_probability = probabilities[candidates.index(chosen)]
            if not np.isclose(chosen_probability, float(row[propensity_col]), atol=1e-8):
                raise ValueError(f"{propensity_col} must equal the chosen action probability at row {row_id}")
        if "predicted_outcomes" in work.columns and not _is_missing_scalar(row["predicted_outcomes"]):
            predictions = _parse_float_list(row["predicted_outcomes"])
            if len(predictions) != len(candidates):
                raise ValueError(f"predicted_outcomes length must match {candidate_col} at row {row_id}")
            if any(value < 0.0 or value > 1.0 for value in predictions):
                raise ValueError(f"predicted_outcomes values must be in [0, 1] at row {row_id}")
        if "exploration_rate" in work.columns and not _is_missing_scalar(row["exploration_rate"]):
            exploration_rate = float(row["exploration_rate"])
            if not 0.0 <= exploration_rate <= 1.0:
                raise ValueError(f"exploration_rate must be in [0, 1] at row {row_id}")
        if not str(row[context_hash_col]):
            raise ValueError(f"{context_hash_col} must be non-empty at row {row_id}")
    return work


def validate_decision_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Validate delayed outcomes linked to immutable decision IDs."""
    required = ["decision_id", "user_id", "item_id", "outcome_timestamp", "outcome", "reward"]
    _require_columns(outcomes, *required, frame_name="outcomes")
    if outcomes.empty:
        raise ValueError("outcomes DataFrame is empty")
    work = outcomes.copy()
    identifiers = work["decision_id"].astype(str)
    if (identifiers.str.len() == 0).any() or identifiers.duplicated().any():
        raise ValueError("outcome decision_id values must be non-empty and unique")
    timestamps = _numeric(work["outcome_timestamp"], "outcome_timestamp")
    if np.any(timestamps < 0):
        raise ValueError("outcome_timestamp must be non-negative")
    for column in ("outcome", "reward"):
        present = work[column].dropna()
        if present.empty:
            continue
        values = _numeric(present, column)
        if column == "outcome" and np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("outcome values must be in [0, 1] or missing")
    if work[["outcome", "reward"]].isna().all(axis=1).any():
        raise ValueError("each decision outcome requires outcome or reward")
    return work


def parse_candidate_list(value: Any) -> list[Any]:
    """Parse a candidate-set cell from a list-like object or JSON string."""
    if isinstance(value, np.ndarray):
        return list(value.tolist())
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("candidate JSON must decode to a list")
            return list(parsed)
        return [part.strip() for part in text.split(",") if part.strip()]
    raise ValueError(f"candidate list must be list-like or JSON string, got {type(value).__name__}")


def _parse_float_list(value: Any) -> list[float]:
    items = parse_candidate_list(value)
    values = [float(item) for item in items]
    if not np.all(np.isfinite(values)):
        raise ValueError("score list contains non-finite values")
    return values


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    return bool(isinstance(value, float) and np.isnan(value))


def _require_columns(frame: pd.DataFrame, *columns: str, frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def _numeric(values: Iterable[Any], name: str) -> np.ndarray:
    array = cast(np.ndarray, pd.to_numeric(pd.Series(list(values)), errors="raise").to_numpy(dtype=float))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array
