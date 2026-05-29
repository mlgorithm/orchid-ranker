"""Typed adaptive-learning event and logged-decision contracts."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, cast

import numpy as np
import pandas as pd

__all__ = [
    "LearnerEvent",
    "LoggedDecision",
    "hash_identifier",
    "learner_events_to_frame",
    "logged_decisions_to_frame",
    "parse_candidate_list",
    "stable_context_hash",
    "validate_learner_events",
    "validate_logged_decisions",
]


@dataclass(frozen=True)
class LearnerEvent:
    """One learner outcome event used by adaptive-learning training/serving."""

    learner_id: Any
    ts: int
    item_id: Any
    concept_id: Optional[Any]
    correct: Optional[int]
    latency_ms: Optional[int] = None
    session_id: Optional[str] = None
    item_text: Optional[str] = None
    item_meta: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class LoggedDecision:
    """One logged serving decision with candidate set and propensity."""

    learner_id: Any
    ts: int
    candidate_item_ids: Sequence[Any]
    chosen_item_id: Any
    propensity: float
    policy_name: str
    policy_version: str
    scores: Sequence[float]
    context_hash: str
    exploration_bonus: Optional[Sequence[float]] = None
    reward: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def learner_events_to_frame(events: Iterable[LearnerEvent]) -> pd.DataFrame:
    """Convert learner event dataclasses into a DataFrame and validate it."""
    frame = pd.DataFrame([event.to_dict() for event in events])
    return validate_learner_events(frame)


def logged_decisions_to_frame(decisions: Iterable[LoggedDecision]) -> pd.DataFrame:
    """Convert logged decision dataclasses into a DataFrame and validate it."""
    frame = pd.DataFrame([decision.to_dict() for decision in decisions])
    reward_col = "reward" if "reward" in frame.columns and frame["reward"].notna().all() else None
    return validate_logged_decisions(frame, reward_col=reward_col)


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


def validate_learner_events(
    events: pd.DataFrame,
    *,
    learner_col: str = "learner_id",
    ts_col: str = "ts",
    item_col: str = "item_id",
    correct_col: str = "correct",
    require_timestamp: bool = True,
) -> pd.DataFrame:
    """Validate adaptive learner events and return a defensive copy."""
    required = {learner_col, item_col, correct_col}
    if require_timestamp:
        required.add(ts_col)
    _require_columns(events, *sorted(required), frame_name="events")
    if events.empty:
        raise ValueError("events DataFrame is empty")

    work = events.copy()
    if require_timestamp:
        ts = _numeric(work[ts_col], ts_col)
        if np.any(ts < 0):
            raise ValueError(f"{ts_col} must be non-negative")
    labels = work[correct_col].dropna()
    if not labels.empty:
        values = pd.to_numeric(labels, errors="raise").to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{correct_col} values must be in [0, 1] or missing")
    return work


def validate_logged_decisions(
    decisions: pd.DataFrame,
    *,
    learner_col: str = "learner_id",
    ts_col: str = "ts",
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
        learner_col,
        ts_col,
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
        if not str(row[context_hash_col]):
            raise ValueError(f"{context_hash_col} must be non-empty at row {row_id}")
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


def _require_columns(frame: pd.DataFrame, *columns: str, frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def _numeric(values: Iterable[Any], name: str) -> np.ndarray:
    array = cast(np.ndarray, pd.to_numeric(pd.Series(list(values)), errors="raise").to_numpy(dtype=float))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array
