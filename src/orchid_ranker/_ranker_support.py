"""Internal validation and statistics helpers for the AdaptiveRanker facade."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .adaptive_learning import (
    AdaptiveLearningRecommender,
)
from .adaptive_schema import (
    LoggedDecision,
    parse_candidate_list,
)


def _require_decision_store(store: Any) -> None:
    """Fail early when a custom decision store misses the persistence contract."""
    required = (
        "get_decision",
        "get_outcome",
        "get_outcome_by_event_id",
        "create_decision",
        "attach_outcome",
        "is_outcome_applied",
        "mark_outcome_applied",
        "pending_outcomes",
        "decisions",
        "outcomes",
    )
    missing = [name for name in required if not callable(getattr(store, name, None))]
    if missing:
        raise TypeError(f"decision_store is missing required methods: {missing}")


def _observe_recommender_locally(
    recommender: AdaptiveLearningRecommender,
    *,
    user_id: Any,
    item_id: Any,
    outcome: int,
    timestamp: float,
) -> Any:
    """Advance learner-specific state while restoring aggregate counters.

    Tracer-based policies keep a learner history in their tracer, whereas the
    recommender also maintains aggregate item/concept counters used for future
    fitting and diagnostics.  Frozen pilots need the former but must retain the
    latter exactly.  ``EmpiricalTracer`` additionally owns aggregate
    global/item counts, so those are restored while its user and user-item
    counts remain advanced.
    """
    aggregate_mappings = (
        "item_support_",
        "item_correct_",
        "concept_support_",
        "concept_correct_",
    )
    mapping_snapshot = {
        name: dict(getattr(recommender, name))
        for name in aggregate_mappings
        if isinstance(getattr(recommender, name, None), dict)
    }
    scalar_names = ("_global_correct_total", "_global_outcome_count", "global_correct_")
    scalar_snapshot = {
        name: getattr(recommender, name)
        for name in scalar_names
        if hasattr(recommender, name)
    }
    tracer = recommender.tracer_
    empirical_mapping_names = ("_item_successes", "_item_count")
    empirical_mapping_snapshot = {
        name: dict(getattr(tracer, name))
        for name in empirical_mapping_names
        if isinstance(getattr(tracer, name, None), dict)
    }
    empirical_scalar_names = ("_global_successes", "_global_count")
    empirical_scalar_snapshot = {
        name: getattr(tracer, name)
        for name in empirical_scalar_names
        if hasattr(tracer, name)
    }
    try:
        return recommender.observe(user_id, item_id, outcome, timestamp=timestamp)
    finally:
        for name, snapshot in mapping_snapshot.items():
            target = getattr(recommender, name)
            target.clear()
            target.update(snapshot)
        for name, snapshot in scalar_snapshot.items():
            setattr(recommender, name, snapshot)
        for name, snapshot in empirical_mapping_snapshot.items():
            target = getattr(tracer, name)
            target.clear()
            target.update(snapshot)
        for name, snapshot in empirical_scalar_snapshot.items():
            setattr(tracer, name, snapshot)


def _decision_request_fingerprint(
    *,
    user_id: Any,
    timestamp: float,
    candidate_item_ids: Sequence[Any],
    exploration: float,
    min_item_support: float,
    min_outcome_probability: float,
    max_outcome_probability: float,
    min_difficulty: Optional[float],
    max_difficulty: Optional[float],
    require_prerequisites: bool,
    allow_unsupported_feedback: bool,
    policy_version: Optional[str],
    context_hash: str,
    concept_goal: Optional[Any],
    decision_metadata: Optional[Mapping[str, Any]],
) -> str:
    """Hash all inputs that determine a logged serving decision."""
    payload = _fingerprint_value(
        {
            "user_id": user_id,
            "timestamp": timestamp,
            "candidate_item_ids": list(candidate_item_ids),
            "exploration": float(exploration),
            "min_item_support": float(min_item_support),
            "min_outcome_probability": float(min_outcome_probability),
            "max_outcome_probability": float(max_outcome_probability),
            "min_difficulty": min_difficulty,
            "max_difficulty": max_difficulty,
            "require_prerequisites": bool(require_prerequisites),
            "allow_unsupported_feedback": bool(allow_unsupported_feedback),
            "policy_version": policy_version,
            "context_hash": context_hash,
            "concept_goal": concept_goal,
            "decision_metadata": decision_metadata,
        }
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_decision_metadata(value: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """Return a deeply JSON-compatible immutable-decision metadata payload."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("decision_metadata must be a mapping")
    return _json_metadata_mapping(value)


def _json_metadata_value(value: Any) -> Any:
    """Validate the portable metadata shape used by durable decision stores."""
    if isinstance(value, np.generic):
        return _json_metadata_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("decision_metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return _json_metadata_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_metadata_value(item) for item in value]
    raise TypeError("decision_metadata values must be JSON-compatible")


def _json_metadata_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("decision_metadata mapping keys must be strings")
        result[key] = _json_metadata_value(item)
    return result


def _require_matching_decision_request(decision: LoggedDecision, request_fingerprint: str) -> None:
    """Reject accidental reuse of an idempotency key for a different request."""
    metadata = decision.policy_metadata or {}
    stored_fingerprint = metadata.get("_orchid_request_fingerprint")
    if stored_fingerprint != request_fingerprint:
        raise ValueError(f"decision_id already exists for a different serving request: {decision.decision_id}")


def _prepare_learning_catalog(
    events: pd.DataFrame,
    *,
    item_col: str,
    category_col: Optional[str],
    difficulty_col: Optional[str],
    catalog: Optional[pd.DataFrame],
    catalog_item_col: str,
    catalog_category_col: str,
    catalog_difficulty_col: str,
) -> tuple[pd.DataFrame, Optional[str], Optional[str], Optional[pd.DataFrame]]:
    """Attach authoritative catalog metadata to historical learning attempts.

    Event-level category/difficulty columns remain supported for compatibility.
    When absent, a catalog becomes the canonical source for those values and
    also registers exercises that have not yet received learner feedback.
    """
    if catalog is None:
        return events, category_col, difficulty_col, None
    if catalog_item_col not in catalog.columns:
        raise ValueError(f"catalog must include item column {catalog_item_col!r}")
    prepared_catalog = catalog.copy()
    if prepared_catalog[catalog_item_col].isna().any():
        raise ValueError("catalog item identifiers must not be missing")
    if prepared_catalog[catalog_item_col].duplicated().any():
        raise ValueError("catalog must contain one canonical row per item")
    catalog_ids = set(prepared_catalog[catalog_item_col].tolist())
    missing_items = [item_id for item_id in events[item_col].drop_duplicates().tolist() if item_id not in catalog_ids]
    if missing_items:
        preview = ", ".join(repr(item_id) for item_id in missing_items[:5])
        raise ValueError(f"catalog is missing historical item IDs: {preview}")

    prepared_events = events.copy()
    lookup = prepared_catalog.set_index(catalog_item_col)
    resolved_category_col = category_col
    if resolved_category_col is None and catalog_category_col in prepared_catalog.columns:
        generated_category_col = _generated_catalog_column(prepared_events, "__orchid_catalog_category__")
        categories = prepared_events[item_col].map(lookup[catalog_category_col])
        if categories.isna().any():
            missing = prepared_events.loc[categories.isna(), item_col].drop_duplicates().tolist()
            preview = ", ".join(repr(item_id) for item_id in missing[:5])
            raise ValueError(f"catalog category metadata is missing for historical item IDs: {preview}")
        prepared_events[generated_category_col] = categories
        resolved_category_col = generated_category_col

    resolved_difficulty_col = difficulty_col
    if resolved_difficulty_col is None and catalog_difficulty_col in prepared_catalog.columns:
        generated_difficulty_col = _generated_catalog_column(prepared_events, "__orchid_catalog_difficulty__")
        difficulties = prepared_events[item_col].map(lookup[catalog_difficulty_col])
        if difficulties.isna().any():
            missing = prepared_events.loc[difficulties.isna(), item_col].drop_duplicates().tolist()
            preview = ", ".join(repr(item_id) for item_id in missing[:5])
            raise ValueError(f"catalog difficulty metadata is missing for historical item IDs: {preview}")
        prepared_events[generated_difficulty_col] = difficulties
        resolved_difficulty_col = generated_difficulty_col
    return prepared_events, resolved_category_col, resolved_difficulty_col, prepared_catalog


def _generated_catalog_column(events: pd.DataFrame, base: str) -> str:
    """Choose an internal metadata column that cannot overwrite caller data."""
    candidate = base
    suffix = 1
    while candidate in events.columns:
        candidate = f"{base}{suffix}"
        suffix += 1
    return candidate


def _first_metadata_value(metadata: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _binary_outcome(value: Any) -> int:
    """Validate one live binary outcome without integer truncation."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome must be exactly 0 or 1") from exc
    if not np.isfinite(numeric) or numeric not in {0.0, 1.0}:
        raise ValueError("outcome must be exactly 0 or 1")
    return int(numeric)


def _require_disjoint_policy_logs(training: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Reject repeated held-out events by ID and by their immutable event signature."""
    if "decision_id" in training.columns and "decision_id" in evaluation.columns:
        training_ids = set(training["decision_id"].astype(str))
        evaluation_ids = set(evaluation["decision_id"].astype(str))
        if training_ids.intersection(evaluation_ids):
            raise ValueError("evaluation_decisions must be disjoint from policy training")
    training_signatures = {_policy_event_signature(row) for _, row in training.iterrows()}
    evaluation_signatures = {_policy_event_signature(row) for _, row in evaluation.iterrows()}
    if training_signatures.intersection(evaluation_signatures):
        raise ValueError("evaluation_decisions must be disjoint from policy training")


def _policy_event_signature(row: pd.Series) -> str:
    """Return a duplicate-resistant signature independent of mutable decision IDs."""
    payload = {
        "user_id": row["user_id"],
        "timestamp": float(row["timestamp"]),
        "context_hash": row["context_hash"],
        "candidate_item_ids": parse_candidate_list(row["candidate_item_ids"]),
        "chosen_item_id": row["chosen_item_id"],
        "reward": float(row["reward"]),
    }
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _base_scores_for_logged_row(row: pd.Series) -> list[float]:
    """Read the base adaptive scores required to replay a hybrid+CQL action.

    Current decision records store them in immutable policy metadata.  Older
    pre-overlay records have only ``scores``; those are accepted because they
    were necessarily the base action scores.  A record made by an older CQL
    overlay without this field cannot be evaluated exactly and is rejected.
    """
    candidates = parse_candidate_list(row["candidate_item_ids"])
    metadata = _logged_policy_metadata(row)
    if "base_scores" in metadata:
        values = [float(value) for value in parse_candidate_list(metadata["base_scores"])]
        if len(values) != len(candidates) or not np.all(np.isfinite(values)):
            raise ValueError("policy_metadata.base_scores must be finite and align with candidate_item_ids")
        return values
    if "+cql" in str(row.get("policy_name", "")):
        raise ValueError(
            "hybrid+CQL logs require policy_metadata.base_scores for exact offline-policy evaluation"
        )
    values = [float(value) for value in parse_candidate_list(row["scores"])]
    if len(values) != len(candidates) or not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite and align with candidate_item_ids")
    return values


def _base_policy_version_for_logged_row(row: pd.Series) -> Optional[str]:
    metadata = _logged_policy_metadata(row)
    if "base_policy_version" in metadata:
        return str(metadata["base_policy_version"])
    # Before the first CQL overlay, the recorded policy version identifies the
    # base itself. Overlay logs need their explicit base identity.
    if "+cql" not in str(row.get("policy_name", "")):
        return str(row["policy_version"])
    return None


def _logged_policy_metadata(row: pd.Series) -> Mapping[str, Any]:
    metadata_raw = row.get("policy_metadata")
    if isinstance(metadata_raw, Mapping):
        return metadata_raw
    if isinstance(metadata_raw, str) and metadata_raw.strip():
        try:
            decoded = json.loads(metadata_raw)
        except json.JSONDecodeError as exc:
            raise ValueError("policy_metadata must be a JSON object when serialized") from exc
        if not isinstance(decoded, dict):
            raise ValueError("policy_metadata must decode to an object")
        return decoded
    return {}


def _update_digest_with_mapping(digest: Any, value: Any) -> None:
    """Hash structured learned state with stable ordering across mapping types."""
    payload = json.dumps(_fingerprint_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest.update(payload.encode("utf-8"))


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            "mapping": [
                (repr(key), _fingerprint_value(item))
                for key, item in sorted(value.items(), key=lambda entry: repr(entry[0]))
            ]
        }
    if isinstance(value, np.ndarray):
        return {"array": value.tolist()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted((_fingerprint_value(item) for item in value), key=repr)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"repr": repr(value)}


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _clamp01(value: Any) -> float:
    numeric = _optional_float(value)
    if numeric is None:
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _cold_start_outcome_prior(*, competence: Optional[float], difficulty: Optional[float]) -> float:
    if competence is None and difficulty is None:
        return 0.5
    if competence is None:
        # difficulty is non-None here (both-None handled above).
        assert difficulty is not None
        return _clamp01(1.0 - difficulty)
    if difficulty is None:
        return _clamp01(competence)
    return _clamp01(0.5 + 0.5 * (competence - difficulty))


def _decision_score_regret(row: pd.Series) -> float:
    candidates = parse_candidate_list(row["candidate_item_ids"])
    scores = [float(value) for value in parse_candidate_list(row["scores"])]
    chosen_index = candidates.index(row["chosen_item_id"])
    return float(max(scores) - scores[chosen_index])


def _chosen_predictions_and_outcomes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[float] = []
    outcomes: list[float] = []
    if frame.empty or "predicted_outcomes" not in frame.columns or "outcome" not in frame.columns:
        return np.asarray(predictions, dtype=float), np.asarray(outcomes, dtype=float)
    for _, row in frame.dropna(subset=["predicted_outcomes", "outcome"]).iterrows():
        candidates = parse_candidate_list(row["candidate_item_ids"])
        values = [float(value) for value in parse_candidate_list(row["predicted_outcomes"])]
        predictions.append(values[candidates.index(row["chosen_item_id"])])
        outcomes.append(float(row["outcome"]))
    return np.asarray(predictions, dtype=float), np.asarray(outcomes, dtype=float)


def _half_window_shift(frame: pd.DataFrame, value_col: str) -> Optional[float]:
    if frame.empty or value_col not in frame.columns:
        return None
    ordered = frame.dropna(subset=[value_col]).sort_values(["outcome_timestamp", "decision_id"], kind="mergesort")
    if len(ordered) < 2:
        return None
    midpoint = len(ordered) // 2
    early = ordered.iloc[:midpoint][value_col].astype(float)
    late = ordered.iloc[midpoint:][value_col].astype(float)
    return float(late.mean() - early.mean())


def _half_window_calibration_shift(predicted: np.ndarray, observed: np.ndarray) -> Optional[float]:
    if predicted.size < 2:
        return None
    midpoint = predicted.size // 2
    early_bias = float(np.mean(predicted[:midpoint] - observed[:midpoint]))
    late_bias = float(np.mean(predicted[midpoint:] - observed[midpoint:]))
    return late_bias - early_bias
