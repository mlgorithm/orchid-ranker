"""Leakage-safe benchmarking helpers for knowledge tracing models."""
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .evaluation import expected_calibration_error
from .kt import AKTTracer, SAKTTracer

__all__ = [
    "KTHoldoutSplit",
    "KTEvaluationReport",
    "time_ordered_user_split",
    "evaluate_tracer_replay",
    "evaluate_sakt_replay",
    "evaluate_akt_replay",
    "evaluate_item_mean_baseline",
    "run_kt_benchmark",
    "run_sakt_benchmark",
    "run_akt_benchmark",
]


@dataclass(frozen=True)
class KTHoldoutSplit:
    """Train/test split for leakage-safe knowledge tracing evaluation."""

    train: pd.DataFrame
    test: pd.DataFrame
    user_col: str
    item_col: str
    correct_col: str
    timestamp_col: Optional[str] = None


@dataclass(frozen=True)
class KTEvaluationReport:
    """Prediction metrics for a knowledge tracing replay."""

    n_events: int
    accuracy: float
    auc: float
    brier: float
    log_loss: float
    ece: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_frame(
    frame: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    correct_col: str,
    timestamp_col: Optional[str],
) -> None:
    required = {user_col, item_col, correct_col}
    if timestamp_col is not None:
        required.add(timestamp_col)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"interactions missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("interactions DataFrame is empty")


def _ordered(
    frame: pd.DataFrame,
    *,
    user_col: str,
    timestamp_col: Optional[str],
) -> pd.DataFrame:
    work = frame.copy()
    work["__orchid_order__"] = np.arange(len(work))
    sort_cols = [user_col]
    if timestamp_col is not None:
        sort_cols.append(timestamp_col)
    sort_cols.append("__orchid_order__")
    return work.sort_values(sort_cols, kind="mergesort").drop(columns=["__orchid_order__"])


def time_ordered_user_split(
    interactions: pd.DataFrame,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    correct_col: str = "correct",
    timestamp_col: Optional[str] = None,
    test_fraction: float = 0.2,
    min_train_events: int = 2,
    min_test_events: int = 1,
    filter_unknown_items: bool = True,
) -> KTHoldoutSplit:
    """Split each learner sequence into past train and future test events.

    Every test event for a learner occurs after that learner's training events.
    When ``filter_unknown_items`` is true, held-out events whose item was never
    seen in training are dropped because ID-based KT models cannot score them.
    """
    _validate_frame(
        interactions,
        user_col=user_col,
        item_col=item_col,
        correct_col=correct_col,
        timestamp_col=timestamp_col,
    )
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if min_train_events < 1:
        raise ValueError("min_train_events must be >= 1")
    if min_test_events < 1:
        raise ValueError("min_test_events must be >= 1")

    ordered = _ordered(interactions, user_col=user_col, timestamp_col=timestamp_col)
    train_parts = []
    test_parts = []
    for _user_id, group in ordered.groupby(user_col, sort=False):
        n = len(group)
        min_total = min_train_events + min_test_events
        if n < min_total:
            continue
        test_count = max(min_test_events, int(np.ceil(n * test_fraction)))
        test_count = min(test_count, n - min_train_events)
        if test_count < min_test_events:
            continue
        split_at = n - test_count
        train_parts.append(group.iloc[:split_at])
        test_parts.append(group.iloc[split_at:])

    if not train_parts or not test_parts:
        raise ValueError("no users had enough events for the requested split")

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    if filter_unknown_items:
        known_items = set(train[item_col].drop_duplicates().tolist())
        test = test[test[item_col].isin(known_items)].reset_index(drop=True)
        if test.empty:
            raise ValueError("test split is empty after filtering unknown items")

    return KTHoldoutSplit(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        user_col=user_col,
        item_col=item_col,
        correct_col=correct_col,
        timestamp_col=timestamp_col,
    )


def _binary_labels(values: Sequence[Any], threshold: float = 0.5) -> np.ndarray:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("labels contain non-finite values")
    return (numeric >= threshold).astype(np.float32)


def _auc(labels: np.ndarray, preds: np.ndarray) -> float:
    positives = preds[labels == 1]
    negatives = preds[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    wins = 0.0
    total = 0
    for pos in positives:
        wins += float(np.sum(pos > negatives))
        wins += 0.5 * float(np.sum(pos == negatives))
        total += int(negatives.size)
    return float(wins / total) if total else float("nan")


def _report(labels: np.ndarray, preds: np.ndarray, *, decision_threshold: float = 0.5) -> KTEvaluationReport:
    if labels.size == 0:
        raise ValueError("cannot evaluate an empty replay")
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    clipped = np.clip(preds, 1e-7, 1.0 - 1e-7)
    accuracy = float(np.mean((preds >= decision_threshold) == labels))
    brier = float(np.mean((preds - labels) ** 2))
    log_loss = float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))
    return KTEvaluationReport(
        n_events=int(labels.size),
        accuracy=accuracy,
        auc=_auc(labels, preds),
        brier=brier,
        log_loss=log_loss,
        ece=expected_calibration_error(preds.astype(float), labels.astype(float), bins=10),
    )


def _observe_with_optional_timestamp(
    tracer: Any,
    user_id: Any,
    item_id: Any,
    label: int,
    *,
    timestamp: Optional[Any],
) -> Any:
    if timestamp is None:
        return tracer.observe(user_id, item_id, label)
    try:
        supports_timestamp = "timestamp" in inspect.signature(tracer.observe).parameters
    except (TypeError, ValueError):
        supports_timestamp = False
    if supports_timestamp:
        return tracer.observe(user_id, item_id, label, timestamp=timestamp)
    return tracer.observe(user_id, item_id, label)


def evaluate_tracer_replay(
    tracer: SAKTTracer,
    split: KTHoldoutSplit,
    *,
    threshold: float = 0.5,
) -> KTEvaluationReport:
    """Evaluate a fitted KT tracer by chronological teacher-forced replay.

    For each held-out event, the tracer predicts correctness first. Only after
    prediction does the evaluator call ``observe`` with the held-out label.
    """
    if not tracer.is_fitted:
        raise RuntimeError("tracer must be fitted before replay evaluation")

    test = _ordered(split.test, user_col=split.user_col, timestamp_col=split.timestamp_col)
    preds = []
    labels = []
    columns = [split.user_col, split.item_col, split.correct_col]
    if split.timestamp_col is not None:
        columns.append(split.timestamp_col)
    for row in test[columns].itertuples(index=False, name=None):
        user_id, item_id, raw_correct = row[:3]
        timestamp = row[3] if split.timestamp_col is not None else None
        label = int(float(raw_correct) >= threshold)
        preds.append(tracer.predict_correct(user_id, item_id))
        labels.append(label)
        _observe_with_optional_timestamp(tracer, user_id, item_id, label, timestamp=timestamp)
    return _report(np.asarray(labels, dtype=np.float32), np.asarray(preds, dtype=np.float32))


def evaluate_sakt_replay(
    tracer: SAKTTracer,
    split: KTHoldoutSplit,
    *,
    threshold: float = 0.5,
) -> KTEvaluationReport:
    """Evaluate a fitted SAKT tracer by chronological teacher-forced replay."""
    return evaluate_tracer_replay(tracer, split, threshold=threshold)


def evaluate_akt_replay(
    tracer: AKTTracer,
    split: KTHoldoutSplit,
    *,
    threshold: float = 0.5,
) -> KTEvaluationReport:
    """Evaluate a fitted AKT-inspired tracer by chronological teacher-forced replay."""
    return evaluate_tracer_replay(tracer, split, threshold=threshold)


def evaluate_item_mean_baseline(
    split: KTHoldoutSplit,
    *,
    threshold: float = 0.5,
    smoothing: float = 1.0,
) -> KTEvaluationReport:
    """Evaluate an item-mean correctness baseline on the same held-out events."""
    train = split.train.copy()
    train["__label__"] = _binary_labels(train[split.correct_col].tolist(), threshold=threshold)
    global_mean = float(train["__label__"].mean())
    grouped = train.groupby(split.item_col)["__label__"].agg(["sum", "count"])
    item_mean = {
        item_id: float((row["sum"] + smoothing * global_mean) / (row["count"] + smoothing))
        for item_id, row in grouped.iterrows()
    }

    test = _ordered(split.test, user_col=split.user_col, timestamp_col=split.timestamp_col)
    preds = [item_mean.get(item_id, global_mean) for item_id in test[split.item_col].tolist()]
    labels = _binary_labels(test[split.correct_col].tolist(), threshold=threshold)
    return _report(labels, np.asarray(preds, dtype=np.float32))


def run_kt_benchmark(
    interactions: pd.DataFrame,
    *,
    model: str = "sakt",
    user_col: str = "user_id",
    item_col: str = "item_id",
    correct_col: str = "correct",
    timestamp_col: Optional[str] = None,
    item_difficulty_col: Optional[str] = None,
    test_fraction: float = 0.2,
    max_seq_len: int = 50,
    d_model: int = 64,
    n_heads: int = 4,
    epochs: int = 5,
    batch_size: int = 128,
    random_state: Optional[int] = 42,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit a KT model on a time split and compare against an item-mean baseline."""
    split = time_ordered_user_split(
        interactions,
        user_col=user_col,
        item_col=item_col,
        correct_col=correct_col,
        timestamp_col=timestamp_col,
        test_fraction=test_fraction,
    )
    normalized = model.lower().replace("_", "-")
    if normalized == "sakt":
        tracer = SAKTTracer(
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
        ).fit(
            split.train,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
        )
        model_key = "sakt"
    elif normalized in {"akt", "akt-inspired"}:
        tracer = AKTTracer(
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=device,
        ).fit(
            split.train,
            user_col=user_col,
            item_col=item_col,
            correct_col=correct_col,
            timestamp_col=timestamp_col,
            item_difficulty_col=item_difficulty_col,
        )
        model_key = "akt"
    else:
        raise ValueError("model must be 'sakt' or 'akt'")

    kt_report = evaluate_tracer_replay(tracer, split)
    baseline = evaluate_item_mean_baseline(split)
    return {
        model_key: kt_report.to_dict(),
        "item_mean": baseline.to_dict(),
        "split": {
            "train_events": float(len(split.train)),
            "test_events": float(len(split.test)),
            "train_users": float(split.train[user_col].nunique()),
            "test_users": float(split.test[user_col].nunique()),
            "train_items": float(split.train[item_col].nunique()),
            "test_items": float(split.test[item_col].nunique()),
        },
    }


def run_sakt_benchmark(
    interactions: pd.DataFrame,
    **kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    """Fit SAKT on a time split and compare against an item-mean baseline."""
    kwargs.pop("model", None)
    return run_kt_benchmark(interactions, model="sakt", **kwargs)


def run_akt_benchmark(
    interactions: pd.DataFrame,
    **kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    """Fit AKT-inspired tracing on a time split and compare against item mean."""
    kwargs.pop("model", None)
    return run_kt_benchmark(interactions, model="akt", **kwargs)
