"""Delayed-gain reward modeling for adaptive-learning policies."""
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from .progression_reward import ProgressionRewardConfig, expected_progression_reward

if TYPE_CHECKING:
    from .kt_benchmark import KTHoldoutSplit

__all__ = [
    "DelayedGainRewardModel",
    "DelayedGainTrainingReport",
    "build_delayed_gain_training_frame",
    "diagnose_delayed_gain_predictions",
    "fit_delayed_gain_reward_model",
    "fit_delayed_gain_reward_model_from_frame",
]

FEATURE_NAMES = [
    "p_correct",
    "difficulty",
    "competence",
    "recent_repetition",
    "progression_expected_reward",
    "progression_stretch_fit",
    "progression_mastery_gain",
    "delayed_gain_prior",
    "item_support_log",
    "concept_support_log",
    "support_score",
]


@dataclass(frozen=True)
class DelayedGainTrainingReport:
    """Summary for a delayed-gain reward model fit."""

    n_examples: int
    target_mean: float
    train_rmse: Optional[float]
    validation_rmse: Optional[float]
    validation_mae: Optional[float]
    validation_ece: Optional[float]
    validation_weighted_rmse: Optional[float]
    validation_weighted_mae: Optional[float]
    cross_fit_rmse: Optional[float]
    cross_fit_mae: Optional[float]
    cross_fit_ece: Optional[float]
    calibrated: bool
    fallback_only: bool
    example_weighting: str
    sample_weight_mean: Optional[float]
    sample_weight_max: Optional[float]
    feature_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DelayedGainRewardModel:
    """Predict bounded delayed same-concept learning gain from ranking features."""

    def __init__(
        self,
        *,
        estimator: Optional[Any],
        calibrator: Optional[Any],
        feature_names: Sequence[str],
        default_prediction: float,
        report: DelayedGainTrainingReport,
    ) -> None:
        self.estimator = estimator
        self.calibrator = calibrator
        self.feature_names = list(feature_names)
        self.default_prediction = _clamp01(default_prediction)
        self.report = report

    def predict_one(self, features: Mapping[str, float]) -> float:
        """Predict delayed-gain reward for one candidate feature mapping."""
        if self.estimator is None:
            return self.default_prediction
        row = np.asarray([[float(features.get(name, 0.0)) for name in self.feature_names]], dtype=float)
        return self._calibrate_one(float(self.estimator.predict(row)[0]))

    def predict_many(self, feature_rows: Sequence[Mapping[str, float]]) -> list[float]:
        """Predict delayed-gain reward for a batch of candidate feature mappings."""
        if not feature_rows:
            return []
        if self.estimator is None:
            return [self.default_prediction for _ in feature_rows]
        rows = np.asarray(
            [[float(features.get(name, 0.0)) for name in self.feature_names] for features in feature_rows],
            dtype=float,
        )
        return self._calibrate_many(self.estimator.predict(rows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_prediction": self.default_prediction,
            **self.report.to_dict(),
        }

    def _calibrate_one(self, value: float) -> float:
        if self.calibrator is None:
            return _clamp01(value)
        return _clamp01(float(self.calibrator.predict(np.asarray([_clamp01(value)], dtype=float))[0]))

    def _calibrate_many(self, values: Sequence[float]) -> list[float]:
        array = np.clip(np.asarray(list(values), dtype=float), 0.0, 1.0)
        if self.calibrator is not None:
            array = np.clip(self.calibrator.predict(array), 0.0, 1.0)
        return [_clamp01(float(value)) for value in array]


def fit_delayed_gain_reward_model(
    split: KTHoldoutSplit,
    *,
    concept_col: str,
    item_difficulty_col: Optional[str] = None,
    item_gain_prior: Optional[Mapping[Any, float]] = None,
    concept_gain_prior: Optional[Mapping[Any, float]] = None,
    global_gain_prior: float = 0.5,
    future_window: int = 5,
    threshold: float = 0.5,
    max_examples: Optional[int] = 50000,
    validation_fraction: float = 0.2,
    example_weighting: str = "uniform",
    sample_weight_col: Optional[str] = None,
    max_sample_weight: float = 20.0,
    cross_fit_folds: int = 1,
    random_state: Optional[int] = 42,
    config: Optional[ProgressionRewardConfig] = None,
    tracer: Optional[Any] = None,
) -> DelayedGainRewardModel:
    """Fit a direct reward model for delayed same-concept gain.

    The model is trained only on the benchmark train split. It uses sequential
    learner state features available before each logged training event, then
    predicts the bounded delayed-gain proxy used by OPE.
    """
    if concept_col not in split.train.columns:
        raise ValueError(f"concept_col={concept_col!r} must exist in the train split")
    if future_window < 1:
        raise ValueError("future_window must be >= 1")
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive when provided")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")

    examples = build_delayed_gain_training_frame(
        split,
        concept_col=concept_col,
        item_difficulty_col=item_difficulty_col,
        item_gain_prior=dict(item_gain_prior or {}),
        concept_gain_prior=dict(concept_gain_prior or {}),
        global_gain_prior=float(global_gain_prior),
        future_window=future_window,
        threshold=threshold,
        config=config or ProgressionRewardConfig(),
        tracer=tracer,
    )
    return fit_delayed_gain_reward_model_from_frame(
        examples,
        max_examples=max_examples,
        validation_fraction=validation_fraction,
        example_weighting=example_weighting,
        sample_weight_col=sample_weight_col,
        max_sample_weight=max_sample_weight,
        cross_fit_folds=cross_fit_folds,
        random_state=random_state,
    )


def build_delayed_gain_training_frame(
    split: KTHoldoutSplit,
    *,
    concept_col: str,
    item_difficulty_col: Optional[str] = None,
    item_gain_prior: Optional[Mapping[Any, float]] = None,
    concept_gain_prior: Optional[Mapping[Any, float]] = None,
    global_gain_prior: float = 0.5,
    future_window: int = 5,
    threshold: float = 0.5,
    config: Optional[ProgressionRewardConfig] = None,
    tracer: Optional[Any] = None,
) -> pd.DataFrame:
    """Build logged delayed-gain examples with reusable model features.

    This is the lower-level entry point for diagnostics and OPE-oriented direct
    model training. Callers with production propensities can join
    ``target_probability`` and ``logging_propensity`` columns onto this frame
    and then use ``example_weighting="mrdr"`` in
    :func:`fit_delayed_gain_reward_model_from_frame`.
    """
    if concept_col not in split.train.columns:
        raise ValueError(f"concept_col={concept_col!r} must exist in the train split")
    if future_window < 1:
        raise ValueError("future_window must be >= 1")

    return _build_training_examples(
        split,
        concept_col=concept_col,
        item_difficulty_col=item_difficulty_col,
        item_gain_prior=dict(item_gain_prior or {}),
        concept_gain_prior=dict(concept_gain_prior or {}),
        global_gain_prior=float(global_gain_prior),
        future_window=future_window,
        threshold=threshold,
        config=config or ProgressionRewardConfig(),
        tracer=tracer,
    )


def fit_delayed_gain_reward_model_from_frame(
    examples: pd.DataFrame,
    *,
    max_examples: Optional[int] = 50000,
    validation_fraction: float = 0.2,
    example_weighting: str = "uniform",
    sample_weight_col: Optional[str] = None,
    max_sample_weight: float = 20.0,
    cross_fit_folds: int = 1,
    random_state: Optional[int] = 42,
) -> DelayedGainRewardModel:
    """Fit a delayed-gain direct model from precomputed examples.

    ``example_weighting="mrdr"`` uses squared importance weights from
    ``target_probability / logging_propensity`` when those columns are present.
    That matches the MRDR idea of optimizing the direct model for DR usefulness
    instead of ordinary unweighted prediction loss. Public datasets usually lack
    true propensities, so the default remains uniform.
    """
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive when provided")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    if cross_fit_folds < 1:
        raise ValueError("cross_fit_folds must be >= 1")
    _validate_training_frame(examples)
    if examples.empty:
        return _fallback_model(default=0.5, n_examples=0, example_weighting=example_weighting)

    if max_examples is not None and len(examples) > max_examples:
        examples = examples.sample(n=int(max_examples), random_state=random_state).reset_index(drop=True)
    else:
        examples = examples.reset_index(drop=True)

    x = examples[FEATURE_NAMES].to_numpy(dtype=float)
    y = examples["delayed_gain_reward"].to_numpy(dtype=float)
    weights = _example_weights(
        examples,
        example_weighting=example_weighting,
        sample_weight_col=sample_weight_col,
        max_sample_weight=max_sample_weight,
    )
    default = float(np.mean(y))
    if len(examples) < 8 or float(np.std(y)) < 1e-8:
        return _fallback_model(
            default=default,
            n_examples=int(len(examples)),
            example_weighting=example_weighting,
            weights=weights,
        )

    fit_idx, validation_idx = _train_validation_indices(
        len(examples),
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    x_fit, y_fit = x[fit_idx], y[fit_idx]
    x_validation, y_validation = x[validation_idx], y[validation_idx]
    fit_weights = weights[fit_idx]
    validation_weights = weights[validation_idx]
    estimator = _make_estimator(random_state=random_state)
    estimator.fit(x_fit, y_fit, sample_weight=fit_weights)
    raw_fit = np.clip(estimator.predict(x_fit), 0.0, 1.0)
    raw_validation = np.clip(estimator.predict(x_validation), 0.0, 1.0)
    calibrator = None
    validation_metric_pred = raw_validation
    validation_metric_y = y_validation
    validation_metric_weights = validation_weights
    if len(validation_idx) >= 20:
        calibration_local, report_local = _train_validation_indices(
            len(validation_idx),
            validation_fraction=0.5,
            random_state=random_state,
        )
        calibrator = _fit_isotonic_calibrator(
            raw_validation[calibration_local],
            y_validation[calibration_local],
            sample_weight=validation_weights[calibration_local],
        )
        validation_metric_pred = _apply_calibrator(raw_validation[report_local], calibrator)
        validation_metric_y = y_validation[report_local]
        validation_metric_weights = validation_weights[report_local]
    fit_pred = _apply_calibrator(raw_fit, calibrator)
    rmse = float(np.sqrt(np.mean((fit_pred - y_fit) ** 2)))
    validation_rmse = float(np.sqrt(np.mean((validation_metric_pred - validation_metric_y) ** 2)))
    validation_mae = float(np.mean(np.abs(validation_metric_pred - validation_metric_y)))
    validation_ece = _regression_ece(validation_metric_pred, validation_metric_y)
    validation_weighted_rmse = _weighted_rmse(validation_metric_pred, validation_metric_y, validation_metric_weights)
    validation_weighted_mae = _weighted_mae(validation_metric_pred, validation_metric_y, validation_metric_weights)
    cross_fit_metrics = _cross_fit_metrics(
        x,
        y,
        weights,
        n_folds=cross_fit_folds,
        random_state=random_state,
    )
    report = DelayedGainTrainingReport(
        n_examples=int(len(examples)),
        target_mean=default,
        train_rmse=rmse,
        validation_rmse=validation_rmse,
        validation_mae=validation_mae,
        validation_ece=validation_ece,
        validation_weighted_rmse=validation_weighted_rmse,
        validation_weighted_mae=validation_weighted_mae,
        cross_fit_rmse=cross_fit_metrics.get("rmse"),
        cross_fit_mae=cross_fit_metrics.get("mae"),
        cross_fit_ece=cross_fit_metrics.get("ece"),
        calibrated=calibrator is not None,
        fallback_only=False,
        example_weighting=example_weighting,
        sample_weight_mean=float(np.mean(weights)),
        sample_weight_max=float(np.max(weights)),
        feature_names=list(FEATURE_NAMES),
    )
    return DelayedGainRewardModel(
        estimator=estimator,
        calibrator=calibrator,
        feature_names=FEATURE_NAMES,
        default_prediction=default,
        report=report,
    )


def diagnose_delayed_gain_predictions(
    labels: Sequence[float],
    predictions: Sequence[float],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return calibration and lift diagnostics for delayed-gain predictions."""
    y = np.asarray(list(labels), dtype=float)
    pred = np.clip(np.asarray(list(predictions), dtype=float), 0.0, 1.0)
    if y.shape != pred.shape:
        raise ValueError("labels and predictions must have the same length")
    if y.size == 0:
        return {
            "n": 0,
            "label_mean": None,
            "prediction_mean": None,
            "bias": None,
            "rmse": None,
            "mae": None,
            "ece": None,
            "top_bottom_lift": None,
            "bins": [],
        }
    order = np.argsort(pred)
    y_sorted = y[order]
    pred_sorted = pred[order]
    bins = []
    for bucket_id, bucket in enumerate(np.array_split(np.arange(y.size), min(max(1, n_bins), y.size)), start=1):
        if bucket.size == 0:
            continue
        bucket_y = y_sorted[bucket]
        bucket_pred = pred_sorted[bucket]
        bins.append(
            {
                "bin": bucket_id,
                "n": int(bucket.size),
                "prediction_mean": float(np.mean(bucket_pred)),
                "label_mean": float(np.mean(bucket_y)),
                "mae": float(np.mean(np.abs(bucket_pred - bucket_y))),
            }
        )
    top = bins[-1]["label_mean"] if bins else None
    bottom = bins[0]["label_mean"] if bins else None
    return {
        "n": int(y.size),
        "label_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(pred)),
        "bias": float(np.mean(pred - y)),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "ece": _regression_ece(pred, y),
        "top_bottom_lift": None if top is None or bottom is None else float(top - bottom),
        "bins": bins,
    }


def _build_training_examples(
    split: KTHoldoutSplit,
    *,
    concept_col: str,
    item_difficulty_col: Optional[str],
    item_gain_prior: Mapping[Any, float],
    concept_gain_prior: Mapping[Any, float],
    global_gain_prior: float,
    future_window: int,
    threshold: float,
    config: ProgressionRewardConfig,
    tracer: Optional[Any],
) -> pd.DataFrame:
    train = _ordered(split.train, user_col=split.user_col, timestamp_col=split.timestamp_col).reset_index(drop=True)
    train["__orchid_row_id__"] = np.arange(len(train))
    train["__orchid_label__"] = _binary_labels(train[split.correct_col].tolist(), threshold=threshold)
    tracer_predictions = _replay_tracer_predictions(
        tracer,
        train,
        user_col=split.user_col,
        item_col=split.item_col,
        label_col="__orchid_label__",
        timestamp_col=split.timestamp_col,
    )
    global_label_prior = 0.5

    rows = []
    histories: dict[Any, dict[Any, list[int]]] = {}
    recent_concepts: dict[Any, list[Any]] = {}
    item_success: dict[Any, float] = {}
    item_count: dict[Any, int] = {}
    concept_success: dict[Any, float] = {}
    concept_count: dict[Any, int] = {}
    user_concept_success: dict[tuple[Any, Any], float] = {}
    user_concept_count: dict[tuple[Any, Any], int] = {}
    total_success = 0.0
    total_count = 0
    for user_id, group in train.groupby(split.user_col, sort=False):
        user_history = histories.setdefault(user_id, {})
        user_recent = recent_concepts.setdefault(user_id, [])
        records = group.to_dict("records")
        future_by_pos = _future_same_concept(records, concept_col=concept_col, label_col="__orchid_label__", future_window=future_window)
        for pos, row in enumerate(records):
            item_id = row[split.item_col]
            concept = row[concept_col]
            label = int(row["__orchid_label__"])
            item_prior = item_success.get(item_id, 0.0) / item_count[item_id] if item_count.get(item_id, 0) else global_label_prior
            concept_prior = (
                concept_success.get(concept, 0.0) / concept_count[concept]
                if concept_count.get(concept, 0)
                else global_label_prior
            )
            user_concept_key = (row[split.user_col], concept)
            prior_success = user_concept_success.get(user_concept_key, 0.0)
            prior_count = user_concept_count.get(user_concept_key, 0)
            user_concept_label_prior = (
                prior_success / prior_count
                if prior_count
                else concept_prior
            )
            future = future_by_pos.get(pos)
            if future is not None:
                future_mean, _future_count = future
                reward_label = float(np.clip(0.5 + 0.5 * (future_mean - float(user_concept_label_prior)), 0.0, 1.0))
                competence = _competence(user_history.get(concept, []), default=config.default_competence)
                p_correct = tracer_predictions.get(
                    int(row["__orchid_row_id__"]),
                    _clamp01(
                        0.55 * competence
                        + 0.25 * float(item_prior)
                        + 0.20 * float(concept_prior)
                    ),
                )
                difficulty = _difficulty(row, item_difficulty_col, item_prior)
                repetition = sum(1 for seen in user_recent[-max(1, config.repetition_window):] if seen == concept)
                progression = expected_progression_reward(
                    p_correct=p_correct,
                    difficulty=difficulty,
                    competence=competence,
                    recent_repetition=repetition,
                    config=config,
                )
                prior = _delayed_prior(
                    item_id,
                    concept,
                    item_gain_prior=item_gain_prior,
                    concept_gain_prior=concept_gain_prior,
                    global_gain_prior=global_gain_prior,
                )
                rows.append(
                    {
                        "user_id": row[split.user_col],
                        "item_id": item_id,
                        "concept_id": concept,
                        "future_same_concept_count": int(_future_count),
                        "delayed_gain_reward": reward_label,
                        **make_delayed_gain_features(
                            p_correct=p_correct,
                            difficulty=progression.difficulty,
                            competence=competence,
                            recent_repetition=repetition,
                            progression_expected_reward=progression.expected_reward,
                            progression_stretch_fit=progression.stretch_fit,
                            progression_mastery_gain=progression.mastery_gain,
                            delayed_gain_prior=prior,
                            item_support=float(item_count.get(item_id, 0)),
                            concept_support=float(concept_count.get(concept, 0)),
                        ),
                    }
                )
            total_success += float(label)
            total_count += 1
            if total_count:
                global_label_prior = total_success / total_count
            item_success[item_id] = item_success.get(item_id, 0.0) + float(label)
            item_count[item_id] = item_count.get(item_id, 0) + 1
            concept_success[concept] = concept_success.get(concept, 0.0) + float(label)
            concept_count[concept] = concept_count.get(concept, 0) + 1
            user_concept_success[user_concept_key] = user_concept_success.get(user_concept_key, 0.0) + float(label)
            user_concept_count[user_concept_key] = user_concept_count.get(user_concept_key, 0) + 1
            user_history.setdefault(concept, []).append(label)
            if len(user_history[concept]) > 20:
                del user_history[concept][: len(user_history[concept]) - 20]
            user_recent.append(concept)
            if len(user_recent) > max(1, config.repetition_window):
                del user_recent[: len(user_recent) - max(1, config.repetition_window)]
    return pd.DataFrame(rows)


def _replay_tracer_predictions(
    tracer: Optional[Any],
    train: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    label_col: str,
    timestamp_col: Optional[str],
) -> dict[int, float]:
    if tracer is None:
        return {}
    if not hasattr(tracer, "observe"):
        raise ValueError("tracer must expose observe() for delayed-gain feature replay")

    original_histories = None
    if hasattr(tracer, "_histories"):
        original_histories = {
            user_id: list(history)
            for user_id, history in getattr(tracer, "_histories").items()
        }
        setattr(tracer, "_histories", {})
    original_history_times = None
    if hasattr(tracer, "_history_times"):
        original_history_times = {
            user_id: list(times)
            for user_id, times in getattr(tracer, "_history_times").items()
        }
        setattr(tracer, "_history_times", {})
    predictions: dict[int, float] = {}
    try:
        for user_id, group in train.groupby(user_col, sort=False):
            for row in group.to_dict("records"):
                item_id = row[item_col]
                if hasattr(tracer, "predict_correct"):
                    pred = tracer.predict_correct(user_id, item_id)
                else:
                    pred = tracer.predict_many(user_id, [item_id])[item_id]
                predictions[int(row["__orchid_row_id__"])] = _clamp01(pred)
                if timestamp_col is not None and timestamp_col in row:
                    _observe_tracer_for_replay(tracer, user_id, item_id, int(row[label_col]), timestamp=row[timestamp_col])
                else:
                    tracer.observe(user_id, item_id, int(row[label_col]))
    finally:
        if original_histories is not None:
            setattr(tracer, "_histories", original_histories)
        if original_history_times is not None:
            setattr(tracer, "_history_times", original_history_times)
    return predictions


def _observe_tracer_for_replay(
    tracer: Any,
    user_id: Any,
    item_id: Any,
    label: int,
    *,
    timestamp: Any,
) -> Any:
    try:
        params = inspect.signature(tracer.observe).parameters
    except (TypeError, ValueError):
        params = {}
    if "timestamp" in params:
        return tracer.observe(user_id, item_id, label, timestamp=timestamp)
    return tracer.observe(user_id, item_id, label)


def _future_same_concept(
    records: list[dict[str, Any]],
    *,
    concept_col: str,
    label_col: str,
    future_window: int,
) -> dict[int, tuple[float, int]]:
    by_concept: dict[Any, list[int]] = {}
    for pos, row in enumerate(records):
        by_concept.setdefault(row[concept_col], []).append(pos)
    future_by_pos: dict[int, tuple[float, int]] = {}
    for positions in by_concept.values():
        labels = [float(records[pos][label_col]) for pos in positions]
        for idx, pos in enumerate(positions[:-1]):
            future = labels[idx + 1: idx + 1 + future_window]
            if future:
                future_by_pos[pos] = (float(np.mean(future)), len(future))
    return future_by_pos


def _binary_labels(values: Sequence[Any], threshold: float = 0.5) -> np.ndarray:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("labels contain non-finite values")
    return (numeric >= threshold).astype(np.float32)


def _train_validation_indices(
    n_examples: int,
    *,
    validation_fraction: float,
    random_state: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_examples)
    rng.shuffle(indices)
    n_validation = max(1, int(round(float(validation_fraction) * n_examples)))
    n_validation = min(n_examples - 1, n_validation)
    return indices[n_validation:], indices[:n_validation]


def _validate_training_frame(examples: pd.DataFrame) -> None:
    missing = [name for name in [*FEATURE_NAMES, "delayed_gain_reward"] if name not in examples.columns]
    if missing:
        raise ValueError(f"examples missing required columns: {missing}")


def _fallback_model(
    *,
    default: float,
    n_examples: int,
    example_weighting: str,
    weights: Optional[np.ndarray] = None,
) -> DelayedGainRewardModel:
    report = DelayedGainTrainingReport(
        n_examples=int(n_examples),
        target_mean=_clamp01(default),
        train_rmse=None,
        validation_rmse=None,
        validation_mae=None,
        validation_ece=None,
        validation_weighted_rmse=None,
        validation_weighted_mae=None,
        cross_fit_rmse=None,
        cross_fit_mae=None,
        cross_fit_ece=None,
        calibrated=False,
        fallback_only=True,
        example_weighting=example_weighting,
        sample_weight_mean=None if weights is None or weights.size == 0 else float(np.mean(weights)),
        sample_weight_max=None if weights is None or weights.size == 0 else float(np.max(weights)),
        feature_names=list(FEATURE_NAMES),
    )
    return DelayedGainRewardModel(
        estimator=None,
        calibrator=None,
        feature_names=FEATURE_NAMES,
        default_prediction=default,
        report=report,
    )


def _example_weights(
    examples: pd.DataFrame,
    *,
    example_weighting: str,
    sample_weight_col: Optional[str],
    max_sample_weight: float,
) -> np.ndarray:
    if max_sample_weight <= 0.0:
        raise ValueError("max_sample_weight must be positive")
    if sample_weight_col is not None:
        if sample_weight_col not in examples.columns:
            raise ValueError(f"sample_weight_col={sample_weight_col!r} not present in examples")
        weights = examples[sample_weight_col].to_numpy(dtype=float)
    elif example_weighting == "uniform":
        weights = np.ones(len(examples), dtype=float)
    elif example_weighting == "support_inverse":
        support = np.clip(examples["support_score"].to_numpy(dtype=float), 1.0 / max_sample_weight, 1.0)
        weights = 1.0 / support
    elif example_weighting == "mrdr":
        required = ["target_probability", "logging_propensity"]
        missing = [column for column in required if column not in examples.columns]
        if missing:
            raise ValueError("example_weighting='mrdr' requires target_probability and logging_propensity columns")
        logging = np.clip(examples["logging_propensity"].to_numpy(dtype=float), 1e-6, 1.0)
        target = np.clip(examples["target_probability"].to_numpy(dtype=float), 0.0, 1.0)
        weights = (target / logging) ** 2
    else:
        raise ValueError("example_weighting must be 'uniform', 'support_inverse', or 'mrdr'")
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample weights must be finite")
    weights = np.clip(weights, 0.0, float(max_sample_weight))
    mean = float(np.mean(weights)) if weights.size else 0.0
    if mean <= 0.0:
        return np.ones(len(examples), dtype=float)
    return weights / mean


def _make_estimator(*, random_state: Optional[int]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=0.05,
        random_state=random_state,
    )


def _cross_fit_metrics(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    n_folds: int,
    random_state: Optional[int],
) -> dict[str, Optional[float]]:
    if n_folds < 2 or len(y) < max(8, n_folds * 2):
        return {"rmse": None, "mae": None, "ece": None}
    n_folds = min(int(n_folds), len(y))
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    predictions = np.full(len(y), np.nan, dtype=float)
    for fold_id, validation_idx in enumerate(np.array_split(indices, n_folds)):
        if validation_idx.size == 0:
            continue
        fit_idx = np.setdiff1d(indices, validation_idx, assume_unique=True)
        if fit_idx.size < 4:
            continue
        estimator = _make_estimator(random_state=None if random_state is None else int(random_state) + fold_id + 1)
        estimator.fit(x[fit_idx], y[fit_idx], sample_weight=weights[fit_idx])
        predictions[validation_idx] = np.clip(estimator.predict(x[validation_idx]), 0.0, 1.0)
    mask = np.isfinite(predictions)
    if not np.any(mask):
        return {"rmse": None, "mae": None, "ece": None}
    pred = predictions[mask]
    labels = y[mask]
    return {
        "rmse": float(np.sqrt(np.mean((pred - labels) ** 2))),
        "mae": float(np.mean(np.abs(pred - labels))),
        "ece": _regression_ece(pred, labels),
    }


def _fit_isotonic_calibrator(
    raw_predictions: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> Optional[IsotonicRegression]:
    if len(raw_predictions) < 10 or len(np.unique(raw_predictions)) < 3:
        return None
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_predictions, labels, sample_weight=sample_weight)
    return calibrator


def _apply_calibrator(predictions: np.ndarray, calibrator: Optional[IsotonicRegression]) -> np.ndarray:
    clipped = np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)
    if calibrator is None:
        return clipped
    return np.clip(calibrator.predict(clipped), 0.0, 1.0)


def _regression_ece(predictions: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
    pred = np.asarray(predictions, dtype=float)
    y = np.asarray(labels, dtype=float)
    if pred.size == 0:
        return 0.0
    order = np.argsort(pred)
    pred = pred[order]
    y = y[order]
    total = 0.0
    for bucket in np.array_split(np.arange(pred.size), min(n_bins, pred.size)):
        if bucket.size == 0:
            continue
        total += float(bucket.size / pred.size) * abs(float(np.mean(pred[bucket]) - np.mean(y[bucket])))
    return float(total)


def _weighted_rmse(predictions: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(_weighted_mean((np.asarray(predictions) - np.asarray(labels)) ** 2, weights)))


def _weighted_mae(predictions: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    return float(_weighted_mean(np.abs(np.asarray(predictions) - np.asarray(labels)), weights))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def make_delayed_gain_features(
    *,
    p_correct: float,
    difficulty: float,
    competence: float,
    recent_repetition: int,
    progression_expected_reward: float,
    progression_stretch_fit: float,
    progression_mastery_gain: float,
    delayed_gain_prior: float,
    item_support: float,
    concept_support: float,
    min_item_support: float = 5.0,
    min_concept_support: float = 20.0,
) -> dict[str, float]:
    """Build the numeric features shared by training and serving."""
    item_support = max(0.0, float(item_support))
    concept_support = max(0.0, float(concept_support))
    item_penalty = max(0.0, min_item_support - item_support) / max(min_item_support, 1.0)
    concept_penalty = max(0.0, min_concept_support - concept_support) / max(min_concept_support, 1.0)
    support_score = _clamp01(1.0 - 0.7 * item_penalty - 0.3 * concept_penalty)
    return {
        "p_correct": _clamp01(p_correct),
        "difficulty": _clamp01(difficulty),
        "competence": _clamp01(competence),
        "recent_repetition": float(max(0, recent_repetition)),
        "progression_expected_reward": _clamp01(progression_expected_reward),
        "progression_stretch_fit": _clamp01(progression_stretch_fit),
        "progression_mastery_gain": _clamp01(progression_mastery_gain),
        "delayed_gain_prior": _clamp01(delayed_gain_prior),
        "item_support_log": float(np.log1p(item_support)),
        "concept_support_log": float(np.log1p(concept_support)),
        "support_score": support_score,
    }


def _ordered(frame: pd.DataFrame, *, user_col: str, timestamp_col: Optional[str]) -> pd.DataFrame:
    work = frame.copy()
    work["__orchid_order__"] = np.arange(len(work))
    sort_cols = [user_col]
    if timestamp_col is not None:
        sort_cols.append(timestamp_col)
    sort_cols.append("__orchid_order__")
    return work.sort_values(sort_cols, kind="mergesort").drop(columns=["__orchid_order__"])


def _competence(history: Sequence[int], *, default: float) -> float:
    if not history:
        return _clamp01(default)
    return _clamp01(sum(history[-10:]) / len(history[-10:]))


def _difficulty(row: Mapping[str, Any], item_difficulty_col: Optional[str], item_accuracy: float) -> float:
    if item_difficulty_col is not None and item_difficulty_col in row:
        return _clamp01(float(row[item_difficulty_col]))
    return _clamp01(1.0 - float(item_accuracy))


def _delayed_prior(
    item_id: Any,
    concept: Any,
    *,
    item_gain_prior: Mapping[Any, float],
    concept_gain_prior: Mapping[Any, float],
    global_gain_prior: float,
) -> float:
    return _clamp01(item_gain_prior.get(item_id, concept_gain_prior.get(concept, global_gain_prior)))


def _clamp01(value: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        return 0.5
    return max(0.0, min(1.0, numeric))
