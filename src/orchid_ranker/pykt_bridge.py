"""Interoperability helpers for pyKT-style knowledge tracing workflows.

pyKT uses a six-line-per-learner sequence format for raw preprocessing:
learner id and sequence length, question ids, concept ids, responses,
timestamps, and answering durations. This module lets Orchid export to that
format, read it back, and consume prediction tables produced by external KT
tooling without depending on pyKT at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PyKTPredictionAdapter",
    "PyKTSequence",
    "export_pykt_sequences",
    "load_pykt_sequences",
    "pykt_sequences_to_interactions",
]


@dataclass(frozen=True)
class PyKTSequence:
    """One pyKT-style learner sequence."""

    user_id: str
    questions: tuple[str, ...]
    concepts: tuple[str, ...]
    responses: tuple[int, ...]
    timestamps: tuple[str, ...]
    durations: tuple[str, ...]

    @property
    def seq_len(self) -> int:
        return len(self.responses)


class PyKTPredictionAdapter:
    """Use exported pyKT predictions through Orchid's KT policy interface.

    The adapter expects a table with one row per user-item prediction and a
    probability-correct column. It implements ``predict_many`` and
    ``predict_correct``, so it can back `KTValuePolicy` and the OPE helpers.
    """

    def __init__(
        self,
        predictions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        probability_col: str = "p_correct",
        fallback: str = "global_mean",
    ) -> None:
        required = {user_col, item_col, probability_col}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions missing required columns: {sorted(missing)}")
        if fallback not in {"global_mean", "item_mean", "raise"}:
            raise ValueError("fallback must be 'global_mean', 'item_mean', or 'raise'")

        work = predictions[[user_col, item_col, probability_col]].copy()
        work[probability_col] = pd.to_numeric(work[probability_col], errors="coerce")
        if work[probability_col].isna().any():
            raise ValueError(f"{probability_col} contains non-numeric values")
        if not work[probability_col].between(0.0, 1.0).all():
            raise ValueError(f"{probability_col} values must be in [0, 1]")

        self.user_col = user_col
        self.item_col = item_col
        self.probability_col = probability_col
        self.fallback = fallback
        self._predictions: Dict[tuple[Any, Any], float] = {
            (row[user_col], row[item_col]): float(row[probability_col])
            for row in work.to_dict("records")
        }
        self._global_mean = float(work[probability_col].mean())
        self._item_mean = {
            item_id: float(value)
            for item_id, value in work.groupby(item_col)[probability_col].mean().items()
        }

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        probability_col: str = "p_correct",
        fallback: str = "global_mean",
    ) -> "PyKTPredictionAdapter":
        """Load exported prediction probabilities from CSV."""
        return cls(
            pd.read_csv(path),
            user_col=user_col,
            item_col=item_col,
            probability_col=probability_col,
            fallback=fallback,
        )

    def predict_many(self, user_id: Any, item_ids: Sequence[Any]) -> Dict[Any, float]:
        """Return probability-correct predictions for candidate items."""
        return {item_id: self.predict_correct(user_id, item_id) for item_id in item_ids}

    def predict_correct(self, user_id: Any, item_id: Any) -> float:
        """Return one probability-correct prediction."""
        key = (user_id, item_id)
        if key in self._predictions:
            return self._predictions[key]
        if self.fallback == "item_mean" and item_id in self._item_mean:
            return self._item_mean[item_id]
        if self.fallback == "global_mean":
            return self._global_mean
        raise KeyError(f"No pyKT prediction for user_id={user_id!r}, item_id={item_id!r}")

    def observe(self, user_id: Any, item_id: Any, correct: Any) -> None:
        """Accept live observations for compatibility with KTValuePolicy.

        Exported prediction tables are static, so observations are intentionally
        not used to update predictions.
        """
        del user_id, item_id, correct
        return None


def export_pykt_sequences(
    interactions: pd.DataFrame,
    output_path: str | Path,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    correct_col: str = "correct",
    concept_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    duration_col: Optional[str] = None,
    min_seq_len: int = 2,
    correct_threshold: float = 0.5,
) -> list[PyKTSequence]:
    """Write interactions to pyKT's six-line sequence format."""
    sequences = _build_sequences(
        interactions,
        user_col=user_col,
        item_col=item_col,
        correct_col=correct_col,
        concept_col=concept_col,
        timestamp_col=timestamp_col,
        duration_col=duration_col,
        min_seq_len=min_seq_len,
        correct_threshold=correct_threshold,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_sequences_to_text(sequences), encoding="utf8")
    return sequences


def load_pykt_sequences(path: str | Path, *, min_seq_len: int = 1) -> list[PyKTSequence]:
    """Read pyKT's six-line sequence format."""
    if min_seq_len < 1:
        raise ValueError("min_seq_len must be >= 1")
    lines = Path(path).read_text(encoding="utf8").splitlines()
    if len(lines) % 6 != 0:
        raise ValueError("pyKT sequence file must contain a multiple of 6 lines")

    sequences: list[PyKTSequence] = []
    for start in range(0, len(lines), 6):
        header = _split_line(lines[start])
        if len(header) != 2:
            raise ValueError(f"invalid pyKT header line: {lines[start]!r}")
        user_id, raw_len = header
        seq_len = int(raw_len)
        questions = _split_payload(lines[start + 1])
        concepts = _split_payload(lines[start + 2])
        responses = tuple(int(value) for value in _split_payload(lines[start + 3]))
        timestamps = _split_payload(lines[start + 4])
        durations = _split_payload(lines[start + 5])
        values = [questions, concepts, responses, timestamps, durations]
        if any(len(value) != seq_len for value in values):
            raise ValueError(f"pyKT sequence length mismatch for user_id={user_id!r}")
        if seq_len >= min_seq_len:
            sequences.append(
                PyKTSequence(
                    user_id=user_id,
                    questions=tuple(questions),
                    concepts=tuple(concepts),
                    responses=responses,
                    timestamps=tuple(timestamps),
                    durations=tuple(durations),
                )
            )
    return sequences


def pykt_sequences_to_interactions(sequences: Sequence[PyKTSequence]) -> pd.DataFrame:
    """Convert pyKT sequences into Orchid's KT interaction schema."""
    rows = []
    for sequence in sequences:
        for offset, (question, concept, response, timestamp, duration) in enumerate(
            zip(
                sequence.questions,
                sequence.concepts,
                sequence.responses,
                sequence.timestamps,
                sequence.durations,
            )
        ):
            rows.append(
                {
                    "user_id": sequence.user_id,
                    "item_id": question,
                    "correct": int(response),
                    "timestamp": _maybe_numeric(timestamp, fallback=float(offset)),
                    "concept_id": concept,
                    "duration": _maybe_numeric(duration, fallback=np.nan),
                }
            )
    return pd.DataFrame(rows)


def _build_sequences(
    interactions: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    correct_col: str,
    concept_col: Optional[str],
    timestamp_col: Optional[str],
    duration_col: Optional[str],
    min_seq_len: int,
    correct_threshold: float,
) -> list[PyKTSequence]:
    if min_seq_len < 1:
        raise ValueError("min_seq_len must be >= 1")
    required = {user_col, item_col, correct_col}
    for optional in (concept_col, timestamp_col, duration_col):
        if optional is not None:
            required.add(optional)
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions missing required columns: {sorted(missing)}")

    work = interactions.copy()
    work["__orchid_order__"] = np.arange(len(work))
    sort_cols = [user_col]
    if timestamp_col is not None:
        sort_cols.append(timestamp_col)
    sort_cols.append("__orchid_order__")
    work = work.sort_values(sort_cols, kind="mergesort")

    sequences: list[PyKTSequence] = []
    for user_id, group in work.groupby(user_col, sort=False):
        questions = tuple(str(value) for value in group[item_col].tolist())
        concepts = _column_or_na(group, concept_col, len(group))
        timestamps = _column_or_na(group, timestamp_col, len(group))
        durations = _column_or_na(group, duration_col, len(group))
        responses = tuple(_label(value, correct_threshold) for value in group[correct_col].tolist())
        if len(responses) < min_seq_len:
            continue
        sequences.append(
            PyKTSequence(
                user_id=str(user_id),
                questions=questions,
                concepts=concepts,
                responses=responses,
                timestamps=timestamps,
                durations=durations,
            )
        )
    return sequences


def _sequences_to_text(sequences: Sequence[PyKTSequence]) -> str:
    lines: list[str] = []
    for sequence in sequences:
        user_id = _validate_pykt_token(sequence.user_id, "user_id")
        questions = [_validate_pykt_token(value, "question") for value in sequence.questions]
        concepts = [_validate_pykt_token(value, "concept") for value in sequence.concepts]
        timestamps = [_validate_pykt_token(value, "timestamp") for value in sequence.timestamps]
        durations = [_validate_pykt_token(value, "duration") for value in sequence.durations]
        lines.append(f"{user_id},{sequence.seq_len}")
        lines.append(",".join(questions))
        lines.append(",".join(concepts))
        lines.append(",".join(str(value) for value in sequence.responses))
        lines.append(",".join(timestamps))
        lines.append(",".join(durations))
    return "\n".join(lines) + ("\n" if lines else "")


def _validate_pykt_token(value: Any, field: str) -> str:
    text = str(value)
    if "," in text or "\n" in text or "\r" in text:
        raise ValueError(f"{field} contains a comma or newline, which pyKT sequence export cannot encode safely")
    return text


def _column_or_na(group: pd.DataFrame, column: Optional[str], length: int) -> tuple[str, ...]:
    if column is None:
        return tuple("NA" for _ in range(length))
    return tuple(str(value) for value in group[column].tolist())


def _label(value: Any, threshold: float) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("correct labels must be finite")
    return int(numeric >= threshold)


def _split_line(line: str) -> list[str]:
    return [part.strip() for part in line.split(",")]


def _split_payload(line: str) -> tuple[str, ...]:
    values = _split_line(line)
    return tuple(values)


def _maybe_numeric(value: Any, *, fallback: float) -> float | str:
    if str(value).strip().upper() == "NA":
        return fallback
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return str(value)
    return float(numeric)
