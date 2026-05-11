#!/usr/bin/env python3
"""Preprocess EdNet KT1-style data into Orchid's KT benchmark schema.

Output schema:
    user_id,item_id,correct,timestamp,difficulty[,elapsed_time]

Supported raw shapes:
    - a single denormalized CSV with user/question/correctness columns
    - a KT1 directory of per-user CSV files plus optional question metadata
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

USER_COLS = ("user_id", "subject_id", "student_id")
ITEM_COLS = ("question_id", "item_id", "content_id")
CORRECT_COLS = ("correct", "is_correct", "answered_correctly")
TIME_COLS = ("timestamp", "solving_id", "order_id")


def _first_existing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _read_csv(path: Path, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=max_rows, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=max_rows, low_memory=False, encoding="latin1")


def _read_interactions(
    path: Path,
    *,
    max_rows: Optional[int] = None,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    if path.is_file():
        return _read_csv(path, max_rows=max_rows)
    if not path.is_dir():
        raise ValueError(f"interactions path does not exist: {path}")

    frames = []
    total_rows = 0
    files = sorted(path.rglob("*.csv"))
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    for csv_path in files:
        remaining = None if max_rows is None else max(0, int(max_rows) - total_rows)
        if remaining == 0:
            break
        frame = _read_csv(csv_path, max_rows=remaining)
        if "user_id" not in frame.columns:
            frame["user_id"] = csv_path.stem
        frames.append(frame)
        total_rows += len(frame)
    if not frames:
        raise ValueError(f"no CSV files found under {path}")
    return pd.concat(frames, ignore_index=True)


def _normalize_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return (numeric.astype(float) >= 0.5).astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y", "correct"}
    return lowered.isin(truthy).astype(int)


def _with_correctness(
    interactions: pd.DataFrame,
    *,
    questions: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, str]:
    correct_col = _first_existing(interactions, CORRECT_COLS)
    if correct_col is not None:
        return interactions.copy(), correct_col
    if "user_answer" not in interactions.columns:
        raise ValueError("EdNet data needs a correctness column or user_answer plus question metadata")

    if "correct_answer" in interactions.columns:
        frame = interactions.copy()
        frame["__correct__"] = (
            frame["user_answer"].astype(str).str.strip().str.lower()
            == frame["correct_answer"].astype(str).str.strip().str.lower()
        ).astype(int)
        return frame, "__correct__"

    if questions is None:
        raise ValueError("questions metadata is required when only user_answer is present")
    if not {"question_id", "correct_answer"}.issubset(questions.columns):
        raise ValueError("questions metadata must include question_id and correct_answer")
    frame = interactions.merge(
        questions[["question_id", "correct_answer"]].drop_duplicates("question_id"),
        on="question_id",
        how="left",
    )
    frame["__correct__"] = (
        frame["user_answer"].astype(str).str.strip().str.lower()
        == frame["correct_answer"].astype(str).str.strip().str.lower()
    ).astype(int)
    return frame, "__correct__"


def preprocess_ednet(
    interactions: pd.DataFrame,
    *,
    questions: Optional[pd.DataFrame] = None,
    min_user_events: int = 3,
    min_item_events: int = 2,
) -> pd.DataFrame:
    frame, correct_col = _with_correctness(interactions, questions=questions)
    user_col = _first_existing(frame, USER_COLS)
    item_col = _first_existing(frame, ITEM_COLS)
    time_col = _first_existing(frame, TIME_COLS)
    missing = [
        name
        for name, value in {
            "user column": user_col,
            "item column": item_col,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"Could not infer {', '.join(missing)} from columns: {sorted(frame.columns)}")

    out = pd.DataFrame(
        {
            "user_id": frame[user_col],
            "item_id": frame[item_col],
            "correct": _normalize_binary(frame[correct_col]),
            "timestamp": frame[time_col] if time_col is not None else np.arange(len(frame)),
        }
    )
    if "elapsed_time" in frame.columns:
        out["elapsed_time"] = pd.to_numeric(frame["elapsed_time"], errors="coerce")

    return _clean_and_filter(out, min_user_events=min_user_events, min_item_events=min_item_events)


def _clean_and_filter(
    frame: pd.DataFrame,
    *,
    min_user_events: int,
    min_item_events: int,
) -> pd.DataFrame:
    out = frame.copy()
    out = out.dropna(subset=["user_id", "item_id", "correct"]).reset_index(drop=True)
    out["user_id"] = pd.factorize(out["user_id"], sort=True)[0].astype(int)
    out["item_id"] = pd.factorize(out["item_id"], sort=True)[0].astype(int)
    out["correct"] = out["correct"].astype(int)
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    fallback_timestamp = pd.Series(np.arange(len(out)), index=out.index, dtype=float)
    out["timestamp"] = out["timestamp"].where(out["timestamp"].notna(), fallback_timestamp).astype(float)

    if min_user_events > 1:
        user_counts = out["user_id"].value_counts()
        out = out[out["user_id"].isin(user_counts[user_counts >= min_user_events].index)]
    if min_item_events > 1:
        item_counts = out["item_id"].value_counts()
        out = out[out["item_id"].isin(item_counts[item_counts >= min_item_events].index)]

    out = out.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    item_accuracy = out.groupby("item_id")["correct"].mean()
    out["difficulty"] = (1.0 - out["item_id"].map(item_accuracy).astype(float)).clip(0.0, 1.0)
    columns = ["user_id", "item_id", "correct", "timestamp", "difficulty"]
    if "elapsed_time" in out.columns:
        columns.append("elapsed_time")
    return out[columns]


def preprocess_path(
    interactions_path: Path,
    *,
    output_path: Path,
    questions_path: Optional[Path] = None,
    max_rows: Optional[int] = None,
    max_files: Optional[int] = None,
    min_user_events: int = 3,
    min_item_events: int = 2,
) -> pd.DataFrame:
    interactions = _read_interactions(interactions_path, max_rows=max_rows, max_files=max_files)
    questions = _read_csv(questions_path) if questions_path is not None else None
    processed = preprocess_ednet(
        interactions,
        questions=questions,
        min_user_events=min_user_events,
        min_item_events=min_item_events,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return processed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess EdNet KT1 data for Orchid KT benchmarks.")
    parser.add_argument("--interactions", type=Path, required=True, help="KT1 CSV or directory of per-user CSVs")
    parser.add_argument("--questions", type=Path, help="Optional EdNet questions metadata CSV")
    parser.add_argument("--output", type=Path, default=Path("data/ednet_kt/interactions.csv"))
    parser.add_argument("--max-rows", type=int, help="Optional row cap for smoke runs")
    parser.add_argument("--max-files", type=int, help="Optional file cap for KT1 directory smoke runs")
    parser.add_argument("--min-user-events", type=int, default=3)
    parser.add_argument("--min-item-events", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    processed = preprocess_path(
        args.interactions,
        output_path=args.output,
        questions_path=args.questions,
        max_rows=args.max_rows,
        max_files=args.max_files,
        min_user_events=args.min_user_events,
        min_item_events=args.min_item_events,
    )
    print(
        f"Wrote {args.output} rows={len(processed)} "
        f"users={processed.user_id.nunique()} items={processed.item_id.nunique()} "
        f"accuracy={processed.correct.mean():.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
