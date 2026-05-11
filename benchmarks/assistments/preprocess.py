#!/usr/bin/env python3
"""Preprocess ASSISTments raw files into Orchid's KT benchmark schema.

Output schema:
    user_id,item_id,correct,timestamp,difficulty[,skill_id,skill_name]

Supported raw shapes:
    classic: ASSISTments 2009/2012-style single interaction CSV.
    foundational: FoundationalASSIST Interactions.csv plus optional Skills.csv.
    auto: infer from available column names.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CLASSIC_USER_COLS = ("user_id", "user", "student_id")
CLASSIC_ITEM_COLS = ("problem_id", "assistment_id", "item_id", "question_id")
CLASSIC_CORRECT_COLS = ("correct", "first_response_correct", "score")
CLASSIC_TIME_COLS = ("order_id", "problem_log_id", "timestamp", "start_time", "end_time")


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


def _normalize_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return (numeric.astype(float) >= 0.5).astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y", "correct"}
    return lowered.isin(truthy).astype(int)


def _difficulty_from_correctness(frame: pd.DataFrame) -> pd.Series:
    item_accuracy = frame.groupby("item_id")["correct"].mean()
    difficulty = 1.0 - frame["item_id"].map(item_accuracy).astype(float)
    return difficulty.clip(0.0, 1.0)


def preprocess_classic(
    interactions: pd.DataFrame,
    *,
    min_user_events: int = 3,
    min_item_events: int = 2,
) -> pd.DataFrame:
    user_col = _first_existing(interactions, CLASSIC_USER_COLS)
    item_col = _first_existing(interactions, CLASSIC_ITEM_COLS)
    correct_col = _first_existing(interactions, CLASSIC_CORRECT_COLS)
    time_col = _first_existing(interactions, CLASSIC_TIME_COLS)
    missing = [
        name
        for name, value in {
            "user column": user_col,
            "item column": item_col,
            "correct column": correct_col,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"Could not infer {', '.join(missing)} from columns: {sorted(interactions.columns)}")

    out = pd.DataFrame(
        {
            "user_id": interactions[user_col],
            "item_id": interactions[item_col],
            "correct": _normalize_binary(interactions[correct_col]),
            "timestamp": interactions[time_col] if time_col is not None else np.arange(len(interactions)),
        }
    )
    for optional in ("skill_id", "skill_name"):
        if optional in interactions.columns:
            out[optional] = interactions[optional]

    return _clean_and_filter(out, min_user_events=min_user_events, min_item_events=min_item_events)


def preprocess_foundational(
    interactions: pd.DataFrame,
    *,
    skills: Optional[pd.DataFrame] = None,
    min_user_events: int = 3,
    min_item_events: int = 2,
) -> pd.DataFrame:
    required = {"user_xid", "problem_id", "discrete_score"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"FoundationalASSIST interactions missing columns: {sorted(missing)}")

    timestamp_col = "end_time" if "end_time" in interactions.columns else None
    out = pd.DataFrame(
        {
            "user_id": interactions["user_xid"],
            "item_id": interactions["problem_id"],
            "correct": _normalize_binary(interactions["discrete_score"]),
            "timestamp": interactions[timestamp_col] if timestamp_col else np.arange(len(interactions)),
        }
    )
    if skills is not None:
        skill_cols = [col for col in ("problem_id", "skill_id", "node_code", "node_name") if col in skills.columns]
        if "problem_id" in skill_cols:
            skill_map = skills[skill_cols].drop_duplicates("problem_id")
            out = out.merge(skill_map, left_on="item_id", right_on="problem_id", how="left")
            out = out.drop(columns=["problem_id"])
            rename = {"node_name": "skill_name"}
            out = out.rename(columns=rename)

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
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").astype("int64", errors="ignore")
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
    out["difficulty"] = _difficulty_from_correctness(out)
    columns = ["user_id", "item_id", "correct", "timestamp", "difficulty"]
    for optional in ("skill_id", "skill_name", "node_code"):
        if optional in out.columns:
            columns.append(optional)
    return out[columns]


def preprocess_file(
    interactions_path: Path,
    *,
    output_path: Path,
    fmt: str = "auto",
    skills_path: Optional[Path] = None,
    max_rows: Optional[int] = None,
    min_user_events: int = 3,
    min_item_events: int = 2,
) -> pd.DataFrame:
    interactions = _read_csv(interactions_path, max_rows=max_rows)
    skills = _read_csv(skills_path) if skills_path is not None else None
    inferred = fmt
    if fmt == "auto":
        inferred = "foundational" if {"user_xid", "problem_id", "discrete_score"}.issubset(interactions.columns) else "classic"

    if inferred == "foundational":
        processed = preprocess_foundational(
            interactions,
            skills=skills,
            min_user_events=min_user_events,
            min_item_events=min_item_events,
        )
    elif inferred == "classic":
        processed = preprocess_classic(
            interactions,
            min_user_events=min_user_events,
            min_item_events=min_item_events,
        )
    else:
        raise ValueError("fmt must be 'auto', 'classic', or 'foundational'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return processed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ASSISTments data for Orchid KT benchmarks.")
    parser.add_argument("--interactions", type=Path, required=True, help="Raw interactions CSV")
    parser.add_argument("--skills", type=Path, help="Optional FoundationalASSIST Skills.csv")
    parser.add_argument("--format", choices=["auto", "classic", "foundational"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("data/assistments_kt/interactions.csv"))
    parser.add_argument("--max-rows", type=int, help="Optional row cap for smoke runs")
    parser.add_argument("--min-user-events", type=int, default=3)
    parser.add_argument("--min-item-events", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    processed = preprocess_file(
        args.interactions,
        output_path=args.output,
        fmt=args.format,
        skills_path=args.skills,
        max_rows=args.max_rows,
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
