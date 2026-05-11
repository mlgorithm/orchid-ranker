"""Tests for ASSISTments preprocessing helpers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.assistments.preprocess import preprocess_classic, preprocess_file, preprocess_foundational


def test_preprocess_classic_outputs_kt_schema():
    raw = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 1, 2, 3],
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "problem_id": ["p1", "p2", "p1", "p1", "p2", "p1"],
            "correct": [1, 0, 1, 0, 1, 0],
            "skill_id": ["s1", "s1", "s1", "s1", "s1", "s1"],
        }
    )

    processed = preprocess_classic(raw, min_user_events=2, min_item_events=2)

    assert list(processed.columns) == ["user_id", "item_id", "correct", "timestamp", "difficulty", "skill_id"]
    assert processed["user_id"].nunique() == 2
    assert processed["item_id"].nunique() == 2
    assert processed["difficulty"].between(0.0, 1.0).all()


def test_preprocess_foundational_merges_skills():
    interactions = pd.DataFrame(
        {
            "user_xid": ["a", "a", "a", "b", "b", "b"],
            "problem_id": [11, 12, 11, 11, 12, 11],
            "discrete_score": [1, 0, 1, 0, 1, 0],
            "end_time": pd.date_range("2026-01-01", periods=6, freq="h"),
        }
    )
    skills = pd.DataFrame(
        {
            "problem_id": [11, 12],
            "skill_id": ["k1", "k2"],
            "node_name": ["Unit rates", "Fractions"],
        }
    )

    processed = preprocess_foundational(interactions, skills=skills, min_user_events=2, min_item_events=2)

    assert "skill_id" in processed.columns
    assert "skill_name" in processed.columns
    assert processed["correct"].isin([0, 1]).all()


def test_preprocess_file_and_cli_on_tiny_fixture(tmp_path):
    output = tmp_path / "assistments_kt.csv"
    processed = preprocess_file(
        interactions_path=Path("benchmarks/fixtures/assistments_tiny_raw.csv"),
        output_path=output,
        fmt="classic",
        min_user_events=2,
        min_item_events=2,
    )

    assert output.exists()
    assert len(processed) == 20

    cli_output = tmp_path / "assistments_cli.csv"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/assistments/preprocess.py",
            "--interactions",
            "benchmarks/fixtures/assistments_tiny_raw.csv",
            "--format",
            "classic",
            "--output",
            str(cli_output),
            "--min-user-events",
            "2",
            "--min-item-events",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert cli_output.exists()
    assert "Wrote" in result.stdout
