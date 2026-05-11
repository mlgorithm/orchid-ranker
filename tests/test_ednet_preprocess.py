from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ednet_preprocess = importlib.import_module("benchmarks.ednet.preprocess")
preprocess_ednet = ednet_preprocess.preprocess_ednet
preprocess_path = ednet_preprocess.preprocess_path


FIXTURE = Path("benchmarks/fixtures/ednet_tiny_kt1.csv")
QUESTIONS = Path("benchmarks/fixtures/ednet_tiny_questions.csv")


def test_preprocess_ednet_with_question_metadata():
    raw = pd.read_csv(FIXTURE)
    questions = pd.read_csv(QUESTIONS)

    processed = preprocess_ednet(raw, questions=questions, min_user_events=1, min_item_events=1)

    assert list(processed.columns) == ["user_id", "item_id", "correct", "timestamp", "difficulty", "elapsed_time"]
    assert len(processed) == 9
    assert processed["user_id"].nunique() == 3
    assert processed["item_id"].nunique() == 3
    assert processed["correct"].tolist() == [1, 1, 0, 0, 1, 1, 1, 0, 1]
    assert processed["difficulty"].between(0.0, 1.0).all()


def test_preprocess_ednet_accepts_denormalized_is_correct(tmp_path):
    raw = pd.read_csv(FIXTURE)
    raw["is_correct"] = [True, True, False, False, True, True, True, False, True]
    raw = raw.drop(columns=["user_answer"])
    path = tmp_path / "ednet.csv"
    output = tmp_path / "interactions.csv"
    raw.to_csv(path, index=False)

    processed = preprocess_path(path, output_path=output, min_user_events=1, min_item_events=1)

    assert output.exists()
    assert len(processed) == 9
    assert processed["correct"].sum() == 6


def test_ednet_preprocess_cli_smoke(tmp_path):
    output = tmp_path / "ednet_kt.csv"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/ednet/preprocess.py",
            "--interactions",
            str(FIXTURE),
            "--questions",
            str(QUESTIONS),
            "--min-user-events",
            "1",
            "--min-item-events",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    processed = pd.read_csv(output)
    assert len(processed) == 9
    assert "accuracy=0.6667" in result.stdout
