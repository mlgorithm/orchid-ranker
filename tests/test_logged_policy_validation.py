"""End-to-end checks for the reproducible logged-policy validation command."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _logged_decisions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = json.dumps(["old", "new"])
    scores = json.dumps([0.0, 0.0])
    random_probabilities = json.dumps([0.5, 0.5])
    adaptive_probabilities = json.dumps([0.25, 0.75])

    # The earlier window makes the simple static baseline prefer ``old``.
    for user_index in range(30):
        for step, (chosen_item_id, reward) in enumerate((("old", 1.0), ("new", 0.0))):
            rows.append(
                {
                    "decision_id": f"train-{user_index}-{step}",
                    "user_id": f"user-{user_index}",
                    "timestamp": user_index * 2 + step,
                    "candidate_item_ids": candidates,
                    "chosen_item_id": chosen_item_id,
                    "propensity": 0.5,
                    "action_probabilities": random_probabilities,
                    "policy_name": "adaptive",
                    "policy_version": "adaptive-v1",
                    "scores": scores,
                    "context_hash": f"train-context-{user_index}-{step}",
                    "reward": reward,
                }
            )

    # Later logs preserve support through exploration while favoring ``new``.
    for user_index in range(30):
        for step, (chosen_item_id, propensity, reward) in enumerate(
            (("old", 0.25, 0.0), ("new", 0.75, 1.0), ("new", 0.75, 1.0), ("new", 0.75, 1.0))
        ):
            rows.append(
                {
                    "decision_id": f"evaluate-{user_index}-{step}",
                    "user_id": f"user-{user_index}",
                    "timestamp": 1_000 + user_index * 4 + step,
                    "candidate_item_ids": candidates,
                    "chosen_item_id": chosen_item_id,
                    "propensity": propensity,
                    "action_probabilities": adaptive_probabilities,
                    "policy_name": "adaptive",
                    "policy_version": "adaptive-v1",
                    "scores": scores,
                    "context_hash": f"evaluation-context-{user_index}-{step}",
                    "reward": reward,
                }
            )
    return pd.DataFrame(rows)


def test_logged_policy_validation_produces_a_gated_reviewer_report(tmp_path: Path) -> None:
    data_path = tmp_path / "completed-decisions.jsonl"
    output_path = tmp_path / "validation.json"
    markdown_path = tmp_path / "validation.md"
    _logged_decisions().to_json(data_path, orient="records", lines=True)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/validate_logged_policy.py",
            "--data",
            str(data_path),
            "--n-bootstrap",
            "40",
            "--evaluation-fraction",
            "0.67",
            "--output",
            str(output_path),
            "--report-md",
            str(markdown_path),
            "--require-pass",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["artifact_schema"] == "orchid-logged-policy-validation/v1"
    assert payload["evidence_gate"]["allowed"] is True
    assert payload["split"]["training_end"] < payload["split"]["evaluation_start"]
    assert payload["uplift"]["bootstrap_ci_low"] > 0.0
    assert 0.20 < payload["static_baseline"]["coverage"] < 0.30
    assert "PASS — eligible for a controlled rollout" in markdown_path.read_text(encoding="utf-8")
