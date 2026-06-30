"""Smoke + validation tests for the progression-reward evidence benchmark."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from orchid_ranker.progression_reward import ProgressionRewardConfig

pytest.importorskip("torch")

_TERMS = (
    "expected_outcome_value",
    "mastery_gain",
    "stretch_fit",
    "difficulty_bonus",
    "easy_penalty",
    "hard_penalty",
    "repetition_penalty",
)


def _write_synth(path, *, seed: int = 0) -> None:
    """Synthetic adaptive-learning logs with same-concept repetition and learning."""
    rng = np.random.default_rng(seed)
    concepts = {10: ("a", 0.25), 20: ("a", 0.45), 30: ("b", 0.55), 40: ("b", 0.70), 50: ("c", 0.35), 60: ("c", 0.80)}
    rows = []
    for user_id in range(1, 41):
        ability = rng.uniform(0.2, 0.9)
        for ts in range(14):
            item = int(rng.choice(list(concepts)))
            skill, diff = concepts[item]
            p = 1.0 / (1.0 + np.exp(-(ability - diff) * 6.0))
            correct = int(rng.random() < p)
            ability = min(0.98, ability + (0.02 if correct else -0.005))
            rows.append(
                {"user_id": user_id, "item_id": item, "correct": correct,
                 "difficulty": diff, "skill_id": skill, "timestamp": ts}
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _run(tmp_path, *extra):
    data = tmp_path / "synth.csv"
    _write_synth(data)
    proc = subprocess.run(
        [sys.executable, "benchmarks/progression_reward_evidence.py",
         "--data", str(data), "--concept-col", "skill_id",
         "--epochs", "1", "--max-events", "800", "--bootstrap", "50", *extra],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_evidence_benchmark_cli_smoke(tmp_path):
    result = _run(tmp_path, "--seeds", "11")
    assert result["assumptions"]["confound_controls"]
    run = result["runs"][0]
    assert run["n_with_realized_gain"] >= 5, "synthetic data too small to exercise analysis"

    overall = run["overall_spearman"]["estimate"]
    partial = run["partial_spearman_given_pcorrect_competence"]["estimate"]
    for corr in (overall, partial, run["ablations"]["correctness_only_spearman"]):
        assert np.isnan(corr) or -1.0 <= corr <= 1.0

    # per-term diagnosis present and complete, with documented signs
    assert set(run["per_term"]) == set(_TERMS)
    assert run["per_term"]["mastery_gain"]["sign_in_reward"] == 1
    assert run["per_term"]["easy_penalty"]["sign_in_reward"] == -1

    assert isinstance(run["verdict"]["evidenced"], bool)
    assert "mean_partial_spearman" in result["summary"]


def test_evidence_benchmark_handles_tiny_data(tmp_path):
    """Tiny data must degrade gracefully to insufficient_data, not crash."""
    data = tmp_path / "tiny.csv"
    pd.DataFrame(
        {"user_id": [1, 1, 1, 2, 2, 2], "item_id": [10, 20, 10, 20, 10, 20],
         "correct": [1, 0, 1, 0, 1, 0], "difficulty": [0.3, 0.5, 0.3, 0.5, 0.3, 0.5],
         "skill_id": ["a", "a", "a", "a", "a", "a"], "timestamp": [0, 1, 2, 0, 1, 2]}
    ).to_csv(data, index=False)
    proc = subprocess.run(
        [sys.executable, "benchmarks/progression_reward_evidence.py",
         "--data", str(data), "--concept-col", "skill_id", "--epochs", "1",
         "--test-fraction", "0.5", "--max-seq-len", "4", "--batch-size", "4", "--seeds", "11"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["summary"].get("insufficient_data") or result["runs"][0].get("insufficient_data")


class TestProgressionRewardConfigValidation:
    def test_defaults_are_valid(self):
        ProgressionRewardConfig()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_correct": 0.0},
            {"target_correct": 1.0},
            {"target_correct": 1.5},
            {"stretch_width": 0.0},
            {"stretch_width": -0.1},
            {"mastery_gain_weight": -1.0},
            {"easy_penalty_weight": -0.01},
            {"repetition_window": 0},
            {"default_competence": 1.5},
            {"hard_correct_threshold": 0.9, "easy_correct_threshold": 0.5},
        ],
    )
    def test_invalid_configs_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ProgressionRewardConfig(**kwargs)
