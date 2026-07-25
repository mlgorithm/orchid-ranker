"""Executable checks for documented use-case examples."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "adaptive_recommender_use_cases.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("adaptive_recommender_use_cases", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_use_case_cookbook_runs_and_returns_expected_domains() -> None:
    cookbook = _load_example_module()

    results = cookbook.run_all()

    assert set(results) == {"onboarding", "compliance_training", "content_discovery"}
    for result in results.values():
        assert result["completed_decisions"] == 1
        assert result["outcome"]
        assert result["guardrail"]
        assert result["served_item"] in result["eligible_items"]
        assert result["policy_version"].endswith("-v1")


def test_reference_pilots_export_completed_json_lines(tmp_path: Path) -> None:
    cookbook = _load_example_module()
    runs = cookbook.run_pilots()

    cookbook.export_completed_logs(runs, tmp_path)

    exported = sorted(tmp_path.glob("*-completed-decisions.jsonl"))
    assert [path.name for path in exported] == [
        "compliance_training-completed-decisions.jsonl",
        "content_discovery-completed-decisions.jsonl",
        "onboarding-completed-decisions.jsonl",
    ]
    for path in exported:
        frame = pd.read_json(path, lines=True)
        assert len(frame) == 1
        assert {"decision_id", "candidate_item_ids", "propensity", "reward"}.issubset(frame.columns)
