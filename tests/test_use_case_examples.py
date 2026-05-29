"""Executable checks for documented use-case examples."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "adaptive_learning_use_cases.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("adaptive_learning_use_cases", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_use_case_cookbook_runs_and_returns_expected_domains() -> None:
    cookbook = _load_example_module()

    results = cookbook.run_all()

    assert set(results) == {
        "compliance_training",
        "language_review",
        "rehab_progression",
        "onboarding_rollout_gate",
    }
    assert results["compliance_training"]["next_modules"] == ["data-handling", "phishing-response"]
    assert results["language_review"]["review_order"][0]["item_id"] == "se souvenir"
    assert results["rehab_progression"]["recommended_exercise"] == "supported-step-up"
    assert results["onboarding_rollout_gate"]["allowed"] is True
