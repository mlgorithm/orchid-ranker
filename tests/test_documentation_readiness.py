"""Documentation and contributor-readiness checks."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "RELEASING.md",
    *sorted((ROOT / "docs").rglob("*.md")),
]


DELETED_PUBLIC_PATHS = [
    "SERIALIZATION_USAGE.md",
    "docs/application-scenarios.md",
    "docs/benchmarking_results.md",
    "docs/benchmarks/cold-start.md",
    "docs/benchmarks/curated-feed.md",
    "docs/benchmarks/end-to-end.md",
    "docs/benchmarks/movielens-1m.md",
    "docs/benchmarks/music.md",
    "docs/benchmarks/taste-progression.md",
    "docs/guides/migration-0.4-to-0.5.md",
    "docs/tutorial_safe_mode.md",
    "docs/tutorial_serialization.md",
    "docs/tutorials/library_walkthrough.md",
    "docs/tutorials/safe_mode.ipynb",
    "examples/kafka_integration.py",
    "examples/movielens_demo.py",
    "examples/quickstart.py",
]


LOCAL_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _markdown_links(path: Path) -> list[str]:
    links: list[str] = []
    for raw_target in LOCAL_LINK_RE.findall(path.read_text(encoding="utf-8")):
        target = raw_target.strip()
        if not target or target.startswith("#"):
            continue
        if "://" in target or target.startswith(("mailto:", "tel:")):
            continue
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        target = target.split("#", 1)[0].strip()
        if not target:
            continue
        links.append(target)
    return links


def test_local_markdown_links_resolve() -> None:
    broken: list[str] = []
    for path in DOC_PATHS:
        for target in _markdown_links(path):
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not broken


def test_public_docs_do_not_reference_deleted_generic_paths() -> None:
    offenders: list[str] = []
    for path in DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        for deleted_path in DELETED_PUBLIC_PATHS:
            if deleted_path in text:
                offenders.append(f"{path.relative_to(ROOT)} references {deleted_path}")
    assert not offenders


def test_coding_standards_are_discoverable() -> None:
    standards = ROOT / "docs" / "coding-standards.md"
    assert standards.exists()
    assert "coding-standards.md" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "coding-standards.md" in (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "coding-standards.md" in (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "Coding standards: coding-standards.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")


def test_use_case_examples_are_discoverable() -> None:
    cookbook = ROOT / "examples" / "adaptive_recommender_use_cases.py"
    docs = ROOT / "docs" / "examples.md"
    assert cookbook.exists()
    assert docs.exists()
    assert "adaptive_recommender_use_cases.py" in docs.read_text(encoding="utf-8")
    assert "Reference pilots: examples.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Compliance training" in docs.read_text(encoding="utf-8")


def test_logged_policy_validation_is_discoverable_and_claim_bounded() -> None:
    validation_script = ROOT / "benchmarks" / "validate_logged_policy.py"
    evidence_docs = ROOT / "docs" / "benchmarks" / "credibility.md"
    text = evidence_docs.read_text(encoding="utf-8")

    assert validation_script.exists()
    assert "validate_logged_policy.py" in text
    assert "JSON Lines" in text
    assert "controlled rollout" in text
    assert "causal benefit" in text
    assert "Validate a rollout: benchmarks/credibility.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")


def test_primary_docs_use_one_high_level_api() -> None:
    for relative_path in ("README.md", "docs/index.md", "docs/quickstart.md"):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "from orchid_ranker import AdaptiveRanker" in text
        assert "from orchid_ranker import AdaptiveLearningEngine" not in text
        assert "AdaptiveRanker().fit(" in text
        assert "fit_kt(" not in text
        assert '"user_id"' in text
        assert '"outcome"' in text
        assert '"timestamp"' in text


def test_primary_quickstart_is_runnable() -> None:
    result = subprocess.run(
        [sys.executable, "examples/adaptive_recommender_quickstart.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Adaptive recommender quickstart complete." in result.stdout


def test_top_level_examples_use_only_the_product_api() -> None:
    examples = sorted((ROOT / "examples").glob("*.py"))
    assert [path.name for path in examples] == [
        "adaptive_recommender_quickstart.py",
        "adaptive_recommender_use_cases.py",
        "production_serving.py",
    ]
    for path in examples:
        text = path.read_text(encoding="utf-8")
        assert "from orchid_ranker import AdaptiveRanker" in text
        assert "AdaptiveLearningEngine" not in text


def test_quality_script_runs_strict_standard_gates() -> None:
    script = (ROOT / "scripts" / "run_full_tests.sh").read_text(encoding="utf-8")
    for command in [
        "-m ruff check .",
        "-m mypy src/orchid_ranker",
        "-m pytest tests -q",
        "-m mkdocs build --strict",
        "-m build",
    ]:
        assert command in script
    assert "|| true" not in script
    assert "test_agentic_ml100k.py" not in script
