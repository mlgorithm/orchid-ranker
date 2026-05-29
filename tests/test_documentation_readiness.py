"""Documentation and contributor-readiness checks."""
from __future__ import annotations

import re
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
    cookbook = ROOT / "examples" / "adaptive_learning_use_cases.py"
    docs = ROOT / "docs" / "examples.md"
    assert cookbook.exists()
    assert docs.exists()
    assert "adaptive_learning_use_cases.py" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Use-case examples: examples.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Compliance Training" in docs.read_text(encoding="utf-8")


def test_quality_script_runs_strict_standard_gates() -> None:
    script = (ROOT / "scripts" / "run_full_tests.sh").read_text(encoding="utf-8")
    for command in [
        "-m ruff check .",
        "-m mypy src/orchid_ranker",
        "-m pytest tests -q",
        "--cov-fail-under=70",
        "-m mkdocs build --strict",
        "-m build",
    ]:
        assert command in script
    assert "|| true" not in script
    assert "test_agentic_ml100k.py" not in script
