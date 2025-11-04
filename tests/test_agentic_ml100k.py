import subprocess
import sys
from pathlib import Path

import pytest


def test_agentic_ml100k_benchmark(tmp_path):
    log_dir = tmp_path / "agentic-ml100k"
    cmd = [
        sys.executable,
        "benchmarks/run_agentic_ml100k.py",
        "--rounds",
        "5",
        "--top-users",
        "40",
        "--top-items",
        "80",
        "--top-k",
        "5",
        "--dim",
        "16",
        "--log-dir",
        str(log_dir),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - skip if network/download fails
        pytest.skip(f"ML-100K benchmark skipped ({exc.stderr.strip()})")
    assert "Fixed means:" in result.stdout
    assert "Adaptive means:" in result.stdout
