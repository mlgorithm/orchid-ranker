import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_agentic_smoke_script_runs(tmp_path):
    log_path = tmp_path / "smoke-log.jsonl"
    cmd = [
        sys.executable,
        "benchmarks/run_agentic_smoke.py",
        "--rounds",
        "1",
        "--users",
        "2",
        "--items",
        "6",
        "--log-path",
        str(log_path),
        "--seed",
        "7",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert log_path.exists(), f"expected log file at {log_path}"
    assert "Completed smoke run" in result.stdout
