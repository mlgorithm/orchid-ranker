"""The supported empirical workflow must run in a PyTorch-free process."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_base_import_and_empirical_serving_without_torch() -> None:
    code = textwrap.dedent(
        """
        import builtins
        import sys

        import pandas as pd
        from orchid_ranker import AdaptiveRanker
        import orchid_ranker.adaptive_learning as learning
        from orchid_ranker.pilot import AdaptivePracticePilot, PilotCatalog

        assert "torch" not in sys.modules
        original_import = builtins.__import__
        def reject_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("No module named 'torch'")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = reject_torch
        original_find_spec = learning.find_spec
        learning.find_spec = lambda name: None if name == "torch" else original_find_spec(name)

        events = pd.DataFrame({
            "user_id": ["a", "a", "b", "b", "c", "c"],
            "item_id": [1, 2, 1, 2, 1, 2],
            "outcome": [1, 0, 0, 1, 1, 0],
            "timestamp": [1, 2, 1, 2, 1, 2],
        })
        ranker = AdaptiveRanker().fit(events)
        assert ranker.learning_readiness()["active_tracer"] == "empirical"
        assert ranker.recommend("a", [1, 2])
        assert "torch" not in sys.modules

        supported = dict(min_kt_events=1, min_kt_users=1, min_kt_items=1,
                         min_kt_median_events_per_user=1)
        fallback = AdaptiveRanker(kt_backbone="sakt", **supported).fit(events)
        readiness = fallback.learning_readiness()
        assert readiness["knowledge_tracing_ready"]
        assert readiness["active_tracer"] == "empirical"
        assert any("PyTorch is not installed" in reason for reason in readiness["reasons"])
        assert any("orchid-ranker[kt]" in item for item in readiness["recommendations"])

        try:
            AdaptiveRanker(kt_backbone="sakt", fallback_to_empirical=False, **supported).fit(events)
        except ImportError as error:
            assert "orchid-ranker[kt]" in str(error)
        else:
            raise AssertionError("explicit neural tracing should explain its extra")
        """
    )
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
