from __future__ import annotations

# ---------------------------------------------------------------------------
# Opt-in debug logging for imports in this module.
# Enable via:
#   - env var: ORCHID_DEBUG_IMPORTS=1 python your_script.py
#   - or at runtime: from this_module import enable_debug_import_logs; enable_debug_import_logs(True)
# ---------------------------------------------------------------------------
import os
import sys
import traceback

_DEBUG_IMPORTS = os.getenv("ORCHID_DEBUG_IMPORTS", "").lower() in {"1", "true", "yes", "on"}

def enable_debug_import_logs(flag: bool = True) -> None:
    """Enable/disable import debug messages at runtime."""
    global _DEBUG_IMPORTS
    _DEBUG_IMPORTS = bool(flag)

def _p(*args) -> None:
    if _DEBUG_IMPORTS:
        print("[orchid_ranker.imports]", *args)

def _ok(name: str, obj) -> None:
    try:
        mod = getattr(obj, "__module__", "?")
        path = None
        if mod in sys.modules:
            m = sys.modules[mod]
            path = getattr(m, "__file__", None)
        _p(f"loaded {name} from {mod}" + (f" ({path})" if path else ""))
    except Exception:
        _p(f"loaded {name} (module path unknown)")

def _fail(ctx: str, err: Exception) -> None:
    _p(f"FAILED to import {ctx}: {err.__class__.__name__}: {err}")
    if _DEBUG_IMPORTS:
        traceback.print_exc()

# ---------------------------------------------------------------------------
# Visible imports with debug
# ---------------------------------------------------------------------------
try:
    from orchid_ranker.agents.recommender_agent import (
        BootTS,
        DualRecommender,
        JSONLLogger,
        LinUCBPolicy,
        TwoTowerRecommender,
    )
    _ok("BootTS", BootTS)
    _ok("DualRecommender", DualRecommender)
    _ok("JSONLLogger", JSONLLogger)
    _ok("LinUCBPolicy", LinUCBPolicy)
    _ok("TwoTowerRecommender", TwoTowerRecommender)
except Exception as e:
    _fail("orchid_ranker.agents.recommender_agent", e)
    raise

try:
    from orchid_ranker.agents.student_agent import (
        ItemMeta,
        StudentAgent,
        StudentAgentFactory,
    )
    _ok("ItemMeta", ItemMeta)
    _ok("StudentAgent", StudentAgent)
    _ok("StudentAgentFactory", StudentAgentFactory)
except Exception as e:
    _fail("orchid_ranker.agents.student_agent", e)
    raise

__all__ = [
    "JSONLLogger",
    "ItemMeta",
    "StudentAgent",
    "StudentAgentFactory",
    "LinUCBPolicy",
    "BootTS",
    "TwoTowerRecommender",
    "DualRecommender",
    "enable_debug_import_logs",
]
