import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if src.exists():
    sys.path.insert(0, str(src))
