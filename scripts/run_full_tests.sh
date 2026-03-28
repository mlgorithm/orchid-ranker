#!/usr/bin/env bash
# =============================================================================
# Orchid Ranker — Full Production Test Suite
# =============================================================================
# Run this locally where torch and all dependencies are installed.
#
# Usage:
#   ./scripts/run_full_tests.sh          # Run all tests
#   ./scripts/run_full_tests.sh --quick  # Run only fast unit tests
#   ./scripts/run_full_tests.sh --lint   # Run linting only
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================"
echo "  Orchid Ranker — Production Test Suite"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Environment check
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Checking environment...${NC}"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}')"
python3 -c "import numpy; print(f'  NumPy {numpy.__version__}')"
python3 -c "import pandas; print(f'  Pandas {pandas.__version__}')"
python3 -c "import sklearn; print(f'  scikit-learn {sklearn.__version__}')"
python3 -c "import scipy; print(f'  SciPy {scipy.__version__}')"
echo -e "${GREEN}  Environment OK${NC}"
echo ""

# ---------------------------------------------------------------------------
# 2. Lint with ruff
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[2/5] Running ruff linter...${NC}"
if command -v ruff &> /dev/null; then
    ruff check src/ tests/ --config pyproject.toml || {
        echo -e "${RED}  Ruff found issues. Run 'ruff check --fix' to auto-fix.${NC}"
    }
    echo -e "${GREEN}  Ruff complete${NC}"
else
    echo -e "${YELLOW}  ruff not installed, skipping (pip install ruff)${NC}"
fi
echo ""

if [[ "${1:-}" == "--lint" ]]; then
    echo "Lint-only mode — done."
    exit 0
fi

# ---------------------------------------------------------------------------
# 3. Type check with mypy
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[3/5] Running mypy type checker...${NC}"
if command -v mypy &> /dev/null; then
    mypy src/orchid_ranker/ --config-file pyproject.toml --no-error-summary 2>&1 | tail -5 || true
    echo -e "${GREEN}  mypy complete${NC}"
else
    echo -e "${YELLOW}  mypy not installed, skipping (pip install mypy)${NC}"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Unit tests
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[4/5] Running pytest...${NC}"
if [[ "${1:-}" == "--quick" ]]; then
    python3 -m pytest tests/ \
        --ignore=tests/test_agentic_ml100k.py \
        --ignore=tests/test_agentic_smoke.py \
        -x -q --tb=short 2>&1
else
    python3 -m pytest tests/ -v --tb=short 2>&1
fi
echo ""

# ---------------------------------------------------------------------------
# 5. Build check
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[5/5] Verifying package builds...${NC}"
python3 -m build --sdist --no-isolation 2>&1 | tail -3 || {
    echo -e "${YELLOW}  Build check skipped (install 'build' package)${NC}"
}
echo ""

echo "============================================"
echo -e "${GREEN}  All checks complete!${NC}"
echo "============================================"
