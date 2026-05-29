#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python}"
MODE="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_full_tests.sh          Run lint, types, tests, docs, and build
  ./scripts/run_full_tests.sh --quick  Run lint, types, docs readiness, and core smoke tests
  ./scripts/run_full_tests.sh --lint   Run lint and type checks only
EOF
}

if [[ "${MODE}" != "" && "${MODE}" != "--quick" && "${MODE}" != "--lint" ]]; then
  usage >&2
  exit 2
fi

run_step() {
  local label="$1"
  shift
  printf '\n==> %s\n' "${label}"
  "$@"
}

run_step "Ruff lint" "${PYTHON_BIN}" -m ruff check .
run_step "Mypy type check" "${PYTHON_BIN}" -m mypy src/orchid_ranker

if [[ "${MODE}" == "--lint" ]]; then
  exit 0
fi

if [[ "${MODE}" == "--quick" ]]; then
  run_step "Quick regression tests" "${PYTHON_BIN}" -m pytest \
    tests/test_documentation_readiness.py \
    tests/test_publish_readiness.py \
    tests/test_adaptive_learning_recommender.py \
    tests/test_knowledge_tracing.py \
    -q
else
  run_step "Full test suite with coverage" "${PYTHON_BIN}" -m pytest tests -q \
    --cov=src/orchid_ranker --cov-report=term-missing --cov-fail-under=70
fi

run_step "Documentation build" "${PYTHON_BIN}" -m mkdocs build --strict
run_step "Package build" "${PYTHON_BIN}" -m build
