#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${1:-runs/ci-safe-smoke}

./scripts/run_ml100k_safe_smoke.sh "$LOG_DIR"

ROUND_LOG="${LOG_DIR}/adaptive.jsonl"
if [[ ! -f "${ROUND_LOG}" ]]; then
  echo "Expected ${ROUND_LOG} to exist after safe smoke run" >&2
  exit 1
fi

if ! grep -q '"safe_gate":' "${ROUND_LOG}"; then
  echo "Safe gate telemetry missing in ${ROUND_LOG}" >&2
  exit 2
fi

echo "[ci_safe_smoke] Safe gate telemetry verified in ${ROUND_LOG}"
