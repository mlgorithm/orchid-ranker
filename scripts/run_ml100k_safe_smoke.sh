#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${1:-runs/ml100k-safe-smoke}

PYTHONPATH=src python3 benchmarks/run_agentic_ml100k.py \
  --rounds 5 \
  --top-users 60 \
  --top-items 120 \
  --top-k 6 \
  --dim 10 \
  --quick --smoke \
  --safe-eb --safe-eb-dr \
  --skip-fixed \
  --timing-rounds 3 \
  --timing-log "${LOG_DIR}/timing.jsonl" \
  --log-dir "$LOG_DIR"
