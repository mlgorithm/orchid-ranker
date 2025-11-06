#!/usr/bin/env bash
set -euo pipefail

SCENARIOS=(
  "temporal_shift --temporal-split 0.9 --drift-round 45 --drift-type rotate --drift-magnitude 0.4 --drift-interval 0"
  "cold_users --cold-users 200"
  "item_coldstart --item-coldstart-frac 0.2"
  "session_drift --drift-round 25 --drift-interval 10 --drift-type rotate --drift-magnitude 0.45"
  "multi_objective"
  "trending --trend-start 40 --trend-window 15 --trend-boost 0.8"
  "constraints --constraint-latency 95 --constraint-complaints 0.008"
  "ood_shift --temporal-split 0.75"
)

rounds="${ROUNDS:-10}"
top_users="${TOP_USERS:-100}"
top_items="${TOP_ITEMS:-200}"
top_k="${TOP_K:-6}"
dim="${DIM:-14}"
seed="${SEED:-42}"

mkdir -p runs

for entry in "${SCENARIOS[@]}"; do
  name=$(echo "$entry" | awk '{print $1}')
  args=$(echo "$entry" | cut -d' ' -f2-)
  log_dir="runs/demo-${name}"
  echo "=== Running scenario: $name ==="
  PYTHONPATH=.:src python3 benchmarks/run_agentic_ml100k.py \
    --rounds "$rounds" \
    --top-users "$top_users" \
    --top-items "$top_items" \
    --top-k "$top_k" \
    --dim "$dim" \
    --seed "$seed" \
    --safe-eb --safe-eb-dr --safe-eb-conformal-alpha 0.1 --quick \
    --smoke --warmup-rounds 3 --warmup-steps 1 \
    --safe-eb-pstep 0.1 --safe-eb-pmin 0.1 --safe-eb-accept-floor 0.0 \
    --sim-agent movie \
    --log-dir "$log_dir" \
    --scenario "$name" \
    $args
done
