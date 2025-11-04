#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-python3}
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

mkdir -p tmp/reports tmp/ml100k_explicit

# Prepare explicit split if missing
if [ ! -f tmp/ml100k_explicit/train.csv ]; then
  PYTHONPATH=src "$PY" - <<'PY'
from surprise import Dataset
import numpy as np, pandas as pd
from pathlib import Path
ml = Dataset.load_builtin('ml-100k', prompt=False)
raw = ml.raw_ratings
df = pd.DataFrame(raw, columns=['user_id','item_id','rating','timestamp']).astype({'user_id':int,'item_id':int,'rating':float})
msk = np.random.default_rng(123).random(len(df)) < 0.8
train, test = df[msk], df[~msk]
u, i = set(train.user_id), set(train.item_id)
test = test[test.user_id.isin(u) & test.item_id.isin(i)]
out = Path('tmp/ml100k_explicit'); out.mkdir(parents=True, exist_ok=True)
train[['user_id','item_id','rating']].to_csv(out/'train.csv', index=False)
test[['user_id','item_id','rating']].to_csv(out/'test.csv', index=False)
print('Prepared explicit CSVs at', out)
PY
fi

echo "[Explicit] Orchid explicit_mf vs Surprise SVD"
PYTHONPATH=src "$PY" benchmarks/compare_surprise.py \
  --train tmp/ml100k_explicit/train.csv \
  --test tmp/ml100k_explicit/test.csv \
  --rating-col rating \
  --orchid-strategy explicit_mf \
  --orchid-epochs 20 \
  --orchid-emb 64 \
  --output tmp/reports/explicit_results.json

echo "[Implicit] Multi-seed apples-to-apples ranking"
PYTHONPATH=src "$PY" benchmarks/eval_implicit.py --seeds 11 13 17 --top-users 400 --top-items 800 --k 10

echo "Artifacts in tmp/reports:"
ls -l tmp/reports

