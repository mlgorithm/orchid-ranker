# Performance Tuning Playbook

## 1. Microbenchmark inference

```bash
PYTHONPATH=src python benchmarks/bench_infer.py --iters 50 --users 128 --items 1024 --candidates 128 --dim 32 --native-score
```

Compare eager vs `--torch-compile` and native scoring to quantify gains.

## 2. Timing clamp

Use `--timing-log` + `--timing-rounds` on benchmarks to profile hot spots without full traces.

## 3. Common optimizations

- Reduce `min_candidates` when reranking costs dominate.
- Enable `--native-score` to keep matmuls on-device.
- Use `--torch-compile` on CUDA (PyTorch 2.x+) for tower models.
- Vectorize MMR and bandit bonuses (already implemented) and avoid `.cpu()` conversions in loops.

## 4. Interpreting timing JSONL

Example entry:

```json
{"round": 3, "phases": {"tower_infer": 35.2, "decide": 4.1}, "total": 48.0}
```

Focus on the largest phase and apply the optimizations above.

## 5. Regression guard

Add `bench_infer` run plus `ci_safe_smoke.sh` to CI so latency regressions are flagged automatically.
