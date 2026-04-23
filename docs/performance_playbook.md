# Performance Tuning Playbook

## 1. Measure inference on a representative workload

Capture timing on the same request shapes you expect in production and compare
eager execution, `--torch-compile`, and native scoring where available.

## 2. Timing clamp

Use timing logs on smoke checks or validation runs to profile hot spots without
full traces.

## 3. Common optimizations

- Reduce `min_candidates` when reranking costs dominate.
- Enable `--native-score` to keep matmuls on-device.
- Use `--torch-compile` on CUDA (PyTorch 2.x+) for tower models; skip on MPS until supported.
- Vectorize MMR and bandit bonuses (already implemented) and avoid `.cpu()` conversions in loops.
- Reduce logging frequency (`log_flush_every`) during perf tests.

## 4. Interpreting timing JSONL

Example entry:

```json
{"round": 3, "phases": {"tower_infer": 35.2, "decide": 4.1}, "total": 48.0}
```

Focus on the largest phase and apply the optimizations above.

## 5. Regression guard

Add a lightweight serving smoke check to CI so latency regressions are flagged
automatically.
