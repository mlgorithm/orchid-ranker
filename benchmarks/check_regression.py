"""Check benchmark results against golden baselines for regressions.

Compares current run metrics against saved golden baselines.
Fails if any strategy's metric drops more than the allowed threshold.

Usage:
    python benchmarks/bench_strategies.py --save-golden   # first time
    python benchmarks/check_regression.py                  # in CI
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "baselines.json"

# Allowed regression thresholds (fraction, e.g., 0.15 = 15% drop allowed)
METRIC_THRESHOLDS = {
    "precision_at_k_mean": 0.15,
    "ndcg_at_k_mean": 0.15,
    "recall_at_k_mean": 0.15,
}

# Performance thresholds (allowed slowdown factor)
PERF_THRESHOLDS = {
    "fit_time_mean_sec": 2.0,    # allow 2x slower (CI machines vary)
    "infer_p95_mean_ms": 3.0,    # allow 3x slower (no GPU in CI)
}


def main():
    if not GOLDEN_PATH.exists():
        print(f"No golden baselines found at {GOLDEN_PATH}")
        print("Run: python benchmarks/bench_strategies.py --save-golden")
        sys.exit(1)

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    golden_agg = golden.get("aggregated", {})
    if not golden_agg:
        print("Golden baselines file has no aggregated results")
        sys.exit(1)

    # Run current benchmarks
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from bench_strategies import main as run_benchmarks
    current = run_benchmarks()
    current_agg = current.get("aggregated", {})

    failures = []

    for strategy, golden_metrics in golden_agg.items():
        if strategy not in current_agg:
            failures.append(f"  {strategy}: MISSING from current run")
            continue

        current_metrics = current_agg[strategy]

        # Check quality metrics
        for metric, threshold in METRIC_THRESHOLDS.items():
            g_val = golden_metrics.get(metric, 0.0)
            c_val = current_metrics.get(metric, 0.0)
            if g_val > 0 and (g_val - c_val) / g_val > threshold:
                failures.append(
                    f"  {strategy}.{metric}: {g_val:.4f} → {c_val:.4f} "
                    f"({(g_val - c_val) / g_val * 100:.1f}% regression, "
                    f"threshold={threshold * 100:.0f}%)"
                )

        # Check performance (loose thresholds for CI)
        for metric, factor in PERF_THRESHOLDS.items():
            g_val = golden_metrics.get(metric, 0.0)
            c_val = current_metrics.get(metric, 0.0)
            if g_val > 0 and c_val > g_val * factor:
                failures.append(
                    f"  {strategy}.{metric}: {g_val} → {c_val} "
                    f"({c_val / g_val:.1f}x slower, threshold={factor}x)"
                )

    if failures:
        print("\n❌ REGRESSION DETECTED:")
        for f in failures:
            print(f)
        sys.exit(1)
    else:
        print("\n✅ All strategies within regression thresholds")
        sys.exit(0)


if __name__ == "__main__":
    main()
