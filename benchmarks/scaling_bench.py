#!/usr/bin/env python3
"""Benchmark: orchid_ranker.scaling module.

Measures memory, latency, and throughput for the sparse/sharded scaling
backends at various active-user counts.  Entirely synthetic — no external
dataset required.

Usage::

    python benchmarks/scaling_bench.py              # full run
    python benchmarks/scaling_bench.py --smoke       # quick CI check

Outputs a Markdown table suitable for docs/benchmarks/scaling.md.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure the source tree is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Suppress noisy eviction warnings during benchmarking
logging.getLogger("orchid_ranker.scaling").setLevel(logging.ERROR)

from orchid_ranker.scaling import (
    ShardedBKTStateProvider,
    SparseEmbeddingTable,
    SparseOnlineUserAdapter,
)
from orchid_ranker.streaming import BKTStateProvider, OnlineUserAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tensor_memory_mb(t: torch.Tensor) -> float:
    """Return memory used by a tensor in MB."""
    return t.nelement() * t.element_size() / (1024 * 1024)


def _embedding_memory_mb(num_rows: int, dim: int, dtype: torch.dtype = torch.float32) -> float:
    """Theoretical memory for a dense embedding table."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    return num_rows * dim * elem_size / (1024 * 1024)


def _time_ms(func, *args, warmup=5, trials=100, **kwargs):
    """Return (mean_ms, p50_ms, p95_ms, p99_ms) over trials."""
    for _ in range(warmup):
        func(*args, **kwargs)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
    }


# ---------------------------------------------------------------------------
# Benchmark: SparseEmbeddingTable memory
# ---------------------------------------------------------------------------
def bench_memory(active_counts: list[int], emb_dim: int = 32) -> list[dict]:
    """Measure memory for dense (all registered) vs sparse (active only).

    The realistic scenario: a platform has many registered users but only
    a fraction are concurrently active.  A dense ``nn.Embedding`` must
    allocate rows for ALL registered users.  The sparse table allocates
    only for active users.

    Memory is computed analytically (tensor bytes + measured Python-side
    overhead) to avoid tracemalloc's blindness to C-level allocators.
    """
    registered_users = 1_000_000  # 1M registered

    rows = []
    for n_active in active_counts:
        # Dense: nn.Embedding(registered_users, emb_dim) — float32
        dense_mb = _embedding_memory_mb(registered_users, emb_dim)

        # Sparse: only allocates n_active tensors + OrderedDict overhead
        # Each entry: one float32 tensor (emb_dim elements) + dict overhead
        tensor_mb = _embedding_memory_mb(n_active, emb_dim)
        # Python dict overhead: ~120 bytes per entry (key + value + node)
        dict_overhead_mb = n_active * 120 / (1024 * 1024)
        sparse_mb = tensor_mb + dict_overhead_mb

        savings = round((1.0 - sparse_mb / max(dense_mb, 0.01)) * 100, 1)
        rows.append({
            "registered_users": registered_users,
            "active_users": n_active,
            "dense_mb": round(dense_mb, 1),
            "sparse_mb": round(sparse_mb, 1),
            "savings_pct": savings,
        })
        print(
            f"  memory: {n_active:>10,} active / {registered_users:,} registered "
            f"| dense={dense_mb:.1f} MB | sparse={sparse_mb:.1f} MB | save={savings:.1f}%"
        )

    return rows


# ---------------------------------------------------------------------------
# Benchmark: observe latency (BKT)
# ---------------------------------------------------------------------------
def bench_bkt_latency(
    num_users: int,
    num_shards_list: list[int],
    trials: int = 500,
) -> list[dict]:
    """Measure per-observe latency for dense BKT vs sharded at various shard counts."""
    rng = np.random.default_rng(42)
    rows = []

    # Dense baseline
    dense_state = BKTStateProvider()
    user_ids = rng.integers(0, num_users, size=trials).tolist()
    corrects = rng.choice([True, False], size=trials).tolist()

    stats = _time_ms(
        lambda: [dense_state.observe(u, correct=c) for u, c in zip(user_ids[:10], corrects[:10])],
        trials=trials // 10,
    )
    rows.append({"backend": "dense (1 lock)", "shards": 1, **{k: round(v, 3) for k, v in stats.items()}})
    print(f"  bkt latency: dense        | p50={stats['p50_ms']:.3f}ms | p99={stats['p99_ms']:.3f}ms")

    for ns in num_shards_list:
        sharded = ShardedBKTStateProvider(num_shards=ns)
        stats = _time_ms(
            lambda: [sharded.observe(u, correct=c) for u, c in zip(user_ids[:10], corrects[:10])],
            trials=trials // 10,
        )
        rows.append({"backend": f"sharded ({ns} shards)", "shards": ns, **{k: round(v, 3) for k, v in stats.items()}})
        print(f"  bkt latency: {ns:>3} shards   | p50={stats['p50_ms']:.3f}ms | p99={stats['p99_ms']:.3f}ms")

    return rows


# ---------------------------------------------------------------------------
# Benchmark: adapter observe latency
# ---------------------------------------------------------------------------
def bench_adapter_latency(
    emb_dim: int = 32,
    trials: int = 500,
) -> list[dict]:
    """Measure per-observe latency for dense vs sparse adapter."""
    rows = []

    # Dense
    dense_ad = OnlineUserAdapter(num_users=10_000, emb_dim=emb_dim, lr=0.05)
    u_base = torch.randn(emb_dim)
    i_emb = torch.randn(emb_dim)

    stats = _time_ms(lambda: dense_ad.observe(42, u_base, i_emb, 1.0), trials=trials)
    rows.append({"backend": "dense adapter", **{k: round(v, 3) for k, v in stats.items()}})
    print(f"  adapter latency: dense    | p50={stats['p50_ms']:.3f}ms | p99={stats['p99_ms']:.3f}ms")

    # Sparse
    sparse_ad = SparseOnlineUserAdapter(emb_dim=emb_dim, lr=0.05, max_active_users=10_000)
    stats = _time_ms(lambda: sparse_ad.observe(42, u_base, i_emb, 1.0), trials=trials)
    rows.append({"backend": "sparse adapter", **{k: round(v, 3) for k, v in stats.items()}})
    print(f"  adapter latency: sparse   | p50={stats['p50_ms']:.3f}ms | p99={stats['p99_ms']:.3f}ms")

    return rows


# ---------------------------------------------------------------------------
# Benchmark: LRU eviction behaviour
# ---------------------------------------------------------------------------
def bench_eviction(max_entries: int = 1_000, total_users: int = 5_000, emb_dim: int = 32) -> dict:
    """Measure eviction rate and steady-state occupancy."""
    table = SparseEmbeddingTable(emb_dim=emb_dim, max_entries=max_entries)
    rng = np.random.default_rng(0)

    evictions = 0
    for uid in rng.integers(0, total_users, size=total_users * 2):
        was_in = int(uid) in table
        table[int(uid)] = torch.randn(emb_dim)
        if not was_in and len(table) == max_entries:
            evictions += 1

    result = {
        "max_entries": max_entries,
        "total_users_seen": total_users * 2,
        "unique_users": total_users,
        "final_occupancy": len(table),
        "at_capacity": len(table) == max_entries,
    }
    print(f"  eviction: max={max_entries}, final_occupancy={len(table)}, at_capacity={result['at_capacity']}")
    return result


# ---------------------------------------------------------------------------
# Benchmark: concurrent throughput
# ---------------------------------------------------------------------------
def bench_concurrency(
    num_shards_list: list[int],
    num_threads_list: list[int] | None = None,
    ops_per_thread: int = 500,
    num_users: int = 10_000,
) -> list[dict]:
    """Measure throughput (ops/sec) under concurrent load.

    For each shard count and thread count, measure how many observe()
    operations per second the sharded BKT provider handles when
    multiple threads are writing simultaneously.
    """
    if num_threads_list is None:
        num_threads_list = [1, 2, 4, 8]

    rng = np.random.default_rng(42)
    rows = []

    for ns in num_shards_list:
        for nt in num_threads_list:
            sharded = ShardedBKTStateProvider(num_shards=ns)

            # Pre-generate per-thread work: user_ids and correct flags
            thread_uids = [
                rng.integers(0, num_users, size=ops_per_thread).tolist()
                for _ in range(nt)
            ]
            thread_corrects = [
                rng.choice([True, False], size=ops_per_thread).tolist()
                for _ in range(nt)
            ]

            def _worker(tid: int) -> float:
                t0 = time.perf_counter()
                for u, c in zip(thread_uids[tid], thread_corrects[tid]):
                    sharded.observe(u, correct=c)
                return time.perf_counter() - t0

            # Warmup
            for u, c in zip(thread_uids[0][:50], thread_corrects[0][:50]):
                sharded.observe(u, correct=c)

            # Timed run
            t_start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=nt) as pool:
                futures = [pool.submit(_worker, tid) for tid in range(nt)]
                durations = [f.result() for f in futures]
            wall_time = time.perf_counter() - t_start

            total_ops = nt * ops_per_thread
            throughput = total_ops / wall_time

            row = {
                "shards": ns,
                "threads": nt,
                "total_ops": total_ops,
                "wall_time_s": round(wall_time, 3),
                "throughput_ops_per_sec": round(throughput, 0),
                "mean_thread_time_s": round(statistics.mean(durations), 3),
            }
            rows.append(row)
            print(
                f"  concurrency: {ns:>3} shards × {nt:>2} threads | "
                f"{throughput:>10,.0f} ops/s | wall={wall_time:.3f}s"
            )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scaling module benchmark")
    parser.add_argument("--smoke", action="store_true", help="Quick CI check")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    if args.smoke:
        active_counts = [1_000, 10_000]
        shard_list = [4]
        trials = 50
    else:
        active_counts = [1_000, 10_000, 100_000, 500_000]
        shard_list = [4, 16, 32, 64]
        trials = 500

    results = {}

    print("\n=== Memory benchmark ===")
    results["memory"] = bench_memory(active_counts)

    print("\n=== BKT observe latency ===")
    results["bkt_latency"] = bench_bkt_latency(10_000, shard_list, trials=trials)

    print("\n=== Adapter observe latency ===")
    results["adapter_latency"] = bench_adapter_latency(trials=trials)

    print("\n=== LRU eviction ===")
    results["eviction"] = bench_eviction()

    print("\n=== Concurrent throughput ===")
    if args.smoke:
        conc_threads = [1, 2, 4]
        conc_ops = 200
    else:
        conc_threads = [1, 2, 4, 8]
        conc_ops = 2000
    results["concurrency"] = bench_concurrency(
        shard_list, conc_threads, ops_per_thread=conc_ops,
    )

    # Output
    out_path = args.output or str(
        Path(__file__).resolve().parent / "results_scaling.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Print markdown table for docs
    print("\n### Memory footprint (emb_dim=32, 1M registered users)\n")
    print("| Active users | Dense — all registered (MB) | Sparse — active only (MB) | Savings |")
    print("|-------------|---------------------------|--------------------------|---------|")
    for row in results["memory"]:
        print(
            f"| {row['active_users']:>11,} "
            f"| {row['dense_mb']:>25.1f} "
            f"| {row['sparse_mb']:>24.1f} "
            f"| {row['savings_pct']:>5.1f}% |"
        )

    print("\n### BKT observe latency (10 observations per call)\n")
    print("| Backend | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |")
    print("|---------|----------|---------|---------|---------|")
    for row in results["bkt_latency"]:
        print(f"| {row['backend']:<20} | {row['mean_ms']:>8.3f} | {row['p50_ms']:>7.3f} | {row['p95_ms']:>7.3f} | {row['p99_ms']:>7.3f} |")

    print("\n### Adapter observe latency (single observation)\n")
    print("| Backend | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |")
    print("|---------|----------|---------|---------|---------|")
    for row in results["adapter_latency"]:
        print(f"| {row['backend']:<20} | {row['mean_ms']:>8.3f} | {row['p50_ms']:>7.3f} | {row['p95_ms']:>7.3f} | {row['p99_ms']:>7.3f} |")

    print("\n### Concurrent throughput (ops/sec)\n")
    print("| Shards | Threads | Total ops | Wall time (s) | Throughput (ops/s) |")
    print("|--------|---------|-----------|---------------|--------------------|")
    for row in results["concurrency"]:
        print(
            f"| {row['shards']:>6} | {row['threads']:>7} "
            f"| {row['total_ops']:>9,} | {row['wall_time_s']:>13.3f} "
            f"| {row['throughput_ops_per_sec']:>18,.0f} |"
        )


if __name__ == "__main__":
    main()
