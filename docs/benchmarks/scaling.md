# Scaling benchmark

> **Status:** auto-generated from `benchmarks/scaling_bench.py`.
> Numbers are from a synthetic single-machine run.
> Real production numbers depend on hardware, concurrency, and workload.

## What this measures

The `orchid_ranker.scaling` module replaces dense data structures
(that allocate memory for ALL registered users) with sparse/sharded
alternatives that allocate only for **active** users.  This benchmark
quantifies the tradeoff across four dimensions:

1. **Memory footprint** -- dense vs. sparse embedding tables
2. **Per-observation latency** -- dense BKT vs. sharded BKT
3. **Adapter latency** -- dense vs. sparse online user adapter
4. **Concurrent throughput** -- ops/sec under multi-threaded load
5. **LRU eviction** -- behaviour when the active set exceeds capacity

## Scenario

A platform with **1,000,000 registered users** but only a fraction
concurrently active.  A dense `nn.Embedding(1M, 32)` allocates
122 MB regardless of activity.  The sparse table allocates only
for active users.

## Results

### Memory footprint (emb_dim=32, 1M registered)

| Active users | Dense (MB) | Sparse (MB) | Savings |
|-------------|-----------|------------|---------|
|       1,000 |     122.1 |        0.2 |  99.8%  |
|      10,000 |     122.1 |        2.4 |  98.1%  |
|     100,000 |     122.1 |       23.7 |  80.6%  |
|     500,000 |     122.1 |      118.3 |   3.1%  |

At typical active-to-registered ratios (1--10%), sparse tables
save 80--99% of memory.  When the active set approaches the total
registration count, the per-entry dict overhead erodes savings.

### BKT observe latency (10 observations per batch)

| Backend | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|----------|---------|---------|---------|
| Dense (1 lock)      | 0.005 | 0.005 | 0.005 | 0.005 |
| Sharded (4 shards)  | 0.006 | 0.006 | 0.006 | 0.006 |

Sharding adds negligible overhead in the single-threaded case.
The value of sharding is realised under concurrent access (see below).

### Adapter observe latency (single observation)

| Backend | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|----------|---------|---------|---------|
| Dense adapter  | 0.014 | 0.014 | 0.016 | 0.018 |
| Sparse adapter | 0.011 | 0.011 | 0.011 | 0.012 |

The sparse adapter is slightly faster at p50 because `OrderedDict`
lookups are faster than indexing a large contiguous tensor when the
active set is small.

### Concurrent throughput

| Shards | Threads | Throughput (ops/s) |
|--------|---------|-------------------|
| 4      | 1       | ~510,000 |
| 4      | 8       | ~838,000 |
| 16     | 1       | ~723,000 |
| 16     | 8       | ~843,000 |
| 32     | 8       | ~841,000 |

More shards reduce lock contention.  At 16+ shards with 8 threads,
throughput reaches **840K+ ops/sec**.

!!! note "GIL limitation"
    Python's GIL limits true parallelism for CPU-bound work.  The
    sharded backend still provides concurrent-safe access, and
    throughput remains high (700K+ ops/s even under contention).
    In production, the GIL overhead is amortised by I/O-bound work
    between observations.

### LRU eviction

| Parameter | Value |
|-----------|-------|
| max_entries | 1,000 |
| total_users_seen | 10,000 |
| unique_users | 5,000 |
| final_occupancy | 1,000 |
| at_capacity | Yes |

The table correctly maintains capacity by evicting least-recently-used
entries.  After processing 10,000 interactions from 5,000 unique users,
the table holds exactly 1,000 entries (the most recently active).

## Caveats

!!! warning "LRU eviction"
    When the active set exceeds `max_active_users`, the oldest entries
    are evicted.  Evicted users revert to cold-start embeddings on
    next access.  Size the `max_active_users` parameter to your
    **concurrent** active users, not your total registration count.

!!! warning "Active-user qualification"
    "Active" means a user who has been observed within the LRU window.
    A user who registered but never interacted occupies zero memory.
    A user who interacted once and then went dormant is eventually
    evicted when the table fills.

## Reproducibility

```bash
# Full run (< 5 min on an M-class Mac)
python benchmarks/scaling_bench.py

# Quick smoke test (< 30 sec)
python benchmarks/scaling_bench.py --smoke
```

Results are written to `benchmarks/results_scaling.json`.
