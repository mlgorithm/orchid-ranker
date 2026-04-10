# Orchid Ranker: Capacity Planning Guide

**Target Audience**: DevOps engineers and platform teams deploying orchid-ranker in production.

---

## Inference Throughput Estimates

Single-core inference performance by strategy tier:

| Strategy Tier | Strategies | Throughput/Core | Batching Speedup |
|---|---|---|---|
| **Lightweight** | Popularity, Random | ~50K infer/sec | N/A |
| **Classical ML** | ALS, UserKNN, LinUCB | 5-10K infer/sec | 5-8x |
| **Neural (CPU)** | NeuralMF, TwoTower | 1-2K infer/sec | 10-20x |
| **Neural (GPU)** | NeuralMF, TwoTower | 10-20K infer/sec | 5-10x |

**Key insight**: `TwoTowerRecommender.infer_batch()` achieves 10-50x speedup over sequential inference. Always use batching for >100 req/sec workloads.

---

## Memory Sizing

| Component | Estimate |
|---|---|
| Base library (PyTorch, numpy, etc.) | ~200MB |
| Per 1M interactions (ALS matrices) | ~500MB |
| Per 1M interactions (neural models) | ~100MB |
| Per 10K users in MultiUserOrchestrator state | ~50MB |
| GPU VRAM (typical TwoTower model) | 2-4GB |

**Formula**: `Total RAM = 200MB + (interactions/1M × strategy_multiplier) + (users/10K × 50MB)`

Example: 10M interactions + 50K users with NeuralMF = `200 + (10×100) + (5×50) = 1,450MB ≈ 2GB`

---

## Training Resource Requirements

| Strategy | CPU-Only | GPU Recommended | Notes |
|---|---|---|---|
| ALS, ImplicitALS | ✓ | ≥100K interactions | Scales linearly; fine on CPU |
| BPR, ImplicitBPR | ✓ | ≥100K interactions | Pairwise sampling; CPU-friendly |
| NeuralMF, TwoTower | Slow | ≥100K interactions | GPU training 5-10x faster |
| DP-SGD (any) | +2-3x | +1-2x | Per-sample gradient clipping overhead |

**Recommendation**: For production retraining with >100K interactions, allocate 1 GPU. Cold-start bootstrap with <10K samples fine on 4 vCPU.

---

## GPU vs CPU Decision Matrix

| Workload | Hardware | Rationale |
|---|---|---|
| <100 req/sec, Popularity/Random | CPU (1 vCPU) | Lightweight; overhead of GPU unjustified |
| 100-1K req/sec, ALS/UserKNN | CPU (4-8 vCPU) | Classical ML scales well; batching sufficient |
| 1K+ req/sec, NeuralMF/TwoTower | GPU (1×A100/H100) | Neural inference dominates; GPU amortizes cost |
| Real-time feedback + retraining | GPU (1× per trainer) | Training overhead must not block inference |

---

## Kubernetes Sizing Recommendations

### Small Deployment (<1K users)

```
Replicas: 1
Requests: 2 vCPU, 4GB RAM
Limits: 4 vCPU, 8GB RAM
Strategy: Popularity or Random
HPA: Not needed
```

### Medium Deployment (1K-100K users)

```
Replicas: 2-4 (static or HPA)
Requests: 4 vCPU, 8GB RAM
Limits: 8 vCPU, 16GB RAM
Strategy: ALS, UserKNN, or LinUCB
HPA Triggers: CPU >70%, p99 latency >150ms
HPA Max: 10 replicas
```

### Large Deployment (100K+ users)

```
Replicas: 4-8 base
Requests: 8 vCPU, 16GB RAM (CPU) or 1 GPU + 8 vCPU, 32GB RAM (GPU)
Limits: 16 vCPU, 32GB RAM
Strategy: NeuralMF, TwoTower with DP-SGD
HPA: Target CPU 60%, p99 latency <100ms, request/rate >80%
GPU nodes: 1-4 A100s, horizontal pod autoscaling on queue depth
```

---

## Monitoring & Scaling Thresholds

### Scale Up (Add Replicas)

- **CPU utilization** > 70% sustained (5 min average)
- **p99 latency** > 100ms
- **Queue depth** > 50 pending requests

Action: Increase replicas by 50%, or enable HPA if not active.

### Add GPU

- **Inference** > 50% of request processing budget (profiled with `perf_counter()`)
- **p99 latency** > 200ms with batching enabled
- **Throughput** > 1K req/sec

Action: Migrate inference to GPU nodes; keep training on separate GPU.

### Differential Privacy Budget Alert

- **Remaining DP budget (ε)** < 20% of quarterly allocation
- **Example**: Allocated ε=8.0, remaining <1.6

Action: Pause DP-SGD training or reduce update frequency.

---

## Quick Sizing Checklist

- [ ] Estimate daily active users (DAU) and interactions/user/day
- [ ] Choose strategy tier based on latency SLA and model freshness requirements
- [ ] Calculate memory footprint using formula above
- [ ] Profile inference latency on target hardware (use `TwoTowerRecommender.infer_batch(batch_size=32)`)
- [ ] Reserve 30% headroom on memory and CPU
- [ ] If DP-SGD enabled, allocate 2x training resources
- [ ] Test HPA with synthetic load; validate max replica performance

---

## References

- See `MultiUserOrchestrator` for stateful user tracking and orchestration overhead
- See `TwoTowerRecommender` for batch inference API
- See differential privacy modules for privacy budget tracking
