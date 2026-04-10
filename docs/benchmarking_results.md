# Benchmark Results

Orchid Ranker vs. popular recommender libraries on the frozen fixture dataset 
(200 users, 300 items, 8000 interactions, seed=42).

## Methodology

- **Dataset**: Synthetic educational interactions with latent skill alignment
- **Split**: 80/20 train/test, deterministic (seed=42)
- **Metrics**: Precision@10, Recall@10, NDCG@10 (quality); fit time, inference p50/p95 (performance)
- **Seeds**: 3-seed averaging (42, 13, 17) for variance estimation
- **Hardware**: [to be filled when benchmark runs]

## Results

> **Note**: Run `python benchmarks/bench_competitors.py` to generate actual numbers.
> The table below will be updated with real results after the first benchmark run.

| Library | Algorithm | P@10 | R@10 | NDCG@10 | Fit (s) | Infer p50 (ms) | Memory (MB) |
|---------|-----------|------|------|---------|---------|----------------|-------------|
| Orchid Ranker | ALS | — | — | — | — | — | — |
| Orchid Ranker | NeuralMF | — | — | — | — | — | — |
| Orchid Ranker | Popularity | — | — | — | — | — | — |
| Orchid Ranker | UserKNN | — | — | — | — | — | — |
| Surprise | SVD | — | — | — | — | — | — |
| Surprise | NMF | — | — | — | — | — | — |
| Surprise | KNNBasic | — | — | — | — | — | — |
| implicit | ALS | — | — | — | — | — | — |
| implicit | BPR | — | — | — | — | — | — |
| LightFM | WARP | — | — | — | — | — | — |
| LightFM | BPR | — | — | — | — | — | — |

## Key Takeaways

*(To be filled after benchmark run)*

## Orchid Ranker's Unique Advantages

While quality metrics tell part of the story, Orchid Ranker differentiates on capabilities no competitor offers:

1. **Education-native**: BKT knowledge tracing, ZPD-aware recommendations, curriculum sequencing, and mastery tracking built in — not bolted on.

2. **Differential privacy**: Production-ready DP-SGD with RDP accounting and configurable epsilon budgets. No other recommender library ships this.

3. **Learner simulation**: Full agentic framework (StudentAgent + MultiUserOrchestrator) for offline evaluation of recommendation policies before deployment.

4. **Enterprise infrastructure**: RBAC, JWT/OIDC auth, HMAC audit logs, secrets management, Prometheus metrics, OpenTelemetry — all in one pip install.

5. **Familiar API**: Surprise-compatible `fit/predict/recommend` interface means zero learning curve for teams already using Surprise.

## How to Reproduce

[instructions to run bench_competitors.py and update this doc]

## Updating This Document

After running the benchmark:
1. Run: `python benchmarks/bench_competitors.py --output benchmarks/golden/competitors.json`
2. The script prints a markdown table — paste it above
3. Commit the updated results
