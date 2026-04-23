# Orchid Ranker

[![PyPI version](https://img.shields.io/pypi/v/orchid-ranker.svg)](https://pypi.org/project/orchid-ranker/)
[![Python](https://img.shields.io/pypi/pyversions/orchid-ranker.svg)](https://pypi.org/project/orchid-ranker/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**The open-source recommender that optimizes for long-term user value, not short-term clicks.**

Orchid Ranker reframes every recommendation problem as a *progression problem* --- the user is getting better at something over time, and a recommender that models that trajectory outperforms one that only maximizes the next click.

## Quickstart

```bash
pip install orchid-ranker
```

```python
import pandas as pd
from orchid_ranker import OrchidRecommender

interactions = pd.read_csv("interactions.csv")  # user_id, item_id columns

# Fit, recommend, done.
rec = OrchidRecommender.from_interactions(interactions, strategy="als")
top5 = rec.recommend(user_id=7, top_k=5)
```

### Live adaptation

Use `neural_mf` when recommendations should change immediately after each
outcome:

```python
rec = OrchidRecommender.from_interactions(interactions, strategy="neural_mf")
streamer = rec.as_streaming(lr=0.05)

streamer.observe(user_id=7, item_id=42, correct=True, category="onboarding")
top5 = streamer.rank(user_id=7, candidate_item_ids=[1, 2, 3, 42, 99], top_k=5)
```

## Try it

Get up and running in under 5 minutes:

- [Quickstart example](examples/quickstart.py) --- full working script
- [Docs quickstart](docs/quickstart.md) --- install, fit, recommend, evaluate
- [Fit offline guide](docs/guides/01-fit-offline.md) --- train on a CSV, save, evaluate
- [Usage scenarios](docs/scenarios.md) --- practical recipes for common Orchid deployments

## Build with it

Go from batch to real-time:

- [Serve streaming](docs/guides/02-serve-streaming.md) --- wrap in `StreamingAdaptiveRanker`, hook up Kafka, rank live
- [Operate safely](docs/guides/03-operate-safely.md) --- add progression guardrails, Prometheus metrics, Grafana dashboards

## Evaluate it

Understand what makes Orchid different:

- [Why Orchid](docs/why-orchid.md) --- the long-term-value thesis, five pillars, honest comparison
- [Competitor comparison](docs/competitors.md) --- when to use Orchid vs RecBole, Merlin, implicit, LightFM, TFRS, Gorse
- [Benchmarks](docs/benchmarks/end-to-end.md) --- reproducible results on MovieLens-1M, music discovery, and learning platforms

---

## Five pillars

1. **Long-term-value-centric.** Optimizes for user growth and satisfaction, not engagement metrics.
2. **Adaptive.** Per-user, per-interaction online updates. Sub-10ms observe-to-rank on CPU.
3. **Streaming.** First-class Kafka ingestion. No batch-only workflows required.
4. **Safety-native.** Doubly-robust confidence sequences + progression-aware circuit breakers gate every rollout on user-outcome metrics.
5. **Privacy-native.** DP-SGD presets, RBAC, HMAC audit chains. GDPR/FERPA/EU AI Act alignment built in.

## Supported strategies

| Strategy | Type | Best for |
|----------|------|----------|
| `auto` | Selector | Let Orchid choose `als` or `explicit_mf` |
| `als` | Matrix factorization | Quick start, implicit feedback |
| `explicit_mf` | Matrix factorization | Explicit rating scales |
| `neural_mf` | Neural MF | Streaming adaptation through `as_streaming()` |
| `linucb` | Contextual bandit | Cold-start exploration |
| `user_knn` | Collaborative filtering | Small catalogs |
| `popularity` | Non-personalized | Baseline comparison |
| + 4 more | | See `OrchidRecommender.available_strategies()` |

For the high-level `OrchidRecommender` API, use `neural_mf` when you want to
promote a fitted model into `StreamingAdaptiveRanker` with `as_streaming()`.
`TwoTowerRecommender` remains available as a lower-level advanced model API, but
it is not a `strategy=` value.

## Status

![Tests](https://img.shields.io/badge/tests-1179%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

## License

Apache 2.0. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

## Citation

```bibtex
@software{orchid_ranker,
  title={Orchid Ranker: Progression-Aware Recommendation},
  author={Sam Urmian},
  year={2024},
  url={https://github.com/mlgorithm/orchid-ranker}
}
```
