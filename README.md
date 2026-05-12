# Orchid Ranker

[![PyPI version](https://img.shields.io/pypi/v/orchid-ranker.svg)](https://pypi.org/project/orchid-ranker/)
[![Python](https://img.shields.io/pypi/pyversions/orchid-ranker.svg)](https://pypi.org/project/orchid-ranker/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**A first-class adaptive-learning engine for products where user outcomes matter more than short-term clicks.**

Orchid Ranker is built for systems where the user is getting better at
something over time: adaptive learning, corporate training, tutoring,
onboarding, rehabilitation, fitness progression, and skill-based practice. It
combines learner-state tracking, prerequisite-aware candidate selection,
live learner-state updates, progression metrics, and safe fallback patterns.

It is not a generic CTR, ads, social-feed, or movie-recommendation model zoo.
The core question is: **what should this learner work on next so they make
measurable progress?**

## Quickstart

```bash
pip install 'orchid-ranker[ml]'
```

```python
import pandas as pd
from orchid_ranker import AdaptiveLearningEngine

outcomes = pd.read_csv("learner_outcomes.csv")  # user_id, item_id, correct
catalog = pd.read_csv("exercise_catalog.csv")   # item_id, concept, difficulty

learner_rec = AdaptiveLearningEngine(
    tracer_model="akt",
    policy="auto",
    epochs=2,
    d_model=32,
).fit(
    outcomes.merge(catalog, on="item_id"),
    correct_col="correct",
    concept_col="concept",
    item_difficulty_col="difficulty",
    prerequisite_by_concept={"fractions": ["number-sense"]},
)

ranked = learner_rec.rank(user_id=7, candidate_item_ids=[101, 102, 201, 202], top_k=3)
learner_rec.observe(user_id=7, item_id=ranked[0].item_id, correct=True)
```

`policy="auto"` uses the progression-value policy: the most stable default for
adaptive learning today. Delayed-gain and support-constrained delayed-gain
policies are available as explicit opt-ins when you have the logged support and
reward-model diagnostics to justify them.

### Generic recommender fallback

Use `OrchidRecommender` when you need ordinary batch recommendations or live
adaptation without learning concepts, difficulty, or prerequisites:

```python
from orchid_ranker import OrchidRecommender

rec = OrchidRecommender.from_interactions(interactions, strategy="neural_mf")
streamer = rec.as_streaming(lr=0.05)

streamer.observe(user_id=7, item_id=42, correct=True, category="onboarding")
top5 = streamer.rank(user_id=7, candidate_item_ids=[1, 2, 3, 42, 99], top_k=5)
```

## Try it

Get up and running in under 5 minutes:

- [Quickstart example](examples/quickstart.py) --- full working script
- [Adaptive learning quickstart](examples/adaptive_learning_quickstart.py) --- learner state + prerequisites + live re-ranking
- [Scenario selection quickstart](examples/scenario_selection.py) --- choose the right Orchid workflow from product/data signals
- [Knowledge tracing quickstart](examples/knowledge_tracing_quickstart.py) --- SAKT-style predicted correctness
- [AKT quickstart](examples/akt_quickstart.py) --- difficulty-aware monotonic attention tracing
- [KT policy quickstart](examples/kt_policy_quickstart.py) --- rank eligible items by predicted learning value
- [Offline policy evaluation quickstart](examples/offline_policy_evaluation_quickstart.py) --- IPS/SNIPS/doubly robust rollout checks
- [Progression policy quickstart](examples/progression_policy_quickstart.py) --- reward stretch and learning progress, not just correctness
- [pyKT bridge quickstart](examples/pykt_bridge_quickstart.py) --- export pyKT sequences and reuse pyKT prediction tables
- [Docs quickstart](docs/quickstart.md) --- install, fit, recommend, evaluate
- [Adaptive learning positioning](docs/adaptive-learning-positioning.md) --- what the library is for and not for
- [Algorithm roadmap](docs/algorithm-roadmap.md) --- KT, semantic exercise recommendation, and policy-learning direction
- [Fit offline guide](docs/guides/01-fit-offline.md) --- train on a CSV, save, evaluate
- [Usage scenarios](docs/scenarios.md) --- practical recipes for common Orchid deployments

## Build with it

Go from adaptive-learning fit to monitored rollout:

- [Serve streaming](docs/guides/02-serve-streaming.md) --- generic live adaptation when learning metadata is unavailable
- [Operate safely](docs/guides/03-operate-safely.md) --- add progression guardrails, Prometheus metrics, Grafana dashboards

## Evaluate it

Understand what makes Orchid different:

- [Why Orchid](docs/why-orchid.md) --- the adaptive-learning thesis, five pillars, honest comparison
- [Competitor comparison](docs/competitors.md) --- when to use Orchid vs RecBole, Merlin, implicit, LightFM, TFRS, Gorse
- [ASSISTments KT benchmark](docs/benchmarks/assistments-kt.md) --- adaptive-learning correctness and policy evidence
- [KT policy OPE benchmark](docs/benchmarks/kt-policy-ope.md) --- evaluate KT-guided next-item policies before rollout
- [Specialty benchmarks](docs/benchmarks/end-to-end.md) --- cold-start, MovieLens, music, and adjacent progression modules
- [Progression policy](docs/progression-policy.md) --- transparent reward design for adaptive sequencing
- [pyKT integration](docs/pykt-integration.md) --- use Orchid around research KT model zoos

---

## Adaptive-learning capabilities

1. **Learner state.** AKT/SAKT and Bayesian tracing estimate competence from learner outcomes.
2. **Catalog structure.** Dependency graphs and difficulty metadata keep recommendations in the valid next-step set.
3. **Adaptive ranking.** Per-user online updates let the next recommendation change after each response.
4. **Progression metrics.** Evaluate learning gain, category coverage, stretch fit, and sequence adherence.
5. **Offline policy evaluation.** IPS, SNIPS, direct-method, and doubly robust estimates test adaptive policies before rollout.
6. **Safe operation.** Guardrails and frozen fallback rankings keep adaptive rollouts reviewable.
7. **Privacy hooks.** DP-SGD presets, RBAC, and HMAC audit chains support regulated deployments.

## Supported strategies

| Strategy | Type | Best for |
|----------|------|----------|
| `auto` | Selector | Let Orchid choose legacy binary MF or `explicit_mf` |
| `als` | Legacy binary MF | Backward-compatible quick start; use `implicit_als` for true ALS |
| `explicit_mf` | Matrix factorization | Explicit rating scales |
| `neural_mf` | Neural MF | Streaming adaptation through `as_streaming()` |
| `linucb` | Contextual bandit | Cold-start exploration |
| `user_knn` | Collaborative filtering | Small catalogs |
| `popularity` | Non-personalized | Baseline comparison |
| + 4 more | | See `OrchidRecommender.available_strategies()` |

Install `orchid-ranker[implicit]` to use true `implicit_als` or `implicit_bpr`.

For adaptive learning, start with `AdaptiveLearningEngine`. It composes
AKT/SAKT-style tracing, progression reward, difficulty/prerequisite metadata,
and live `observe()` updates into one fit/rank/observe API. Use lower-level
pieces such as `BayesianKnowledgeTracing`, `DependencyGraph`,
`ProgressionRecommender`, `orchid_ranker.kt.SAKTTracer`, and
`orchid_ranker.kt.AKTTracer` only when you need a custom policy. Use
`orchid_ranker.ope` to evaluate a new learning policy from logged randomized
traffic before serving it. Modern KT and policy-learning algorithms are
tracked in the [algorithm roadmap](docs/algorithm-roadmap.md).

For the high-level `OrchidRecommender` API, use `neural_mf` when you want to
promote a fitted model into `StreamingAdaptiveRanker` with `as_streaming()`.
`TwoTowerRecommender` remains available as a lower-level advanced model API, but
it is not a `strategy=` value.

## Status

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

## License

Apache 2.0. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

## Citation

```bibtex
@software{orchid_ranker,
  title={Orchid Ranker: Adaptive-Learning Engine},
  author={Sam Urmian},
  year={2024},
  url={https://github.com/mlgorithm/orchid-ranker}
}
```
