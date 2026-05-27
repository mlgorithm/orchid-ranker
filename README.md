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
pip install 'orchid-ranker[adaptive]'
```

```python
import pandas as pd
from orchid_ranker import AdaptiveRanker

outcomes = pd.read_csv("learner_outcomes.csv")  # learner_id, item_id, correct, ts
catalog = pd.read_csv("exercise_catalog.csv")   # item_id, concept_id, difficulty, item_text

ranker = AdaptiveRanker(
    kt_backbone="saint+",
    policy="auto",
    epochs=2,
    d_model=32,
).fit_kt(
    outcomes.merge(catalog, on="item_id"),
    correct_col="correct",
    concept_col="concept_id",
    item_difficulty_col="difficulty",
)

ranker.fit_semantic_items(catalog, text_col="item_text", metadata_cols=["concept_id"])
ranked = ranker.recommend(learner_id="7", candidate_item_ids=[101, 102, 201, 202], top_k=3)
ranker.observe(learner_id="7", ts=123, item_id=ranked[0].item_id, concept_id=None, correct=1)
```

`policy="auto"` uses the progression-value policy: the most stable default for
adaptive learning today. Delayed-gain and support-constrained delayed-gain
policies are available as explicit opt-ins when you have the logged support and
reward-model diagnostics to justify them.

### Compatibility note

Orchid's public product surface is adaptive learning. Historical generic
recommender APIs are kept under `orchid_ranker.legacy` only for migration and
old experiment replay.

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

1. **Learner state.** SAINT+/SAINT, AKT/SAKT, DKT/DKVMN-style, PFA/AFM, BKT, and IRT components estimate competence from learner outcomes.
2. **Catalog structure.** Dependency graphs and difficulty metadata keep recommendations in the valid next-step set.
3. **Semantic cold start.** Hashing and dense-adapter semantic encoders retrieve new exercises from text and metadata before interaction support exists.
4. **Adaptive ranking.** Per-user online updates let the next recommendation change after each response.
5. **Adaptive testing.** `IRTAdaptiveSelector` supports Rasch/2PL/3PL-style placement and mastery-check item selection by information.
6. **Logged policy learning.** `AdaptiveRanker.fit_policy(..., algo="cql")` trains a conservative CQL-style contextual-bandit policy from candidate sets, rewards, and normalized inverse-propensity update weights.
7. **Personalized exploration.** `PersonalizedLinUCB` scores `phi(learner, item)` features instead of item-only bandit features.
8. **Retention scheduling.** `FSRSScheduler` adds FSRS-style review urgency for forgetting-aware practice.
9. **Sketch mode.** Count-Min, Bloom-filter, reservoir, and exact-vector utilities shrink candidate generation before final reranking.
10. **Offline policy evaluation.** IPS, SNIPS, direct-method, doubly robust, bootstrap, rollout gates, and tabular FQE test adaptive policies before rollout.
11. **Safe operation.** Guardrails and frozen fallback rankings keep adaptive rollouts reviewable.
12. **Privacy hooks.** Opt-in DP-SGD presets, RBAC, HMAC audit chains, and hashed event IDs support regulated deployments.

## Adaptive algorithm collection

| Family | Orchid APIs | Scenario |
|--------|-------------|----------|
| Transformer KT | `SAKTTracer`, `AKTTracer`, `SAINTTracer`, `SAINTPlusTracer` | Main next-correctness state models |
| Recurrent / memory KT | `DKTTracer`, `DKVMNTracer` | Compact sequence baselines and ablations |
| Classical EDM | `PFATracer`, `AFMTracer`, `fit_bkt_em`, `BayesianKnowledgeTracing` | Small-data, interpretable learner-state baselines |
| Adaptive testing | `IRTAdaptiveSelector`, `IRTItem` | Placement, mastery checks, item-information selection |
| Semantic retrieval | `SemanticItemEncoder`, `DenseSemanticItemEncoder`, `SemanticExerciseRanker` | Cold-start and metadata-aware exercise retrieval |
| Policy learning | `CQLDiscretePolicy`, `TabularFQE`, `evaluate_logged_policy` | Logged-policy learning and rollout evidence |
| Exploration | `PersonalizedLinUCB` | Safe personalized exploration with explicit feature maps |
| Retention | `FSRSScheduler` | Review scheduling and forgetting-risk ranking |

For adaptive learning, start with `AdaptiveRanker` when you want staged
KT/reward/policy/OPE workflows, or `AdaptiveLearningEngine` when you only need
fit/rank/observe. They compose
SAINT+/SAINT, AKT/SAKT, DKT/DKVMN-style tracing, progression reward,
difficulty/prerequisite metadata, semantic item retrieval, and live
`observe()` updates into one fit/rank/observe API. Use lower-level
pieces such as `BayesianKnowledgeTracing`, `DependencyGraph`,
`ProgressionRecommender`, `orchid_ranker.kt.SAKTTracer`, and
`orchid_ranker.kt.SAINTPlusTracer` only when you need a custom policy. Use
`orchid_ranker.ope` to evaluate a new learning policy from logged randomized
traffic before serving it, `bootstrap_logged_policy` when rollout decisions
need row-resampled confidence intervals, and `evaluate_rollout_gate` to enforce
minimum support/coverage/clipping thresholds before live learners see a policy. Modern KT and policy-learning
algorithms are tracked in the [algorithm roadmap](docs/algorithm-roadmap.md).

`orchid_ranker.legacy.OrchidRecommender` remains available for old generic
recommender experiments, but it is not the main library surface.

## Status

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11--3.13-blue)
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
