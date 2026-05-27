# Algorithm roadmap

This roadmap separates what Orchid already ships from the adaptive-learning
algorithms that would make the library feel more current and research-grade.
The goal is not to chase every recommender paper. The goal is to add algorithms
that strengthen the adaptive-learning loop: trace knowledge, choose the next
item, evaluate safely, and explain why.

## Current foundation

| Area | Shipped today | Role in adaptive learning |
|------|---------------|---------------------------|
| Learner state | `BayesianKnowledgeTracing`, `fit_bkt_em`, `PFATracer`, `AFMTracer`, `DKTTracer`, `DKVMNTracer`, `SAINTPlusTracer`, `SAINTTracer`, `AKTTracer`, `SAKTTracer` | Online competence and correctness estimates per learner and item |
| Catalog constraints | `DependencyGraph`, `ProgressionRecommender` | Prerequisite-aware eligibility and path ordering |
| Adaptive ranking | `AdaptiveLearningEngine`, `ProgressionValuePolicy` | Next-item ranking from KT, difficulty, prerequisites, and progression reward |
| Adaptive testing | `IRTAdaptiveSelector` | Placement tests and mastery checks via item information |
| Live adaptation | `AdaptiveLearningEngine.observe`, `StreamingAdaptiveRanker` | Per-outcome updates without full retraining |
| Offline policy evaluation | `orchid_ranker.ope`, `TabularFQE` | IPS/SNIPS/direct-method/doubly robust, bootstrap, rollout gates, and fitted-Q checks before adaptive rollout |
| Personalized exploration | `PersonalizedLinUCB` | User-conditioned exploration from explicit learner/item feature maps |
| Retention | `FSRSScheduler` | Forgetting-aware review scheduling |
| Semantic retrieval | `orchid_ranker.semantic` | Text/metadata candidate generation for sparse or new exercises |
| Safety | progression monitors, guardrails, safe fallback | Stops harmful adaptive behavior before broad rollout |

This is a strong product foundation, and the next modeling layer should keep
removing the generic recommender fallback from the main narrative so Orchid is
evaluated as an adaptive-learning engine.

## Research direction

As of May 2026, the most relevant model families are:

| Priority | Family | Why it belongs in Orchid | Reference starting points |
|----------|--------|--------------------------|---------------------------|
| P0 | Deep and attentive knowledge tracing | Learner-state models are the center of adaptive learning. Orchid should support sequence models that predict future correctness, not only BKT. | [DKT](https://papers.nips.cc/paper/5654-deep-knowledge-tracing), [SAKT](https://arxiv.org/abs/1907.06837), [AKT](https://arxiv.org/abs/2007.12324), [KT survey](https://arxiv.org/abs/2201.06953) |
| P0 | Standardized KT benchmarking | Prevents inflated claims and makes algorithm wins reproducible. | [pyKT](https://arxiv.org/abs/2206.11460) |
| P1 | Semantic exercise recommendation | Uses item text or metadata so new questions can be recommended before they have many interactions. | [ExRec](https://arxiv.org/abs/2507.11060), [OpenReview record](https://openreview.net/forum?id=ILZ7ZPEHD5) |
| P1 | KT-guided policy optimization | Optimizes learning gain over a path, not only next-item relevance. Now that OPE is shipped, the next step is logged-policy benchmarks and conservative serving gates. | [ALPN](https://arxiv.org/abs/2305.04475), [ExRec](https://arxiv.org/abs/2507.11060) |
| P2 | Explainable KT and prerequisite reasoning | Helps teachers and operators trust the recommendation path. | [Explainable KT survey](https://arxiv.org/abs/2403.07279) |
| P2 | LLM-assisted metadata | Useful for concept labels, item explanations, and prerequisite suggestions, but should stay outside the trusted ranking core until validated. | LLM-KT and learning-path work should be treated as experimental |

## Proposed library shape

Add adaptive-learning algorithms under explicit, boring names:

| Module | Proposed API | Purpose |
|--------|--------------|---------|
| `orchid_ranker.kt` | `SAKTTracer`, `AKTTracer`, `SAINTTracer`, `SAINTPlusTracer`, `DKTTracer`, `DKVMNTracer` | Predict next correctness and expose learner-state vectors |
| `orchid_ranker.edm` | `PFATracer`, `AFMTracer` | Classical interpretable EDM baselines |
| `orchid_ranker.bkt_em` | `fit_bkt_em` | Estimate BKT parameters from learner sequences |
| `orchid_ranker.calibration` | `TemperatureScaler`, `IsotonicProbabilityCalibrator`, `expected_calibration_error` | Calibrate predicted correctness before policy serving |
| `orchid_ranker.learning_policy` | `KTValuePolicy`, future `StretchBanditPolicy` | Choose next item from eligible candidates using expected progress |
| `orchid_ranker.bandits` | `PersonalizedLinUCB` | Safe personalized exploration using `phi(learner, item)` features |
| `orchid_ranker.spaced_repetition` | `FSRSScheduler` | Review urgency and retention-aware ranking |
| `orchid_ranker.fqe` | `TabularFQE` | Model-based fixed-policy value cross-check |
| `orchid_ranker.pykt_bridge` | `export_pykt_sequences`, `PyKTPredictionAdapter` | Interoperate with pyKT research models and bring predictions back into Orchid policy/OPE |
| `orchid_ranker.semantic` | `SemanticItemEncoder`, `SemanticExerciseRanker` | Score new exercises using text/metadata embeddings |
| `orchid_ranker.ope` | `evaluate_logged_policy`, `compare_logged_policies`, bootstrap variants | Evaluate adaptive policies before serving them |

Keep `AdaptiveRanker` and `AdaptiveLearningEngine` as the beginner APIs for the
primary adaptive-learning workflow. Historical generic recommenders stay under
`orchid_ranker.legacy` only for compatibility and old experiment replay.

## Implementation order

1. **Transformer KT baseline.** First experimental slice is now
   `orchid_ranker.kt.SAKTTracer`: item/response embeddings, self-attention over
   prior learner events, and a probability-correct head.
2. **Benchmark SAKT.** Added `orchid_ranker.kt_benchmark` and
   `benchmarks/kt_sakt_benchmark.py` for time-ordered replay evaluation against
   an item-mean baseline on ASSISTments/EdNet-style CSVs.
3. **ASSISTments preprocessing.** Added `benchmarks/assistments/preprocess.py`
   for classic ASSISTments and FoundationalASSIST raw CSVs, plus a tiny fixture
   smoke path. Full public-dataset metrics still require a locally downloaded
   dataset.
4. **KT-guided next-item policy.** Added experimental
   `orchid_ranker.learning_policy.KTValuePolicy`, which ranks eligible items by
   stretch fit, uncertainty, and an expected-gain proxy.
5. **AKT-inspired variant.** Added experimental `orchid_ranker.kt.AKTTracer`
   with item difficulty features and recency-biased monotonic attention.
6. **Offline policy evaluation.** Added `orchid_ranker.ope` with IPS, SNIPS,
   direct-method, doubly robust estimates, paired uplift, effective sample size,
   coverage, weight diagnostics, and confidence intervals.
7. **Logged policy benchmarks.** Use OPE on ASSISTments/EdNet-style action logs
   to evaluate adaptive policies against item mean, random eligible, and frozen
   curriculum baselines. The first multi-seed ASSISTments slice reports mean
   doubly robust uplift of -0.0028 over a random-uniform candidate baseline for
   `KTValuePolicy` under synthetic candidate propensities, so the original
   heuristic policy is not a policy-lift win at `target_correct=0.70`.
8. **Progression reward policy.** Added `orchid_ranker.progression_reward` and
   `orchid_ranker.learning_policy.ProgressionValuePolicy`. A first
   ASSISTments OPE replay reports +0.3212 mean doubly robust uplift on the
   progression reward objective, versus +0.1880 for an easy-correctness
   sensitivity run. This makes the reward-design distinction explicit:
   correctness optimization is not the same as learning progress.
9. **pyKT bridge.** Added `orchid_ranker.pykt_bridge` to export Orchid logs to
   pyKT's six-line learner-sequence format and wrap pyKT prediction tables for
   `KTValuePolicy` and OPE. This makes Orchid the production/evaluation layer
   around external KT model-zoo outputs.
10. **Adaptive-first facade.** Added `AdaptiveRanker` with staged KT,
    reward-model, logged-policy, sketch-mode, OPE, and observe workflows.
11. **SAINT and SAINT+.** Added `SAINTTracer` and `SAINTPlusTracer` to the
    in-repo KT registry. SAINT+ uses elapsed-time and lag-time history features
    when timestamped events are available.
12. **Semantic cold start.** Added `SemanticItemEncoder` and
    `SemanticExerciseRanker` for deterministic text/metadata candidate
    retrieval before final adaptive reranking. The `AdaptiveRanker` facade can
    also score semantic catalog items that were not present in KT training.
    `DenseSemanticItemEncoder` adds the provider-neutral adapter for modern
    dense embeddings.
13. **Bootstrap OPE intervals.** Added row- and cluster-bootstrap OPE reports
    for single policies and paired comparisons.
14. **Propensity-aware tabular CQL.** The lightweight CQL-style policy now uses
    clipped, normalized inverse-propensity update weights from logged decisions.
15. **IRT adaptive selector.** Added a small Rasch/2PL/3PL-compatible selector
    for placement tests and mastery checks using item information.
16. **Curated adaptive baseline collection.** Added DKT and DKVMN-style tracers,
    PFA/AFM, EM fitting for BKT, probability calibration helpers, FSRS-style
    retention scheduling, personalized LinUCB, and tabular FQE. These are the
    "best useful collection" layer: enough breadth to compare adaptive methods
    honestly without turning Orchid into a generic recommender model zoo.

## Claim discipline

Do not call an algorithm "SOTA" in user-facing docs until it has:

- a public benchmark script in `benchmarks/`
- a documented train/test protocol with leakage checks
- at least one strong baseline from the same task family
- an adaptive-learning metric, not only AUC or NDCG

Until then, use precise language: "SAKT-style", "AKT-inspired",
"KT-guided", or "experimental".
