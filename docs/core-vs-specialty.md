# Core adaptive-learning surface

Orchid Ranker is an adaptive-learning and knowledge-tracing library. The core
surface is intentionally narrow: modules are core only when they support
next-learning-item decisions from learner state, catalog structure, progression
reward, logged-policy evidence, or safe rollout.

## Core

The core library serves domains that satisfy three conditions:

1. Items have learning structure: concepts, difficulty, prerequisites, review
   timing, or authored progression.
2. Interactions produce outcome signals such as correctness, completion,
   mastery, retention, adherence, or skill gain.
3. The stakeholder measures long-term progress, not only short-term clicks.

Core modules:

| Area | APIs |
|------|------|
| Product facade | `AdaptiveRanker`, `AdaptiveLearningEngine`, `AdaptiveLearningRecommender` |
| Learner state | `SAKTTracer`, `AKTTracer`, `SAINTTracer`, `SAINTPlusTracer`, `DKTTracer`, `DKVMNTracer` |
| Classical KT / EDM | `BayesianKnowledgeTracing`, `PFATracer`, `AFMTracer`, `fit_bkt_em` |
| Adaptive testing | `IRTAdaptiveSelector`, `IRTItem` |
| Progression structure | `DependencyGraph`, `ProgressionRecommender`, `ProgressionValuePolicy`, `ProgressionRewardConfig` |
| Semantic exercise retrieval | `SemanticItemEncoder`, `DenseSemanticItemEncoder`, `SemanticExerciseRanker` |
| Logged policy evidence | `CQLDiscretePolicy`, `TabularFQE`, `evaluate_logged_policy`, `compare_logged_policies`, `evaluate_rollout_gate` |
| Calibration and support diagnostics | `TemperatureScaler`, `IsotonicProbabilityCalibrator`, delayed-gain diagnostics |
| Retention and exploration | `FSRSScheduler`, `PersonalizedLinUCB` |
| Safety and operations | `ProgressionGuardrail`, `SafeSwitchDR`, audit, observability, connectors |

## Advanced

Advanced modules are useful for research, simulation, or large deployments, but
they do not define Orchid's public claim by themselves:

| Area | APIs |
|------|------|
| Agentic simulation | `AdaptiveAgent`, `AdaptiveAgentFactory`, `MultiUserOrchestrator`, `MultiConfig` |
| Candidate narrowing | `BloomFilter`, `CountMinSketch`, `ReservoirSampler`, `SketchCandidateGenerator` |
| Optional operations | visualization, Prometheus metrics, RBAC/audit, Snowflake/BigQuery/S3/MLflow connectors |

Generic collaborative filtering, movie/music recommendation, commerce taste
ranking, and social-feed ranking are not Orchid's product surface. Historical
implementation files may remain temporarily while tests and migration paths are
unwound, but new docs, examples, benchmarks, and top-level exports should not
promote them.

## Graduation Bar

A module graduates into the core adaptive surface only when all three are true:

1. It is directly useful for adaptive learning, KT, progression, OPE, or safe
   rollout.
2. It has focused tests and an example that uses learner outcomes or catalog
   learning metadata.
3. Its claims are backed by public benchmark artifacts or clearly marked as
   experimental.

This keeps Orchid's claim precise: best-in-class adaptive learning and
knowledge-tracing recommendation, not another general recommender toolkit.
