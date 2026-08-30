# Adaptive-practice product roadmap

Orchid’s first product is an embedded adaptive-practice engine for assessed
professional and technical learning. The goal is not to build an LMS, a generic
feed recommender, or a new knowledge-tracing model zoo. The goal is to help one
learning product choose the next eligible exercise and demonstrate whether that
path improves retained mastery over its authored path.

## Phase 0 — Honest learning foundation

**Status: delivered in this release.**

- Position Orchid around adaptive practice rather than generic recommendation.
- Add data readiness and a transparent empirical fallback for sparse pilots.
- Make delayed retained mastery the north-star outcome; keep next-attempt
  correctness as a diagnostic/adaptation signal.
- Keep exact candidate sets, policy versions, propensities, and outcomes in the
  decision record.
- Keep CQL and delayed-gain policy promotion experimental and out of the
  headline product path.

Exit criterion: a learning-platform engineer can fit completed attempts,
inspect data readiness, produce an eligible candidate set, understand the
recommendation, and safely use the empirical path during a pilot.

## Phase 1 — Pilot-grade integration

**Status: foundations delivered; application integration remains.**

- Provide a versioned exercise-catalog validator: exercise, skill,
  author-reviewed difficulty, prerequisites, course/module, and
  assessment-only status; it reports conflicting metadata, incomplete skill
  coverage, dangling prerequisites, and cycles.
- Provide idempotent immutable decision/outcome storage with an in-memory
  default and a single-host SQLite implementation. An application still owns
  learner-state recovery, experiment assignment, and model/policy snapshots.
- Provide a reference adaptive-practice pilot adapter with a frozen catalog
  snapshot, author-controlled eligibility, sticky control/treatment routing,
  and shared immutable decision/outcome records. A real LMS connector and
  delayed-assessment importer remain application integration work.
- Integrate catalog versioning and assessment-only exclusions into the
  application's eligibility query; the ranker must receive only eligible IDs.
- Add deterministic replay of a learner path from catalog, event log, version,
  and random seed.
- Add learning-design explanations: target skill, predicted challenge band,
  prerequisite status, repetition, and support level.
- Add authored baselines: fixed sequence, prerequisite-first, and
  most-missed-skill.

These are the non-negotiable delivery gates from the simulated
[design-partner council](design-partner-council.md): a durable event path,
versioned curriculum metadata, an authored fallback, and designer-visible
reasons are required before a production pilot.

Exit criterion: one course can run Orchid beside an authored sequence, replay
any learner path, and explain every served exercise to a learning designer.

## Phase 2 — Prove learning impact

- Run an A/A logging check, then shadow recommendations.
- Randomize learners between the existing authored path and Orchid, holding
  content, eligibility, UI, and time allowance constant.
- Pre-specify a primary independent delayed outcome: retention quiz,
  assessment on unserved items, or certification pass result.
- Calculate sample size from the partner’s historical variance and attrition;
  do not use a universal event threshold.
- Monitor completion, repeated failures, time/items to mastery, outcome-join
  rate, and subgroup outcomes as guardrails.

Exit criterion: a powered, reproducible treatment-vs-control result for one
defined course and learner population, with no material learner-experience
regression.

## Phase 3 — Package the evidence-backed workflow

- Publish the winning integration as an Adaptive Practice Starter for the same
  learning ecosystem.
- Build engineer, learning-designer, and program-owner views around the proven
  workflow.
- Add richer outcomes, multi-skill items, spaced review, instructor overrides,
  and featureized policy learning only where real logged support exists.
- Consider CQL promotion only after reusable learning-state features replace
  per-user context hashes and a real randomized log clears the existing
  chronological rollout gate.

Exit criterion: repeat the same controlled outcome in a second course before
making broad learning-efficacy claims.

## What we will not prioritize first

- Generic content, product, music, or feed recommendation.
- High-stakes access or credentialing decisions.
- A hosted LMS, content-authoring suite, or chatbot tutor.
- More KT architectures before the simple baselines, curriculum contract, and
  real experiment are in place.
