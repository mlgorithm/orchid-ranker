# Adaptive learning positioning

Orchid Ranker is a Python library for adaptive-learning products that need to
choose the next best learning item from three signals:

- what the learner appears to know now
- what the catalog says is allowed next
- what the last outcome says about the learner's trajectory

The library is not trying to be a generic movie, music, ads, or social-feed
recommender. Its commercial fit is narrower and easier to defend: products
where recommendations should improve competence, completion, confidence, or
safe progression over time.

## Primary market

The first market is adaptive learning:

| Buyer or builder | Job Orchid helps with |
|------------------|-----------------------|
| EdTech platform teams | Pick the next exercise, lesson, quiz, or review item after each response |
| Corporate learning teams | Reduce time-to-competence and keep certification paths auditable |
| Test-prep and tutoring products | Keep learners in the stretch zone without violating prerequisites |
| Internal enablement platforms | Recommend the next onboarding or product task based on demonstrated adoption |

Adjacent markets such as rehabilitation, fitness, and skill-based games can use
the same progression model, but they should not blur the core message. The
homepage, quickstart, and examples should lead with adaptive learning.

## Product promise

Use Orchid when the product question is:

> Given this learner's current state and this structured catalog, what should
> they work on next so they make measurable progress?

That promise breaks into four capabilities:

| Capability | What it means in the product |
|------------|------------------------------|
| Learner state | Track competence from outcomes, not just clicks |
| Catalog structure | Respect prerequisites, difficulty, and eligible item pools |
| Adaptive ranking | Re-rank immediately after a response, completion, or failure |
| Safe operation | Measure progression and fall back when the adaptive policy degrades |

## What Orchid should not claim

Avoid positioning Orchid as:

- a general replacement for RecBole, Merlin, LightFM, or TensorFlow Recommenders
- a CTR or watch-time optimizer
- a full LMS, content authoring system, or tutoring UI
- a model zoo with every recommender architecture
- an LLM tutor that generates teaching content by itself

LLMs can be useful later for content metadata, explanations, or prerequisite
suggestions, but the core library should remain the ranking and progression
engine.

## Success metrics

The business story should use learning and progression metrics:

| Metric | Why it matters |
|--------|----------------|
| Learning gain | Direct measure of user outcome improvement |
| Time-to-competence | Commercial ROI for training and test prep |
| Mastery probability | Per-concept state for adaptive decisions |
| Stretch-zone hit rate | Whether recommendations are appropriately hard |
| Prerequisite violation rate | Whether ranking respects the curriculum contract |
| Safe fallback rate | Whether adaptive serving is operationally trustworthy |

CTR, raw engagement, and session length are secondary. They can be monitored,
but they should not be the primary optimization target.

## First-run story

The first example should show this loop:

1. Load a small learning catalog with concepts, difficulty, and prerequisites.
2. Fit a model from historical learner outcomes.
3. Build the eligible candidate set from the prerequisite graph.
4. Observe a live answer.
5. Re-rank immediately for the same learner.
6. Print competence and latency so the user sees adaptation happened.

That story is the simplest defensible answer to "what is this library for?"

## Evidence stance

The honest evidence story is:

- Generic collaborative filtering is not Orchid's strongest wedge; specialized
  libraries such as `implicit` can win plain top-K recommendation benchmarks.
- Adaptive correctness prediction is promising: AKT improves over item-mean
  correctness baselines on ASSISTments-style replay.
- The most defensible adaptive policy today is the progression-value policy,
  which shows positive offline policy-evaluation uplift on progression reward.
- Delayed-gain policies are still experimental. They need stronger support,
  calibration, and real logged propensities before becoming a headline claim.

That means public messaging should lead with **KT + progression reward +
prerequisites + live `observe()` + safe rollout**, not with generic recommender
benchmark dominance.
