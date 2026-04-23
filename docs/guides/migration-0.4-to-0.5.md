# Migration Guide: v0.4 → v0.5

This guide covers all breaking changes and deprecations in the v0.5 release. The vocabulary has been migrated from education-specific terminology to domain-neutral terms. **All old names still work in v0.5 with `DeprecationWarning`s** and will be removed in v1.0.

## Quick summary

Orchid Ranker v0.5 reframes the library as a general-purpose progression-aware recommender. Education-origin terms have been replaced with domain-neutral equivalents. No code changes are required immediately — all old names emit deprecation warnings pointing to the new name.

## Symbol renames

### Functions

| Old name | New name | Module |
|----------|----------|--------|
| `difficulty_appropriateness()` | `stretch_fit()` | `orchid_ranker.evaluation` |
| `proficiency_coverage()` | `category_coverage()` | `orchid_ranker.evaluation` |
| `learning_gain()` | `progression_gain()` | `orchid_ranker.evaluation` |
| `knowledge_coverage()` | `category_coverage()` | `orchid_ranker.evaluation` |
| `curriculum_adherence()` | `sequence_adherence()` | `orchid_ranker.evaluation` |

### Classes

| Old name | New name | Module |
|----------|----------|--------|
| `EducationalReport` | `ProgressionReport` | `orchid_ranker.evaluation` |
| `PrerequisiteGraph` | `DependencyGraph` | `orchid_ranker.curriculum` |
| `CurriculumRecommender` | `ProgressionRecommender` | `orchid_ranker.curriculum` |
| `ProficiencyTracker` | `CompetencyTracker` | `orchid_ranker.knowledge_tracing` |

### Methods

| Old name | New name | Class |
|----------|----------|-------|
| `.mastery()` | `.competence()` | `BKTStateProvider`, `StreamingAdaptiveRanker` |
| `.mastered()` | `.succeeded()` | `ProficiencyTracker` / `CompetencyTracker` |
| `.unmastered()` | `.remaining()` | `ProficiencyTracker` / `CompetencyTracker` |
| `.available_skills()` | `.available_categories()` | `DependencyGraph` |

### Parameters

| Old param | New param | Where |
|-----------|-----------|-------|
| `student_ability` | `user_competence` | `stretch_fit()` |
| `zpd_width` | `stretch_width` | `stretch_fit()`, `RollingProgressionMonitor` |
| `mastered_skills` | `successful_categories` | `category_coverage()` |
| `total_skills` | `total_categories` | `category_coverage()` |
| `mastered` | `succeeded` | `sequence_adherence()`, `DependencyGraph` methods |
| `mastery_threshold` | `success_threshold` | `CompetencyTracker`, `RollingProgressionMonitor` |
| `skill` | `category` | `RollingProgressionMonitor.record()`, `BKTStateProvider`, `StreamingAdaptiveRanker` |
| `pre_mastery` / `post_mastery` | `pre_competence` / `post_competence` | `RollingProgressionMonitor.record()` |
| `default_skill` | `default_category` | `BKTStateProvider` |
| `student_mastery` | `user_competence` | `ProgressionRecommender.recommend()` |

### Dataclass fields

| Old field | New field | Class |
|-----------|-----------|-------|
| `proficiency_coverage` | `category_coverage` | `ProgressionSnapshot` |
| `difficulty_appropriateness` | `stretch_fit` | `ProgressionSnapshot` |
| `coverage` | `category_coverage` | `ProgressionReport` |
| `difficulty_fit` | `stretch_fit` | `ProgressionReport` |

## New API additions

These are new in v0.5 — no migration needed:

| Addition | Description |
|----------|-------------|
| `OrchidRecommender.from_interactions()` | One-call fit from a DataFrame |
| `OrchidRecommender.as_streaming()` | Bridge to `StreamingAdaptiveRanker` |
| `OrchidRecommender.baseline_rank()` | Frozen fallback ranking for guardrail halts |
| `.tower` property | Access the neural tower module |
| `.user_features` property | Auto-materialized user feature tensor |
| `.item_features` property | Auto-materialized item feature tensor |

## How to migrate

### Step 1: Find deprecation warnings

Run your test suite with warnings visible:

```bash
python -W all -m pytest your_tests/
```

### Step 2: Update imports

```python
# Before
from orchid_ranker.evaluation import difficulty_appropriateness, proficiency_coverage
from orchid_ranker.curriculum import CurriculumRecommender

# After
from orchid_ranker.evaluation import stretch_fit, category_coverage
from orchid_ranker.curriculum import ProgressionRecommender
```

### Step 3: Update function calls

```python
# Before
score = difficulty_appropriateness(difficulties, student_ability=0.6, zpd_width=0.25)
coverage = proficiency_coverage(mastered_skills=achieved, total_skills=all_skills)

# After
score = stretch_fit(difficulties, user_competence=0.6, stretch_width=0.25)
coverage = category_coverage(successful_categories=achieved, total_categories=all_skills)
```

### Step 4: Update method calls

```python
# Before
p = tracker.mastered()
m = streamer.mastery(user_id=7)

# After
p = tracker.succeeded()
m = streamer.competence(user_id=7)
```

## Timeline

- **v0.5.0**: All old names work with `DeprecationWarning`
- **v1.0.0**: Old names removed. Code using old names will raise `AttributeError` or `TypeError`

## Questions?

Open an issue on [GitHub](https://github.com/mlgorithm/orchid-ranker/issues).
