# Contributing to Orchid Ranker

Thank you for your interest in contributing to Orchid Ranker! This document provides guidelines and information for contributors.

## Vocabulary Guidelines

All new code **must** use the domain-neutral terms below. Education-origin terms are kept as deprecated aliases for one minor version, then removed. See `docs/guides/migration-0.4-to-0.5.md` for the full migration guide.

| Old (education-origin) | New (domain-neutral) | Notes |
|:----------------------|:---------------------|:------|
| `student` | `user` | Everywhere except `learning/` submodule |
| `student_ability` | `user_competence` | Public API |
| `mastered_skills` | `successful_categories` | Public API |
| `total_skills` | `total_categories` | Public API |
| `mastery_threshold` | `success_threshold` | Public API |
| `mastered` (verb/adj) | `succeeded` or `acquired` | Context-dependent |
| `mastery` (noun) | `competence` | Class names, methods |
| `learner` | `user` | Everywhere |
| `progression_gain` | `progression_gain` | **KEEP** — generic enough |
| `proficiency_coverage` | `category_coverage` | Public API |
| `sequence_adherence` | `sequence_adherence` | **KEEP** — generic |
| `difficulty_appropriateness` | `stretch_fit` | Public API |
| `ZPD` / `zone_of_proximal_development` | `stretch_zone` | Docs and prose |
| `zpd_width` | `stretch_width` | Public API |
| `zpd_weight` | `stretch_weight` | Public API |
| `curriculum` | `structured_catalog` | Docs prose |
| `prerequisite_graph` | `prerequisite_graph` | **KEEP** — generic |
| `skill` | `category` or `attribute` | Context-dependent |
| `knowledge_tracing` | `outcome_tracing` | In prose only; module name stays |
| `teacher` (two-tower) | `anchor` | Private attr for frozen copy |
| `student_agent.py` | `user_agent.py` | File rename (future) |
