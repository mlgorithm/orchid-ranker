"""Scenario recipes for choosing an Orchid workflow.

This module is deliberately lightweight and dependency-free. It gives product
teams a deterministic way to map the signals they have to the Orchid algorithm
path they should start with.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScenarioRecipe:
    """A documented Orchid workflow and its algorithm path."""

    id: str
    name: str
    summary: str
    use_when: str
    signals: tuple[str, ...]
    algorithms: tuple[str, ...]
    entrypoints: tuple[str, ...]
    docs_anchor: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioFit:
    """A scored scenario recommendation."""

    scenario: ScenarioRecipe
    score: float
    reasons: tuple[str, ...]


_SCENARIOS: tuple[ScenarioRecipe, ...] = (
    ScenarioRecipe(
        id="adaptive_learning_next_item",
        name="Adaptive-learning next item",
        summary="Choose the next exercise, task, lesson, or action from live learner state.",
        use_when=(
            "Each outcome should change the next recommendation and the catalog has "
            "learning concepts, difficulty, or prerequisite structure."
        ),
        signals=(
            "learner outcomes",
            "item concepts",
            "item difficulty",
            "optional prerequisites",
            "live outcome events",
        ),
        algorithms=(
            "AKT/SAKT knowledge tracing",
            "progression-value policy",
            "hard prerequisite gating",
            "optional delayed-gain policy after diagnostics",
        ),
        entrypoints=(
            "AdaptiveLearningRecommender.fit",
            "AdaptiveLearningRecommender.rank",
            "AdaptiveLearningRecommender.observe",
        ),
        docs_anchor="adaptive-learning-next-item-recommendation",
        tags=("adaptive", "learning", "primary"),
    ),
    ScenarioRecipe(
        id="safe_adaptive_rollout",
        name="Safe adaptive rollout",
        summary="Ship adaptive recommendations with monitoring, fallback, and OPE checks.",
        use_when="Recommendations are user-facing and policy regressions need a clear stop path.",
        signals=(
            "logged outcomes",
            "policy propensities or deterministic policy probabilities",
            "live progression metrics",
            "baseline policy",
        ),
        algorithms=(
            "offline policy evaluation",
            "progression guardrails",
            "baseline fallback",
        ),
        entrypoints=(
            "evaluate_logged_policy",
            "compare_logged_policies",
            "RollingProgressionMonitor",
            "ProgressionGuardrail",
        ),
        docs_anchor="safe-adaptive-rollout",
        tags=("adaptive", "safety", "ope"),
    ),
    ScenarioRecipe(
        id="regulated_training",
        name="Regulated training or clinical workflow",
        summary="Add auditability and privacy controls around adaptive-learning ranking.",
        use_when=(
            "The domain requires traceability, operator accountability, privacy controls, "
            "or conservative fallback behavior."
        ),
        signals=(
            "completion outcomes",
            "category or competency labels",
            "operator identity",
            "deployment metadata",
        ),
        algorithms=(
            "adaptive-learning recommender",
            "guardrail policy",
            "audit logging",
            "privacy hooks",
        ),
        entrypoints=(
            "AdaptiveLearningRecommender",
            "ProgressionGuardrail",
            "AuditLogger",
            "get_dp_config",
        ),
        docs_anchor="regulated-training-or-clinical-workflows",
        tags=("regulated", "training", "audit"),
    ),
    ScenarioRecipe(
        id="new_user_cold_start",
        name="New learner or new exercise cold start",
        summary="Use semantic exercise metadata while learner-state evidence is still sparse.",
        use_when=(
            "A learner or exercise has little history, but the catalog has text, "
            "concept, difficulty, or prerequisite metadata."
        ),
        signals=(
            "exercise text",
            "concept labels",
            "difficulty metadata",
            "early learner outcomes",
        ),
        algorithms=(
            "semantic exercise retrieval",
            "progression-aware cold-start prior",
            "prerequisite gating",
        ),
        entrypoints=(
            "AdaptiveRanker.fit_semantic_items",
            "AdaptiveRanker.recommend",
        ),
        docs_anchor="new-learner-or-new-exercise-cold-start",
        tags=("adaptive", "learning", "cold-start"),
    ),
)

_SCENARIOS_BY_ID = {scenario.id: scenario for scenario in _SCENARIOS}
_SCENARIO_ORDER = {scenario.id: index for index, scenario in enumerate(_SCENARIOS)}

_SIGNAL_RULES: dict[str, tuple[tuple[str, float, str], ...]] = {
    "adaptive_learning_next_item": (
        ("has_outcomes", 2.0, "Outcome labels can update learner state."),
        ("has_concepts", 1.5, "Concept metadata supports mastery estimates."),
        ("has_difficulty", 1.0, "Difficulty supports stretch-fit scoring."),
        ("has_prerequisites", 1.0, "Prerequisites support hard eligibility gates."),
        ("needs_live_adaptation", 2.0, "Each new outcome should affect the next item."),
    ),
    "safe_adaptive_rollout": (
        ("needs_safe_rollout", 3.0, "You need guardrails or fallback for an adaptive policy."),
        ("needs_live_adaptation", 1.0, "The policy changes during user sessions."),
        ("is_regulated", 1.0, "Regulated domains benefit from conservative rollout controls."),
        ("has_outcomes", 0.5, "Logged outcomes can support OPE and progression monitoring."),
    ),
    "regulated_training": (
        ("is_regulated", 3.0, "The workflow needs auditability or compliance controls."),
        ("has_outcomes", 1.0, "Completion outcomes support competency tracking."),
        ("has_concepts", 1.0, "Competency labels support progression reporting."),
        ("needs_safe_rollout", 1.0, "Guardrails fit regulated deployment requirements."),
    ),
    "new_user_cold_start": (
        ("has_new_users", 2.0, "Sparse learner history needs a cold-start prior."),
        ("has_item_features", 2.0, "Exercise text or metadata can power semantic retrieval."),
        ("has_concepts", 1.0, "Concept metadata keeps cold-start ranking learning-aware."),
        ("has_difficulty", 1.0, "Difficulty supports target-correctness priors for new exercises."),
        ("has_prerequisites", 0.5, "Prerequisites can still gate cold-start candidates."),
    ),
}

_KEYWORDS: dict[str, tuple[tuple[str, float, str], ...]] = {
    "adaptive_learning_next_item": (
        ("adaptive", 1.0, "The use case explicitly asks for adaptation."),
        ("learning", 1.0, "The use case is learning-centered."),
        ("learner", 1.0, "The use case is learner-centered."),
        ("student", 1.0, "The use case is student-centered."),
        ("tutor", 1.0, "Tutoring maps to next-item adaptive learning."),
        ("exercise", 1.0, "Exercises are natural adaptive-learning items."),
        ("practice", 1.0, "Practice outcomes can update learner state."),
    ),
    "safe_adaptive_rollout": (
        ("guardrail", 1.0, "Guardrails map to safe adaptive rollout."),
        ("fallback", 1.0, "Fallback behavior is a rollout-safety concern."),
        ("ope", 1.0, "OPE is part of safe policy rollout."),
        ("rollout", 1.0, "Rollout language suggests safety and monitoring."),
    ),
    "regulated_training": (
        ("regulated", 1.0, "Regulated domains need audit controls."),
        ("compliance", 1.0, "Compliance maps to regulated training."),
        ("certification", 1.0, "Certification needs auditable progression."),
        ("clinical", 1.0, "Clinical workflows need conservative controls."),
        ("audit", 1.0, "Auditability is central to this scenario."),
    ),
    "new_user_cold_start": (
        ("cold start", 1.0, "Cold-start wording maps directly to this bridge."),
        ("new learner", 1.0, "New learners need cold-start behavior."),
        ("new exercise", 1.0, "New exercises can enter through semantic metadata."),
        ("catalog", 0.5, "Catalog metadata can support semantic exercise retrieval."),
    ),
}


def available_scenarios() -> tuple[ScenarioRecipe, ...]:
    """Return the stable Orchid scenario catalog."""

    return _SCENARIOS


def recommend_scenarios(
    *,
    has_interactions: bool = False,
    has_outcomes: bool = False,
    has_concepts: bool = False,
    has_difficulty: bool = False,
    has_prerequisites: bool = False,
    needs_live_adaptation: bool = False,
    needs_safe_rollout: bool = False,
    has_new_users: bool = False,
    is_regulated: bool = False,
    has_item_features: bool = False,
    has_expertise_curve: bool = False,
    has_fresh_content: bool = False,
    use_case: str | None = None,
    top_k: int = 3,
) -> list[ScenarioFit]:
    """Rank Orchid scenarios for a product and data shape.

    Parameters are intentionally product-level booleans. The function is meant
    to be useful before a team knows which model class they want to instantiate.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    signals = {
        "has_interactions": has_interactions,
        "has_outcomes": has_outcomes,
        "has_concepts": has_concepts,
        "has_difficulty": has_difficulty,
        "has_prerequisites": has_prerequisites,
        "needs_live_adaptation": needs_live_adaptation,
        "needs_safe_rollout": needs_safe_rollout,
        "has_new_users": has_new_users,
        "is_regulated": is_regulated,
        "has_item_features": has_item_features,
        "has_expertise_curve": has_expertise_curve,
        "has_fresh_content": has_fresh_content,
    }

    fits = [
        _score_scenario(
            scenario,
            signals=signals,
            use_case=use_case,
            has_learning_metadata=has_concepts or has_difficulty or has_prerequisites,
        )
        for scenario in _SCENARIOS
    ]
    if all(fit.score <= 0.0 for fit in fits):
        default = _SCENARIOS_BY_ID["adaptive_learning_next_item"]
        return [
            ScenarioFit(
                scenario=default,
                score=0.1,
                reasons=("No signals were supplied; start from the primary adaptive-learning recipe.",),
            )
        ]

    return sorted(
        fits,
        key=lambda fit: (-fit.score, _SCENARIO_ORDER[fit.scenario.id]),
    )[:top_k]


def _score_scenario(
    scenario: ScenarioRecipe,
    *,
    signals: dict[str, bool],
    use_case: str | None,
    has_learning_metadata: bool,
) -> ScenarioFit:
    score = 0.0
    reasons: list[str] = []

    for signal_name, weight, reason in _SIGNAL_RULES[scenario.id]:
        if signals[signal_name]:
            score += weight
            reasons.append(reason)

    if use_case:
        keyword_score, keyword_reasons = _keyword_score(scenario.id, use_case)
        score += keyword_score
        reasons.extend(keyword_reasons)

    return ScenarioFit(scenario=scenario, score=round(score, 3), reasons=tuple(reasons))


def _keyword_score(scenario_id: str, use_case: str) -> tuple[float, list[str]]:
    normalized = _normalize_use_case(use_case)
    score = 0.0
    reasons: list[str] = []
    for keyword, weight, reason in _KEYWORDS.get(scenario_id, ()):
        if keyword in normalized:
            score += weight
            reasons.append(reason)
    return score, reasons


def _normalize_use_case(use_case: Any) -> str:
    return " ".join(str(use_case).casefold().replace("-", " ").split())
