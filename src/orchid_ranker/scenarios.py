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
        name="New-user cold start",
        summary="Blend content and popularity until a user has enough history.",
        use_when="New users need reasonable recommendations before collaborative signals exist.",
        signals=(
            "item features",
            "optional popularity prior",
            "seed items",
            "early interactions",
        ),
        algorithms=(
            "content-similarity bridge",
            "popularity prior",
            "warmth-aware blend",
        ),
        entrypoints=(
            "ColdStartBridge",
            "ColdStartConfig",
        ),
        docs_anchor="new-user-cold-start",
        tags=("cold-start", "catalog"),
    ),
    ScenarioRecipe(
        id="batch_catalog_recommendation",
        name="Batch catalog recommendation",
        summary="Fit a standard recommender from historical interactions.",
        use_when="The product needs offline recommendations or a simple service endpoint.",
        signals=(
            "user_id",
            "item_id",
            "optional rating or binary label",
        ),
        algorithms=(
            "ALS or explicit matrix factorization",
            "NeuralMF when PyTorch is available",
            "baseline ranking",
        ),
        entrypoints=(
            "OrchidRecommender.from_interactions",
            "OrchidRecommender.recommend",
            "OrchidRecommender.baseline_rank",
        ),
        docs_anchor="batch-catalog-recommendations",
        tags=("batch", "catalog"),
    ),
    ScenarioRecipe(
        id="generic_streaming_recommender",
        name="Generic streaming recommender",
        summary="Adapt quickly from live feedback when learning metadata is not available.",
        use_when=(
            "The system needs online updates but does not have concepts, difficulty, "
            "or prerequisites."
        ),
        signals=(
            "historical interactions",
            "live interaction events",
            "candidate item set",
        ),
        algorithms=(
            "streaming NeuralMF",
            "rolling progression monitor when category labels exist",
            "batch baseline fallback",
        ),
        entrypoints=(
            "OrchidRecommender.as_streaming",
            "StreamingRecommender.rank",
            "StreamingRecommender.observe",
        ),
        docs_anchor="generic-streaming-recommender",
        tags=("streaming", "fallback"),
    ),
    ScenarioRecipe(
        id="expertise_commerce",
        name="Expertise-driven commerce",
        summary="Recommend products along a user's taste or expertise curve.",
        use_when="Users progress from beginner to advanced preferences inside a product category.",
        signals=(
            "purchases",
            "returns",
            "ratings",
            "product category",
            "sophistication score",
        ),
        algorithms=(
            "taste progression",
            "stretch-fit scoring",
            "momentum and exploration",
        ),
        entrypoints=(
            "SophisticationMapper",
            "TasteProgressionRanker",
        ),
        docs_anchor="expertise-driven-commerce",
        tags=("commerce", "expertise"),
    ),
    ScenarioRecipe(
        id="curated_publication_feed",
        name="Curated publication feed",
        summary="Rank content by relevance, freshness, topic growth, and difficulty.",
        use_when="Readers should build topic competence instead of only maximizing clicks.",
        signals=(
            "topic",
            "difficulty",
            "publication time",
            "meaningful engagement",
        ),
        algorithms=(
            "freshness scoring",
            "topic competence",
            "stretch-fit ranking",
            "topic diversity",
        ),
        entrypoints=(
            "FeedItem",
            "FeedRanker",
            "FreshnessScorer",
        ),
        docs_anchor="curated-publication-feed",
        tags=("feed", "content", "progression"),
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
        ("has_new_users", 3.0, "New users need recommendations before history exists."),
        ("has_item_features", 1.5, "Item features can power content similarity."),
        ("has_interactions", 0.5, "Existing interactions provide a baseline once users warm up."),
    ),
    "batch_catalog_recommendation": (
        ("has_interactions", 2.0, "Historical interactions are enough for a batch recommender."),
        ("has_item_features", 0.5, "Item features can help later cold-start extensions."),
    ),
    "generic_streaming_recommender": (
        ("needs_live_adaptation", 1.5, "Live feedback should update the recommender."),
        ("has_interactions", 1.0, "Historical interactions can initialize the streaming model."),
    ),
    "expertise_commerce": (
        ("has_expertise_curve", 3.0, "The category has a real beginner-to-expert trajectory."),
        ("has_interactions", 1.0, "Purchases or ratings can initialize taste estimates."),
        ("has_item_features", 0.5, "Product attributes can support sophistication mapping."),
    ),
    "curated_publication_feed": (
        ("has_fresh_content", 3.0, "Freshness is central to the ranking problem."),
        ("has_concepts", 1.0, "Topics support competence-aware feed ranking."),
        ("has_difficulty", 1.0, "Difficulty enables stretch-fit reading progression."),
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
        ("new user", 1.0, "New users need cold-start behavior."),
        ("onboarding", 0.5, "Onboarding often starts with sparse history."),
    ),
    "batch_catalog_recommendation": (
        ("batch", 1.0, "Batch workflows map to standard recommendation."),
        ("catalog", 1.0, "Catalog ranking maps to standard recommendation."),
        ("rating", 0.5, "Ratings are direct recommender signals."),
    ),
    "generic_streaming_recommender": (
        ("streaming", 1.0, "Streaming maps to online updates."),
        ("online", 0.5, "Online feedback can use the streaming path."),
    ),
    "expertise_commerce": (
        ("commerce", 1.0, "Commerce with expertise progression maps to this ranker."),
        ("wine", 1.0, "Wine is a typical expertise-commerce category."),
        ("coffee", 1.0, "Coffee is a typical expertise-commerce category."),
        ("fashion", 0.5, "Fashion can have taste progression."),
        ("taste", 1.0, "Taste evolution maps to expertise commerce."),
    ),
    "curated_publication_feed": (
        ("feed", 1.0, "Feed language maps to curated publication ranking."),
        ("newsletter", 1.0, "Newsletters are curated publication feeds."),
        ("article", 0.5, "Articles can be ranked by topic progression."),
        ("publication", 1.0, "Publication ranking maps to this scenario."),
        ("reading", 0.5, "Reading progression fits curated feeds."),
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

    if (
        scenario.id == "generic_streaming_recommender"
        and signals["needs_live_adaptation"]
        and not has_learning_metadata
    ):
        score += 2.0
        reasons.append("Live adaptation is needed, but learning metadata is missing.")

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
