#!/usr/bin/env python3
"""Small use-case cookbook for Orchid Ranker.

Run with:
    python examples/adaptive_learning_use_cases.py

These examples use the torch-free core APIs so they work with the base
``pip install orchid-ranker`` package. Use ``AdaptiveLearningEngine`` when you
also want PyTorch-backed knowledge tracing from learner histories.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchid_ranker import DependencyGraph, FSRSReviewState, FSRSScheduler, ProgressionRecommender
from orchid_ranker.ope import compare_logged_policies, deterministic_policy_probabilities, evaluate_rollout_gate
from orchid_ranker.progression_reward import expected_progression_reward


def compliance_training_path() -> dict[str, Any]:
    """Choose the next compliance modules from completed prerequisites."""
    graph = DependencyGraph(
        [
            ("policy-basics", "secure-passwords"),
            ("policy-basics", "data-handling"),
            ("secure-passwords", "phishing-response"),
            ("data-handling", "incident-reporting"),
            ("phishing-response", "incident-reporting"),
        ]
    )
    difficulty = {
        "policy-basics": 0.10,
        "secure-passwords": 0.25,
        "data-handling": 0.35,
        "phishing-response": 0.45,
        "incident-reporting": 0.70,
    }
    completed = {"policy-basics", "secure-passwords"}
    recommender = ProgressionRecommender(graph, difficulty_map=difficulty)

    return {
        "learner": "analyst-17",
        "completed": sorted(completed),
        "next_modules": recommender.recommend(completed, n=3),
        "why": "Only modules with satisfied prerequisites are eligible; easier eligible modules rank first.",
    }


def language_review_queue(now: datetime | None = None) -> dict[str, Any]:
    """Rank vocabulary cards by forgetting urgency."""
    active_now = now or datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
    scheduler = FSRSScheduler(request_retention=0.90)
    states = {
        "bonjour": FSRSReviewState(
            stability=2.0,
            difficulty=3.5,
            due_at=active_now - timedelta(hours=6),
            last_review_at=active_now - timedelta(days=4),
            repetitions=3,
        ),
        "aeroport": FSRSReviewState(
            stability=5.0,
            difficulty=5.5,
            due_at=active_now + timedelta(days=1),
            last_review_at=active_now - timedelta(days=2),
            repetitions=2,
        ),
        "se souvenir": FSRSReviewState(
            stability=1.2,
            difficulty=7.0,
            due_at=active_now - timedelta(days=1),
            last_review_at=active_now - timedelta(days=6),
            repetitions=1,
            lapses=1,
        ),
    }

    recommendations = scheduler.recommend_reviews(states, now=active_now, top_k=3)
    reviewed = scheduler.review(states[recommendations[0].item_id], grade=3, now=active_now)
    return {
        "learner": "french-a2-learner",
        "review_order": [
            {
                "item_id": rec.item_id,
                "urgency": round(rec.urgency, 3),
                "due": rec.due,
                "retrievability": round(rec.retrievability, 3),
            }
            for rec in recommendations
        ],
        "after_review": reviewed.to_dict(),
    }


def rehab_exercise_progression() -> dict[str, Any]:
    """Score physical-therapy exercises by progression value, not raw success."""
    candidates = {
        "range-of-motion-review": {"p_correct": 0.95, "difficulty": 0.20, "competence": 0.55},
        "supported-step-up": {"p_correct": 0.72, "difficulty": 0.64, "competence": 0.55},
        "unassisted-balance-hop": {"p_correct": 0.26, "difficulty": 0.90, "competence": 0.55},
    }
    scored = []
    for exercise_id, features in candidates.items():
        reward = expected_progression_reward(**features)
        scored.append(
            {
                "exercise_id": exercise_id,
                "expected_reward": round(reward.expected_reward, 3),
                "p_correct": features["p_correct"],
                "difficulty": features["difficulty"],
                "stretch_fit": round(reward.stretch_fit, 3),
            }
        )
    scored.sort(key=lambda row: row["expected_reward"], reverse=True)
    return {
        "learner": "post-op-knee-week-4",
        "recommended_exercise": scored[0]["exercise_id"],
        "ranked_exercises": scored,
        "why": "The best item is challenging enough for growth without being too hard for the current competence estimate.",
    }


def onboarding_rollout_gate() -> dict[str, Any]:
    """Check whether a new onboarding policy is safe enough to roll out."""
    rows = []
    actions = ["review-basics", "ship-first-task"]
    for learner_id in range(80):
        target_action = "review-basics" if learner_id % 4 == 0 else "ship-first-task"
        for action in actions:
            rows.append(
                {
                    "learner_id": learner_id,
                    "action": action,
                    "progression_reward": 0.90 if action == target_action else 0.10,
                    "logging_probability": 0.50,
                    "target_action": target_action,
                    "baseline_action": "review-basics",
                }
            )
    events = pd.DataFrame(rows)
    events["target_probability"] = deterministic_policy_probabilities(
        events["action"].tolist(),
        events["target_action"].tolist(),
    )
    events["baseline_probability"] = deterministic_policy_probabilities(
        events["action"].tolist(),
        events["baseline_action"].tolist(),
    )

    report = compare_logged_policies(
        events,
        reward_col="progression_reward",
        propensity_col="logging_probability",
        target_probability_col="target_probability",
        baseline_probability_col="baseline_probability",
    )
    gate = evaluate_rollout_gate(report, min_effect=0.05, min_coverage=0.20, min_ess_fraction=0.20)
    return {
        "logged_learners": int(events["learner_id"].nunique()),
        "estimated_uplift": round(report.uplift, 3),
        "confidence_interval": [round(report.ci_low, 3), round(report.ci_high, 3)],
        "allowed": gate.allowed,
        "reasons": list(gate.reasons),
    }


def run_all() -> dict[str, Any]:
    """Return every cookbook result as JSON-serializable data."""
    return {
        "compliance_training": compliance_training_path(),
        "language_review": language_review_queue(),
        "rehab_progression": rehab_exercise_progression(),
        "onboarding_rollout_gate": onboarding_rollout_gate(),
    }


def main() -> None:
    print(json.dumps(run_all(), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
