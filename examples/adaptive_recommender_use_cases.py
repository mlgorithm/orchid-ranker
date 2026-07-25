#!/usr/bin/env python3
"""Three copyable reference pilots using Orchid's one adaptive loop.

Run with:
    python examples/adaptive_recommender_use_cases.py

To export validation-ready completed decision logs:
    python examples/adaptive_recommender_use_cases.py --output-dir artifacts/reference-pilots
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchid_ranker import AdaptiveRanker


@dataclass(frozen=True)
class PilotRun:
    """One simulated pilot summary and its completed decision records."""

    summary: dict[str, Any]
    completed_decisions: pd.DataFrame


def _history(item_ids: Sequence[str]) -> pd.DataFrame:
    """Create a small stand-in for completed historical product events.

    Replace this function with a chronological query of completed events in a
    real pilot. The shape is always user, item, binary outcome, and timestamp.
    """
    outcome_patterns = ([1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0])
    if len(item_ids) != len(outcome_patterns[0]):
        raise ValueError("reference-pilot histories require exactly four items")
    rows: list[dict[str, Any]] = []
    for user_index, outcomes in enumerate(outcome_patterns):
        for timestamp, (item_id, outcome) in enumerate(zip(item_ids, outcomes), start=1):
            rows.append(
                {
                    "user_id": f"user-{user_index}",
                    "item_id": item_id,
                    "outcome": outcome,
                    "timestamp": user_index * 100 + timestamp,
                }
            )
    return pd.DataFrame(rows)


def _demo_ranker() -> AdaptiveRanker:
    """Use a small deterministic training budget for these simulated pilots.

    The project integration remains the same in production; applications can
    start with ``AdaptiveRanker()`` and their own historical volume.
    """
    return AdaptiveRanker(epochs=1, d_model=8, n_heads=2, batch_size=8, device="cpu")


def _run_pilot(
    *,
    name: str,
    history: pd.DataFrame,
    user_id: str,
    eligible_items: Sequence[str],
    outcome_name: str,
    observed_outcome: int,
    guardrail: str,
) -> PilotRun:
    """Fit, serve, log, observe, and return one validation-ready decision log."""
    ranker = _demo_ranker().fit(history)
    ranked, decision = ranker.recommend_and_log(
        user_id=user_id,
        candidate_item_ids=eligible_items,
        timestamp=int(history["timestamp"].max()) + 1,
        top_k=min(3, len(eligible_items)),
        exploration=0.05,
        min_item_support=0,
        min_outcome_probability=0.0,
        max_outcome_probability=1.0,
        policy_version=f"{name}-v1",
    )
    linked = ranker.observe_decision(
        decision.decision_id,
        outcome=observed_outcome,
        timestamp=decision.timestamp + 1,
    )
    completed = ranker.decision_log_frame(completed_only=True)
    summary = {
        "outcome": outcome_name,
        "guardrail": guardrail,
        "eligible_items": list(eligible_items),
        "ranked_items": [recommendation.item_id for recommendation in ranked],
        "served_item": decision.chosen_item_id,
        "observed_outcome": linked.outcome,
        "policy_version": decision.policy_version,
        "completed_decisions": int(len(completed)),
    }
    return PilotRun(summary=summary, completed_decisions=completed)


def onboarding() -> PilotRun:
    """B2B onboarding: recommend the next eligible activation step."""
    steps = ["create-profile", "connect-data", "create-project", "invite-teammate"]
    completed_steps = {"create-profile"}
    eligible_steps = [step for step in steps if step not in completed_steps]
    return _run_pilot(
        name="onboarding",
        history=_history(steps),
        user_id="user-0",
        eligible_items=eligible_steps,
        outcome_name="recommended step completed within 24 hours",
        observed_outcome=1,
        guardrail="Do not recommend unavailable steps or send more than one onboarding prompt per day.",
    )


def compliance_training() -> PilotRun:
    """Training: recommend a passable module without bypassing prerequisites."""
    modules = ["privacy-basics", "phishing", "incident-reporting", "secure-handling"]
    prerequisites = {
        "privacy-basics": set(),
        "phishing": {"privacy-basics"},
        "incident-reporting": {"privacy-basics"},
        "secure-handling": {"privacy-basics", "phishing"},
    }
    passed_modules = {"privacy-basics"}
    eligible_modules = [
        module
        for module in modules
        if module not in passed_modules and prerequisites[module].issubset(passed_modules)
    ]
    return _run_pilot(
        name="training",
        history=_history(modules),
        user_id="user-1",
        eligible_items=eligible_modules,
        outcome_name="recommended module passed on the next attempt",
        observed_outcome=1,
        guardrail="Certification, role, and legal prerequisites remain application rules, not ranking scores.",
    )


def content_discovery() -> PilotRun:
    """Content: recommend an eligible article using meaningful completion feedback."""
    articles = ["setup-guide", "team-playbook", "reporting-guide", "automation-guide"]
    catalog = {
        "setup-guide": {"locale": "en", "plan": "all"},
        "team-playbook": {"locale": "en", "plan": "all"},
        "reporting-guide": {"locale": "en", "plan": "pro"},
        "automation-guide": {"locale": "en", "plan": "pro"},
    }
    user_locale = "en"
    user_plan = "pro"
    seen_articles = {"setup-guide"}
    eligible_articles = [
        article
        for article in articles
        if article not in seen_articles
        and catalog[article]["locale"] == user_locale
        and catalog[article]["plan"] in {"all", user_plan}
    ]
    return _run_pilot(
        name="content",
        history=_history(articles),
        user_id="user-2",
        eligible_items=eligible_articles,
        outcome_name="recommended article read to completion within 24 hours",
        observed_outcome=0,
        guardrail="Use completion or a downstream task outcome, not a raw click, when that is the real objective.",
    )


def run_pilots() -> dict[str, PilotRun]:
    """Run each reference pilot once."""
    return {
        "onboarding": onboarding(),
        "compliance_training": compliance_training(),
        "content_discovery": content_discovery(),
    }


def run_all() -> dict[str, dict[str, Any]]:
    """Return JSON-serializable summaries for all reference pilots."""
    return {name: run.summary for name, run in run_pilots().items()}


def export_completed_logs(runs: dict[str, PilotRun], output_dir: Path) -> None:
    """Write one validation-ready JSON Lines file per simulated pilot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        run.completed_decisions.to_json(output_dir / f"{name}-completed-decisions.jsonl", orient="records", lines=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Orchid reference-pilot integrations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for validation-ready completed decision logs.",
    )
    args = parser.parse_args(argv)
    runs = run_pilots()
    if args.output_dir is not None:
        export_completed_logs(runs, args.output_dir)
    print(json.dumps({name: run.summary for name, run in runs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
