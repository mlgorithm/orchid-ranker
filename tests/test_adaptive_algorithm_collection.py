from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from orchid_ranker import (
    AdaptiveRanker,
    AFMTracer,
    FSRSReviewState,
    FSRSScheduler,
    IsotonicProbabilityCalibrator,
    PersonalizedLinUCB,
    PFATracer,
    TabularFQE,
    TemperatureScaler,
    expected_calibration_error,
    fit_bkt_em,
)


def _events() -> pd.DataFrame:
    rows = []
    sequences = {
        "a": [("i1", "basics", 1), ("i2", "fractions", 0), ("i3", "fractions", 1), ("i4", "ratios", 0)],
        "b": [("i1", "basics", 1), ("i2", "fractions", 1), ("i3", "fractions", 1), ("i4", "ratios", 0)],
        "c": [("i1", "basics", 0), ("i2", "fractions", 0), ("i3", "fractions", 1), ("i4", "ratios", 1)],
    }
    for learner_id, outcomes in sequences.items():
        for ts, (item_id, concept_id, correct) in enumerate(outcomes):
            rows.append(
                {
                    "learner_id": learner_id,
                    "item_id": item_id,
                    "concept_id": concept_id,
                    "correct": correct,
                    "ts": ts,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("backbone", ["dkt", "dkvmn"])
def test_adaptive_ranker_supports_curated_kt_backbones(backbone: str):
    ranker = AdaptiveRanker(
        kt_backbone=backbone,
        epochs=1,
        d_model=8,
        n_heads=1,
        batch_size=4,
        device="cpu",
    ).fit_kt(_events())

    ranked = ranker.recommend("a", ["i2", "i3"], top_k=2)

    assert len(ranked) == 2
    assert all(0.0 <= rec.p_correct <= 1.0 for rec in ranked)


def test_pfa_and_afm_are_interpretable_tracers():
    events = _events().rename(columns={"learner_id": "user_id", "concept_id": "concept"})
    pfa = PFATracer().fit(events, timestamp_col="ts", concept_col="concept")
    afm = AFMTracer().fit(events, timestamp_col="ts", concept_col="concept")

    pfa_before = pfa.predict_correct("a", "i2")
    length = pfa.observe("a", "i2", 1)
    pfa_after = pfa.predict_correct("a", "i2")

    assert pfa.report_.n_concepts == 3
    assert afm.report_.n_examples == len(events)
    assert length >= 1
    assert 0.0 <= pfa_before <= 1.0
    assert 0.0 <= pfa_after <= 1.0


def test_bkt_em_fits_valid_parameters_and_tracer():
    report = fit_bkt_em(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1],
        ],
        max_iter=10,
    )
    tracer = report.to_tracer()

    assert report.n_sequences == 3
    assert 0.0 < report.p_init < 1.0
    assert 0.0 < report.p_transit < 1.0
    assert 0.0 < report.p_slip < 1.0
    assert 0.0 < report.p_guess < 1.0
    assert 0.0 <= tracer.update(True) <= 1.0


def test_probability_calibrators_report_ece_and_brier():
    labels = [0, 0, 1, 1]
    probs = [0.2, 0.4, 0.6, 0.8]

    temp = TemperatureScaler().fit(probs, labels)
    iso = IsotonicProbabilityCalibrator().fit(probs, labels)

    assert temp.temperature_ is not None
    assert temp.report_.n == 4
    assert iso.predict_proba([0.3, 0.7]).shape == (2,)
    assert expected_calibration_error(probs, labels) >= 0.0


def test_fsrs_scheduler_ranks_due_reviews_by_forgetting_risk():
    now = datetime(2026, 5, 13, tzinfo=timezone.utc)
    scheduler = FSRSScheduler()
    state = scheduler.review(None, grade=3, now=now - timedelta(days=5))
    new_state = scheduler.review(state, grade=4, now=now)
    recs = scheduler.recommend_reviews(
        {
            "due": FSRSReviewState(stability=1.0, due_at=now - timedelta(days=1), last_review_at=now - timedelta(days=3)),
            "fresh": new_state,
        },
        now=now,
        top_k=2,
    )

    assert new_state.stability > 0.0
    assert recs[0].item_id == "due"
    assert recs[0].due is True


def test_tabular_fqe_evaluates_fixed_policy_values():
    transitions = pd.DataFrame(
        [
            {
                "context_hash": "s0",
                "chosen_item_id": "a",
                "reward": 1.0,
                "next_context_hash": "s1",
                "target_action_id": "b",
                "done": False,
            },
            {
                "context_hash": "s1",
                "chosen_item_id": "b",
                "reward": 2.0,
                "next_context_hash": "terminal",
                "target_action_id": "b",
                "done": True,
            },
        ]
    )
    fqe = TabularFQE(gamma=0.9, epochs=80, learning_rate=0.2).fit(transitions)

    assert fqe.report_.n_transitions == 2
    assert fqe.score("s0", "a") > 1.0
    assert fqe.value(["s0"], ["a"]) == pytest.approx(fqe.score("s0", "a"))


def test_personalized_linucb_changes_scores_by_user_context():
    bandit = PersonalizedLinUCB(alpha=0.0)
    bandit.update([1.0, 0.0], [1.0, 0.0], reward=1.0)
    bandit.update([1.0, 0.0], [0.0, 1.0], reward=0.0)
    for _ in range(3):
        bandit.update([0.0, 1.0], [0.0, 1.0], reward=1.0)
        bandit.update([0.0, 1.0], [1.0, 0.0], reward=0.0)

    items = {"x": [1.0, 0.0], "y": [0.0, 1.0]}
    user_a = bandit.score([1.0, 0.0], items)
    user_b = bandit.score([0.0, 1.0], items)

    assert user_a["x"].score > user_a["y"].score
    assert user_b["y"].score > user_b["x"].score
