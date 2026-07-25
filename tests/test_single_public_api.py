from __future__ import annotations

import pandas as pd
import pytest

import orchid_ranker
from orchid_ranker import AdaptiveRanker


def _neutral_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "item_id": [101, 102, 201, 101, 102, 201, 101, 102, 201],
            "outcome": [1, 1, 0, 1, 0, 0, 1, 1, 1],
            "timestamp": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        }
    )


def _ranker() -> AdaptiveRanker:
    return AdaptiveRanker(
        kt_backbone="sakt",
        epochs=1,
        d_model=8,
        n_heads=2,
        batch_size=4,
        device="cpu",
        random_state=7,
    )


def test_package_star_surface_has_one_product_object() -> None:
    assert orchid_ranker.__all__ == ["AdaptiveRanker"]


def test_neutral_fit_recommend_observe_loop() -> None:
    ranker = _ranker().fit(_neutral_events())

    ranked = ranker.recommend(
        user_id="a",
        candidate_item_ids=[101, 102, 201],
        top_k=2,
    )
    length = ranker.observe(
        user_id="a",
        item_id=ranked[0].item_id,
        outcome=1,
        timestamp=4,
    )

    assert ranked
    assert 0.0 <= ranked[0].outcome_probability <= 1.0
    assert ranked[0].to_dict()["outcome_probability"] == ranked[0].outcome_probability
    assert length >= 1


def test_neutral_production_log_and_observe() -> None:
    ranker = _ranker().fit(_neutral_events())

    ranked, decision = ranker.recommend_and_log(
        user_id="a",
        candidate_item_ids=[101, 102, 201],
        timestamp=4,
        exploration=0.0,
        require_prerequisites=False,
    )
    linked = ranker.observe_decision(
        decision.decision_id,
        outcome=1,
        timestamp=5,
    )

    assert ranked[0].item_id == decision.chosen_item_id
    assert linked.outcome == 1
    assert linked.outcome_timestamp == 5


def test_fit_rejects_legacy_schema() -> None:
    legacy = _neutral_events().rename(
        columns={
            "user_id": "learner_id",
            "outcome": "correct",
            "timestamp": "ts",
        }
    )

    with pytest.raises(ValueError, match="user"):
        _ranker().fit(legacy)


def test_fit_rejects_missing_required_columns() -> None:
    events = _neutral_events().drop(columns="timestamp")

    with pytest.raises(ValueError, match="timestamp"):
        _ranker().fit(events)


@pytest.mark.parametrize("invalid_outcome", [0.5, None, "yes"])
def test_fit_requires_a_complete_binary_outcome(invalid_outcome: object) -> None:
    events = _neutral_events()
    events["outcome"] = events["outcome"].astype(object)
    events.loc[0, "outcome"] = invalid_outcome

    with pytest.raises(ValueError, match="binary 0 or 1"):
        _ranker().fit(events)
