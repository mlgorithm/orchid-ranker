from __future__ import annotations

import pandas as pd
import pytest

from orchid_ranker import IRTAdaptiveSelector, IRTItem


def test_irt_probability_tracks_item_difficulty():
    selector = IRTAdaptiveSelector(initial_theta=0.0).fit_items(
        [
            IRTItem("easy", difficulty=-2.0),
            IRTItem("hard", difficulty=2.0),
        ]
    )

    assert selector.probability("easy") > selector.probability("hard")


def test_irt_observe_correct_response_increases_ability():
    selector = IRTAdaptiveSelector(initial_theta=0.0, learning_rate=0.5).fit_items(
        [IRTItem("middle", difficulty=0.0)]
    )

    before = selector.theta
    after = selector.observe("middle", correct=True)

    assert after > before
    assert selector.history_[-1][0] == "middle"


def test_irt_recommend_prefers_informative_items_near_ability():
    selector = IRTAdaptiveSelector(initial_theta=0.0).fit_items(
        pd.DataFrame(
            {
                "item_id": ["too_easy", "near", "too_hard"],
                "difficulty": [-4.0, 0.0, 4.0],
                "concept_id": ["warmup", "algebra", "calculus"],
            }
        ),
        concept_col="concept_id",
    )

    ranked = selector.recommend(top_k=3)

    assert ranked[0].item_id == "near"
    assert ranked[0].information > ranked[-1].information


def test_irt_recommend_respects_prerequisite_constraints():
    selector = IRTAdaptiveSelector(initial_theta=0.0).fit_items(
        pd.DataFrame(
            {
                "item_id": ["foundation", "advanced"],
                "difficulty": [0.0, 0.1],
                "concept_id": ["basics", "advanced"],
            }
        ),
        concept_col="concept_id",
    )

    ranked = selector.recommend(
        top_k=2,
        prerequisite_by_concept={"advanced": ["basics"]},
        mastered_concepts=[],
    )

    assert [rec.item_id for rec in ranked] == ["foundation"]


def test_irt_rejects_invalid_item_parameters():
    with pytest.raises(ValueError, match="discrimination"):
        IRTAdaptiveSelector().fit_items([IRTItem("bad", difficulty=0.0, discrimination=0.0)])
