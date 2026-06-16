"""Backward-compatible keyword-alias coverage for the flagship adaptive APIs.

``AdaptiveRanker`` uses ``kt_backbone``/``learner_id`` while
``AdaptiveLearningEngine`` uses ``tracer_model``/``user_id``. To make snippets
portable, each side additively accepts the other's spelling as an alias. These
tests pin that behavior, including the "pass only one" and "unknown kwarg"
guards.
"""
from __future__ import annotations

import pandas as pd
import pytest

from orchid_ranker import (
    AdaptiveLearningEngine,
    AdaptiveRanker,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [101, 201, 101, 202, 101, 201],
            "correct": [1, 0, 1, 1, 0, 1],
            "concept": ["ns", "fr", "ns", "fr", "ns", "fr"],
            "difficulty": [0.20, 0.45, 0.20, 0.50, 0.20, 0.45],
        }
    )


# --- Config aliases (no torch required) -----------------------------------


def test_learning_config_accepts_kt_backbone_alias() -> None:
    eng = AdaptiveLearningEngine(kt_backbone="sakt", epochs=1, d_model=8, n_heads=2)
    assert eng.config.tracer_model == "sakt"


def test_ranker_config_accepts_tracer_model_alias() -> None:
    ranker = AdaptiveRanker(tracer_model="saint+", epochs=1, d_model=8, n_heads=2)
    assert ranker.config.kt_backbone == "saint+"


def test_config_alias_and_primary_together_raise() -> None:
    with pytest.raises(TypeError, match="pass only one"):
        AdaptiveRanker(kt_backbone="akt", tracer_model="sakt")
    with pytest.raises(TypeError, match="pass only one"):
        AdaptiveLearningEngine(tracer_model="akt", kt_backbone="sakt")


def test_unknown_config_kwarg_still_rejected() -> None:
    with pytest.raises(TypeError, match="Unknown AdaptiveRankerConfig fields"):
        AdaptiveRanker(bogus=1)
    with pytest.raises(TypeError, match="Unknown AdaptiveLearningConfig fields"):
        AdaptiveLearningEngine(bogus=1)


# --- Runtime method aliases (require torch via fit) ------------------------


def _fit_engine() -> AdaptiveLearningEngine:
    return AdaptiveLearningEngine(
        tracer_model="akt", policy="auto", epochs=1, d_model=8, n_heads=2, batch_size=4, device="cpu"
    ).fit(
        _events(),
        correct_col="correct",
        concept_col="concept",
        item_difficulty_col="difficulty",
        prerequisite_by_concept={"fr": ["ns"]},
    )


def _fit_ranker() -> AdaptiveRanker:
    events = _events().rename(columns={"user_id": "learner_id", "concept": "concept_id"}).assign(ts=range(6))
    return AdaptiveRanker(
        kt_backbone="akt", policy="auto", epochs=1, d_model=8, n_heads=2, batch_size=4, device="cpu"
    ).fit_kt(
        events,
        correct_col="correct",
        concept_col="concept_id",
        item_difficulty_col="difficulty",
    )


def test_engine_rank_accepts_learner_id_alias() -> None:
    pytest.importorskip("torch")
    eng = _fit_engine()
    canonical = eng.rank(user_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
    via_alias = eng.rank(learner_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
    positional = eng.rank(1, [101, 201, 202], top_k=2)
    assert [r.item_id for r in canonical] == [r.item_id for r in via_alias]
    assert [r.item_id for r in canonical] == [r.item_id for r in positional]


def test_engine_recommend_alias_method_accepts_learner_id() -> None:
    pytest.importorskip("torch")
    eng = _fit_engine()
    canonical = eng.rank(user_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
    via_alias = eng.recommend(learner_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
    assert [r.item_id for r in canonical] == [r.item_id for r in via_alias]


def test_engine_observe_accepts_both_spellings() -> None:
    pytest.importorskip("torch")
    eng = _fit_engine()
    eng.observe(user_id=1, item_id=101, correct=True)
    eng.observe(learner_id=1, item_id=101, correct=True)


def test_engine_rank_both_ids_raise() -> None:
    pytest.importorskip("torch")
    eng = _fit_engine()
    with pytest.raises(TypeError, match="pass only one"):
        eng.rank(user_id=1, learner_id=1, candidate_item_ids=[101])


def test_ranker_recommend_accepts_user_id_alias() -> None:
    pytest.importorskip("torch")
    ranker = _fit_ranker()
    canonical = ranker.recommend(learner_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
    via_alias = ranker.recommend(user_id=1, candidate_item_ids=[101, 201, 202], top_k=2)
    positional = ranker.recommend(1, [101, 201, 202], top_k=2)
    assert [r.item_id for r in canonical] == [r.item_id for r in via_alias]
    assert [r.item_id for r in canonical] == [r.item_id for r in positional]


def test_ranker_observe_accepts_user_id_alias() -> None:
    pytest.importorskip("torch")
    ranker = _fit_ranker()
    recs = ranker.recommend(1, [101, 201, 202], top_k=2)
    item_id = recs[0].item_id
    ranker.observe(learner_id=1, item_id=item_id, correct=1, ts=99, concept_id=None)
    ranker.observe(user_id=1, item_id=item_id, correct=1, ts=100, concept_id=None)


def test_ranker_observe_both_ids_raise() -> None:
    pytest.importorskip("torch")
    ranker = _fit_ranker()
    recs = ranker.recommend(1, [101, 201, 202], top_k=2)
    with pytest.raises(TypeError, match="pass only one"):
        ranker.observe(learner_id=1, user_id=1, item_id=recs[0].item_id, correct=1, ts=101, concept_id=None)
