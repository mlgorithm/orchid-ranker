from __future__ import annotations

import pandas as pd
import pytest

from orchid_ranker import (
    AdaptiveRanker,
    CQLDiscretePolicy,
    ExactEmbeddingIndex,
    LearnerEvent,
    LoggedDecision,
    SemanticItemEncoder,
    SketchCandidateGenerator,
    logged_decisions_to_frame,
    stable_context_hash,
    validate_logged_decisions,
)


def _events() -> pd.DataFrame:
    rows = []
    sequences = {
        "learner-a": [("i1", "basics", 1), ("i2", "fractions", 0), ("i3", "fractions", 1), ("i4", "ratios", 0)],
        "learner-b": [("i1", "basics", 1), ("i2", "fractions", 1), ("i3", "fractions", 1), ("i4", "ratios", 0)],
        "learner-c": [("i1", "basics", 0), ("i2", "fractions", 0), ("i3", "fractions", 1), ("i4", "ratios", 1)],
    }
    difficulty = {"i1": 0.2, "i2": 0.45, "i3": 0.5, "i4": 0.7}
    for learner_id, outcomes in sequences.items():
        for ts, (item_id, concept_id, correct) in enumerate(outcomes):
            rows.append(
                {
                    "learner_id": learner_id,
                    "item_id": item_id,
                    "concept_id": concept_id,
                    "difficulty": difficulty[item_id],
                    "correct": correct,
                    "ts": ts,
                }
            )
    return pd.DataFrame(rows)


def _decisions() -> pd.DataFrame:
    rows = []
    ctx = stable_context_hash("learner-a", "fractions")
    for idx in range(16):
        chosen = "i3" if idx % 2 == 0 else "i2"
        rows.append(
            LoggedDecision(
                learner_id="learner-a",
                ts=idx,
                candidate_item_ids=["i2", "i3"],
                chosen_item_id=chosen,
                propensity=0.5,
                policy_name="logging",
                policy_version="v1",
                scores=[0.3, 0.7],
                context_hash=ctx,
                reward=1.0 if chosen == "i3" else 0.0,
            ).to_dict()
        )
    return pd.DataFrame(rows)


def test_logged_decision_schema_requires_chosen_action_in_candidates():
    frame = _decisions()
    frame.loc[0, "chosen_item_id"] = "missing"

    with pytest.raises(ValueError, match="chosen_item_id"):
        validate_logged_decisions(frame, reward_col="reward")


def test_logged_decision_frame_allows_unrewarded_serving_logs():
    frame = logged_decisions_to_frame([
        LoggedDecision(
            learner_id="learner-a",
            ts=1,
            candidate_item_ids=["i2", "i3"],
            chosen_item_id="i2",
            propensity=0.5,
            policy_name="shadow",
            policy_version="v1",
            scores=[0.4, 0.6],
            context_hash=stable_context_hash("learner-a", "fractions"),
        )
    ])

    assert frame.loc[0, "chosen_item_id"] == "i2"


def test_cql_discrete_policy_learns_conservative_best_action():
    policy = CQLDiscretePolicy(epochs=80, learning_rate=0.05, conservative_weight=0.2).fit(_decisions())
    ctx = stable_context_hash("learner-a", "fractions")

    ranked = policy.recommend(ctx, ["i2", "i3"], top_k=2)

    assert ranked[0] == "i3"
    assert policy.report_.n_events == len(_decisions())
    assert policy.score(ctx, ["i2", "i3"])["i3"] > policy.score(ctx, ["i2", "i3"])["i2"]


def test_sketch_candidate_generator_combines_heavy_hitters_and_vector_search():
    index = ExactEmbeddingIndex()
    index.add("i3", [1.0, 0.0])
    index.add("i4", [0.8, 0.2])
    generator = SketchCandidateGenerator(ann_index=index, top_m=4)
    for _ in range(5):
        generator.update("other-learner", "fractions", "i2", correct=1)
    generator.update("other-learner", "fractions", "i3", correct=1)
    generator.mark_seen("target-learner", "i2")

    candidates = generator.candidates("target-learner", "fractions", item_query_vec=[1.0, 0.0], top_m=3)

    assert "i2" not in candidates
    assert candidates[0] == "i3"


def test_adaptive_ranker_facade_trains_policy_runs_ope_and_observes():
    ctx = stable_context_hash("learner-a", "fractions")
    ranker = AdaptiveRanker(
        kt_backbone="sakt",
        epochs=1,
        d_model=16,
        n_heads=2,
        batch_size=4,
        device="cpu",
        offline_policy_weight=5.0,
    ).fit_kt(_events(), item_difficulty_col="difficulty")

    policy_report = ranker.fit_policy(_decisions(), conservative_weight=0.2, epochs=80)
    ranked = ranker.recommend("learner-a", ["i2", "i3"], top_k=2, context_hash=ctx)
    ope = ranker.ope_report(_decisions())
    length = ranker.observe(LearnerEvent("learner-a", 10, ranked[0].item_id, None, 1))
    diagnostics = ranker.diagnostics()

    assert policy_report.n_events == len(_decisions())
    assert ranked[0].item_id == "i3"
    assert ope.n_events == len(_decisions())
    assert 0.0 <= ope.value <= 1.0
    assert length >= 1
    assert diagnostics["adaptive_ranker"]["offline_policy"]["n_events"] == len(_decisions())


def test_adaptive_ranker_accepts_saint_plus_and_semantic_candidates():
    catalog = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "item_text": [
                "warm up with whole numbers",
                "compare fractions with like denominators",
                "add fractions and simplify answers",
                "solve ratio tables",
            ],
            "concept": ["basics", "fractions", "fractions", "ratios"],
        }
    )
    ranker = AdaptiveRanker(
        kt_backbone="saint+",
        epochs=1,
        d_model=16,
        n_heads=2,
        batch_size=4,
        device="cpu",
    ).fit_kt(_events(), item_difficulty_col="difficulty")
    ranker.fit_semantic_items(catalog, metadata_cols=["concept"], n_features=256)

    ranked = ranker.recommend("learner-a", top_k=2, item_query_text="fraction addition practice")
    diagnostics = ranker.diagnostics()

    assert len(ranked) == 2
    assert {rec.item_id for rec in ranked}.issubset(set(catalog["item_id"]))
    assert diagnostics["adaptive_ranker"]["kt_backbone"] == "saint+"
    assert diagnostics["adaptive_ranker"]["semantic_encoder"]["n_items"] == 4


def test_semantic_item_encoder_ranks_matching_exercises():
    catalog = pd.DataFrame(
        {
            "item_id": ["fractions", "ratios", "geometry"],
            "item_text": [
                "add fractions with common denominators",
                "solve ratio and proportion word problems",
                "identify acute and obtuse angles",
            ],
        }
    )
    encoder = SemanticItemEncoder(n_features=256).fit(catalog)

    ranked = encoder.similar_items("add fractions common denominators", top_k=2)

    assert ranked[0] == "fractions"
    assert encoder.scores("ratio table", candidate_item_ids=["ratios"])["ratios"] > 0.0


def test_semantic_item_encoder_profiles_from_known_items():
    catalog = pd.DataFrame(
        {
            "item_id": ["linear", "quadratic", "biology"],
            "item_text": [
                "solve linear equations in algebra",
                "solve quadratic equations in algebra",
                "describe photosynthesis in plants",
            ],
        }
    )
    encoder = SemanticItemEncoder(n_features=256).fit(catalog)

    ranked = encoder.similar_to_items(["linear"], top_k=2)

    assert ranked[0] == "quadratic"
    assert "linear" not in ranked
