"""
End-to-end integration tests for Orchid Ranker.

These tests validate complete workflows from raw data through
fit → evaluate → save → load → recommend → knowledge trace → curriculum.
"""

import math
import os
import tempfile

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════
# 1.  FULL RECOMMENDER PIPELINE
# ════════════════════════════════════════════════════════════════════════

class TestRecommenderPipeline:
    """Raw data → split → fit → evaluate → save → load → recommend."""

    def test_full_pipeline_popularity(self):
        from orchid_ranker import (
            OrchidRecommender,
            save_model,
            load_model,
            train_test_split,
            evaluate_on_holdout,
        )

        # 1. Generate raw data
        rng = np.random.RandomState(42)
        interactions = pd.DataFrame({
            "user_id": rng.randint(0, 20, size=500),
            "item_id": rng.randint(0, 30, size=500),
            "rating":  rng.uniform(1, 5, size=500).round(1),
        })

        # 2. Split
        train, test = train_test_split(interactions, test_size=0.2, by_user=True, random_state=42)
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) == len(interactions)

        # 3. Fit
        rec = OrchidRecommender(strategy="popularity")
        rec.fit(train)

        # 4. Evaluate
        metrics = evaluate_on_holdout(rec, test)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # 5. Recommend
        uid = train["user_id"].iloc[0]
        recs = rec.recommend(user_id=uid, top_k=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

        # 6. Save
        with tempfile.NamedTemporaryFile(suffix=".orchid", delete=False) as f:
            path = f.name
        try:
            save_model(rec, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            # 7. Load
            loaded = load_model(path)

            # 8. Loaded model produces same recommendations
            recs_loaded = loaded.recommend(user_id=uid, top_k=5)
            assert recs == recs_loaded
        finally:
            os.unlink(path)

    def test_full_pipeline_als(self):
        from orchid_ranker import (
            OrchidRecommender,
            save_model,
            load_model,
            train_test_split,
        )

        rng = np.random.RandomState(7)
        interactions = pd.DataFrame({
            "user_id": rng.randint(0, 15, size=300),
            "item_id": rng.randint(0, 25, size=300),
            "rating":  rng.uniform(1, 5, size=300).round(1),
        })

        train, test = train_test_split(interactions, test_size=0.2)
        rec = OrchidRecommender(strategy="als", epochs=3)
        rec.fit(train)

        uid = train["user_id"].iloc[0]
        recs = rec.recommend(user_id=uid, top_k=5)
        assert len(recs) <= 5

        with tempfile.NamedTemporaryFile(suffix=".orchid", delete=False) as f:
            path = f.name
        try:
            save_model(rec, path)
            loaded = load_model(path)
            recs2 = loaded.recommend(user_id=uid, top_k=5)
            assert recs == recs2
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════════════════
# 2.  CROSS-VALIDATION + TUNING PIPELINE
# ════════════════════════════════════════════════════════════════════════

class TestCVTuningPipeline:
    """Cross-validate → compare → tune → best model."""

    def test_cross_validate_and_compare(self):
        from orchid_ranker import cross_validate, compare_models

        rng = np.random.RandomState(42)
        interactions = pd.DataFrame({
            "user_id": rng.randint(0, 15, size=400),
            "item_id": rng.randint(0, 20, size=400),
            "rating":  rng.uniform(1, 5, size=400).round(1),
        })

        # Cross-validate a single strategy
        cv_results = cross_validate(interactions, strategy="popularity", k=3)
        assert isinstance(cv_results, dict)

        # Compare multiple strategies
        comparison = compare_models(interactions, strategies=["popularity"], k=2)
        assert isinstance(comparison, pd.DataFrame)

    def test_grid_search_pipeline(self):
        from orchid_ranker import GridSearchCV

        rng = np.random.RandomState(42)
        interactions = pd.DataFrame({
            "user_id": rng.randint(0, 15, size=400),
            "item_id": rng.randint(0, 20, size=400),
            "rating":  rng.uniform(1, 5, size=400).round(1),
        })

        gs = GridSearchCV(
            strategy="popularity",
            param_grid={"n_recommendations": [5, 10]},
            cv=2,
        )
        gs.fit(interactions)
        assert gs.best_params_ is not None
        assert gs.best_score_ is not None


# ════════════════════════════════════════════════════════════════════════
# 3.  KNOWLEDGE TRACING PIPELINE
# ════════════════════════════════════════════════════════════════════════

class TestKnowledgeTracingPipeline:
    """BKT → MasteryTracker → ForgettingCurve → educational metrics."""

    def test_bkt_to_mastery_to_metrics(self):
        from orchid_ranker import (
            BayesianKnowledgeTracing,
            MasteryTracker,
            ForgettingCurve,
            learning_gain,
            knowledge_coverage,
        )

        # 1. Track mastery across skills
        skills = ["algebra", "geometry", "calculus"]
        tracker = MasteryTracker(skills=skills)

        # 2. Simulate student attempts
        rng = np.random.RandomState(42)
        for _ in range(100):
            skill = rng.choice(skills)
            correct = rng.random() > 0.3
            tracker.update(skill, correct=correct)

        # 3. Check mastery
        mastered = tracker.mastered_skills()
        assert isinstance(mastered, (set, list, frozenset))

        # 4. Compute educational metrics
        coverage = knowledge_coverage(set(mastered), set(skills))
        assert 0.0 <= coverage <= 1.0

        gain = learning_gain(pre_score=0.3, post_score=0.7)
        assert gain > 0.0

        # 5. Forgetting curve
        fc = ForgettingCurve()
        fc.review()
        retention = fc.retention_at(3600.0)
        assert 0.0 <= retention <= 1.0
        assert fc.should_review(threshold=0.5) in (True, False)

    def test_bkt_numerical_stability_pipeline(self):
        from orchid_ranker import BayesianKnowledgeTracing

        bkt = BayesianKnowledgeTracing(p_init=0.05, p_transit=0.05, p_slip=0.15, p_guess=0.25)

        # Run through a realistic student trajectory
        trajectory = [True, False, True, True, False, True, True, True, True, True] * 10
        for correct in trajectory:
            bkt.update(correct=correct)

        p = bkt.p_known
        assert 0.0 <= p <= 1.0
        assert not math.isnan(p)
        assert not math.isinf(p)


# ════════════════════════════════════════════════════════════════════════
# 4.  CURRICULUM PIPELINE
# ════════════════════════════════════════════════════════════════════════

class TestCurriculumPipeline:
    """Graph → topological order → recommend → advance mastery → recommend again."""

    def test_full_curriculum_workflow(self):
        from orchid_ranker import (
            PrerequisiteGraph,
            CurriculumRecommender,
            MasteryTracker,
        )

        # 1. Build prerequisite graph
        graph = PrerequisiteGraph()
        graph.add_edge("fractions", "algebra")
        graph.add_edge("algebra", "calculus")
        graph.add_edge("algebra", "statistics")
        graph.add_edge("calculus", "differential_equations")

        # 2. Verify topological order
        order = graph.topological_order()
        assert order.index("fractions") < order.index("algebra")
        assert order.index("algebra") < order.index("calculus")

        # 3. Create recommender
        all_skills = ["fractions", "algebra", "calculus", "statistics", "differential_equations"]
        cr = CurriculumRecommender(
            graph=graph,
            difficulty_map={s: 0.2 + 0.15 * i for i, s in enumerate(all_skills)},
        )

        # 4. Start with nothing mastered
        recs = cr.recommend(student_mastery=set(), n=5)
        assert "fractions" in recs
        assert "calculus" not in recs  # prerequisite not met

        # 5. Master fractions → algebra becomes available
        recs = cr.recommend(student_mastery={"fractions"}, n=5)
        assert "algebra" in recs

        # 6. Master algebra → calculus and statistics become available
        recs = cr.recommend(student_mastery={"fractions", "algebra"}, n=5)
        assert "calculus" in recs or "statistics" in recs
        assert "differential_equations" not in recs  # needs calculus first

        # 7. Serialize and restore graph
        data = graph.to_dict()
        graph2 = PrerequisiteGraph.from_dict(data)
        assert graph2.topological_order() == graph.topological_order()


# ════════════════════════════════════════════════════════════════════════
# 5.  EDUCATIONAL METRICS PIPELINE
# ════════════════════════════════════════════════════════════════════════

class TestEducationalMetricsPipeline:
    """Full evaluation pipeline combining ranking and educational metrics."""

    def test_combined_evaluation(self):
        from orchid_ranker import (
            OrchidRecommender,
            train_test_split,
            learning_gain,
            knowledge_coverage,
            difficulty_appropriateness,
            engagement_score,
        )

        rng = np.random.RandomState(42)
        interactions = pd.DataFrame({
            "user_id": rng.randint(0, 20, size=500),
            "item_id": rng.randint(0, 30, size=500),
            "rating":  rng.uniform(1, 5, size=500).round(1),
        })

        train, test = train_test_split(interactions, test_size=0.2)
        rec = OrchidRecommender(strategy="popularity")
        rec.fit(train)

        uid = train["user_id"].iloc[0]
        recs = rec.recommend(user_id=uid, top_k=10)

        # Educational metrics
        gain = learning_gain(0.3, 0.75)
        assert 0.0 < gain <= 1.0

        coverage = knowledge_coverage({"a", "b"}, {"a", "b", "c", "d"})
        assert coverage == 0.5

        appropriateness = difficulty_appropriateness([0.5, 0.55, 0.6], 0.5, 0.25)
        assert 0.0 <= appropriateness <= 1.0

        eng = engagement_score([1, 2, 3, 4, 5], 10)
        assert 0.0 <= eng <= 1.0


# ════════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════════
def run_all():
    """Simple test runner for E2E tests."""
    test_classes = [
        TestRecommenderPipeline,
        TestCVTuningPipeline,
        TestKnowledgeTracingPipeline,
        TestCurriculumPipeline,
        TestEducationalMetricsPipeline,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = 0
    failures = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        print(f"\n--- {cls.__name__} ---")
        for method_name in sorted(methods):
            total += 1
            test_id = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS  {test_id}")
            except AssertionError as e:
                failed += 1
                failures.append((test_id, "FAIL", str(e)))
                print(f"  FAIL  {test_id}: {e}")
            except Exception as e:
                errors += 1
                failures.append((test_id, "ERROR", f"{type(e).__name__}: {e}"))
                print(f"  ERROR {test_id}: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"E2E RESULTS: {passed} passed, {failed} failed, {errors} errors / {total} total")
    print(f"{'='*60}")

    if failures:
        print("\nFAILURES:")
        for test_id, status, msg in failures:
            print(f"  [{status}] {test_id}")
            print(f"         {msg[:200]}")

    return passed, failed, errors, total


if __name__ == "__main__":
    run_all()
