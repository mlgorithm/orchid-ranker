"""Comprehensive tests for under-tested agent modules:
- config.py (MultiConfig, UserCtx, PolicyState, OnlineState)
- policies.py (LinUCBPolicy, BootTS)
- dual_recommender.py (DualRecommender)
- logging_util.py (JSONLLogger)
- timing.py (_TimingRecorder)
- rec_shim.py (RecShim)
- orchestrator.py (MultiUserOrchestrator)
"""
import sys
sys.path.insert(0, "src")

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from orchid_ranker.agents.config import MultiConfig, OnlineState, PolicyState, UserCtx
from orchid_ranker.agents.policies import LinUCBPolicy, BootTS
from orchid_ranker.agents.logging_util import JSONLLogger
from orchid_ranker.agents.timing import _TimingRecorder


# ─────────────────────────────────────────────
# MultiConfig
# ─────────────────────────────────────────────
class TestMultiConfig:
    def test_default_values(self):
        cfg = MultiConfig()
        assert cfg.rounds == 10
        assert cfg.top_k_base == 5
        assert cfg.zpd_margin == 0.12
        assert cfg.min_candidates == 100
        assert cfg.epsilon_total_global == 0.0
        assert cfg.novelty_bonus == 0.10
        assert cfg.mmr_lambda == 0.25
        assert cfg.log_path is None
        assert cfg.privacy_mode == "standard"
        assert cfg.share_signals is False

    def test_custom_values(self):
        cfg = MultiConfig(rounds=50, top_k_base=10, zpd_margin=0.2, epsilon_total_global=1.0)
        assert cfg.rounds == 50
        assert cfg.top_k_base == 10
        assert cfg.zpd_margin == 0.2
        assert cfg.epsilon_total_global == 1.0

    def test_tuple_bounds(self):
        cfg = MultiConfig(alpha_bounds=(0.2, 0.9), k_bounds=(3, 8))
        assert cfg.alpha_bounds == (0.2, 0.9)
        assert cfg.k_bounds == (3, 8)

    def test_deterministic_pool(self):
        cfg = MultiConfig(deterministic_pool=True, pool_seed=99)
        assert cfg.deterministic_pool is True
        assert cfg.pool_seed == 99

    def test_warmup_settings(self):
        cfg = MultiConfig(warmup_rounds=5, warmup_steps=3, warmup_preloop=True)
        assert cfg.warmup_rounds == 5
        assert cfg.warmup_steps == 3
        assert cfg.warmup_preloop is True

    def test_funk_settings(self):
        cfg = MultiConfig(funk_distill=True, funk_lambda=0.5, use_funk_candidates=True, funk_pool_size=50)
        assert cfg.funk_distill is True
        assert cfg.funk_lambda == 0.5
        assert cfg.use_funk_candidates is True
        assert cfg.funk_pool_size == 50

    def test_training_augmentation(self):
        cfg = MultiConfig(train_on_all_shown=True, train_steps_per_round=3)
        assert cfg.train_on_all_shown is True
        assert cfg.train_steps_per_round == 3


# ─────────────────────────────────────────────
# PolicyState
# ─────────────────────────────────────────────
class TestPolicyState:
    def test_construction_required_fields(self):
        ps = PolicyState(alpha=0.5, lam=0.3, top_k=5, zpd_delta=0.12, novelty=0.1)
        assert ps.alpha == 0.5
        assert ps.lam == 0.3
        assert ps.top_k == 5
        assert ps.zpd_delta == 0.12
        assert ps.novelty == 0.1

    def test_default_moving_averages(self):
        ps = PolicyState(alpha=0.5, lam=0.3, top_k=5, zpd_delta=0.12, novelty=0.1)
        assert ps.accept_ma == 0.5
        assert ps.acc_ma == 0.6
        assert ps.novelty_ma == 0.5
        assert ps.reward_ma == 0.55
        assert ps.knowledge_ma == 0.5
        assert ps.knowledge_delta_ma == 0.0
        assert ps.rounds == 0

    def test_custom_moving_averages(self):
        ps = PolicyState(
            alpha=0.5, lam=0.3, top_k=5, zpd_delta=0.12, novelty=0.1,
            accept_ma=0.8, acc_ma=0.9, rounds=10,
        )
        assert ps.accept_ma == 0.8
        assert ps.acc_ma == 0.9
        assert ps.rounds == 10

    def test_mutation(self):
        ps = PolicyState(alpha=0.5, lam=0.3, top_k=5, zpd_delta=0.12, novelty=0.1)
        ps.alpha = 0.7
        ps.rounds = 5
        assert ps.alpha == 0.7
        assert ps.rounds == 5


# ─────────────────────────────────────────────
# OnlineState
# ─────────────────────────────────────────────
class TestOnlineState:
    def test_empty_state(self):
        state = OnlineState()
        result = state.get(999)
        assert result["knowledge"] == 0.5
        assert result["fatigue"] == 0.2
        assert result["trust"] == 0.5
        assert result["engagement"] == 0.6
        assert result["uncertainty"] == 0.5

    def test_set_and_get(self):
        state = OnlineState()
        state.set_initial(1, knowledge=0.8, fatigue=0.1, engagement=0.9, trust=0.7, uncertainty=0.3)
        result = state.get(1)
        assert result["knowledge"] == 0.8
        assert result["fatigue"] == 0.1
        assert result["engagement"] == 0.9
        assert result["trust"] == 0.7
        assert result["uncertainty"] == 0.3

    def test_multiple_users(self):
        state = OnlineState()
        state.set_initial(1, knowledge=0.8, fatigue=0.1, engagement=0.9, trust=0.7, uncertainty=0.3)
        state.set_initial(2, knowledge=0.3, fatigue=0.5, engagement=0.4, trust=0.2, uncertainty=0.8)
        r1 = state.get(1)
        r2 = state.get(2)
        assert r1["knowledge"] == 0.8
        assert r2["knowledge"] == 0.3

    def test_get_returns_copy(self):
        state = OnlineState()
        state.set_initial(1, knowledge=0.8, fatigue=0.1, engagement=0.9, trust=0.7, uncertainty=0.3)
        r1 = state.get(1)
        r1["knowledge"] = 0.0
        r2 = state.get(1)
        assert r2["knowledge"] == 0.8

    def test_overwrite_user(self):
        state = OnlineState()
        state.set_initial(1, knowledge=0.5, fatigue=0.2, engagement=0.6, trust=0.5, uncertainty=0.5)
        state.set_initial(1, knowledge=0.9, fatigue=0.0, engagement=1.0, trust=1.0, uncertainty=0.0)
        r = state.get(1)
        assert r["knowledge"] == 0.9


# ─────────────────────────────────────────────
# UserCtx
# ─────────────────────────────────────────────
class TestUserCtx:
    def test_construction(self):
        vec = torch.randn(1, 4)
        ctx = UserCtx(user_id=42, user_idx=0, student=None, user_vec=vec)
        assert ctx.user_id == 42
        assert ctx.user_idx == 0
        assert ctx.profile is None
        assert ctx.name is None

    def test_with_optional_fields(self):
        vec = torch.randn(1, 4)
        ctx = UserCtx(user_id=1, user_idx=0, student="mock", user_vec=vec, profile="advanced", name="Alice")
        assert ctx.profile == "advanced"
        assert ctx.name == "Alice"


# ─────────────────────────────────────────────
# LinUCBPolicy (deeper tests beyond test_two_tower.py)
# ─────────────────────────────────────────────
class TestLinUCBPolicyDetailed:
    def test_ensure_creates_matrices(self):
        policy = LinUCBPolicy(d=5, alpha=1.0, l2=1.0)
        assert len(policy.A) == 0
        policy._ensure(0)
        assert 0 in policy.A
        assert policy.A[0].shape == (5, 5)
        assert policy.b[0].shape == (5,)

    def test_score_multiple_arms(self):
        policy = LinUCBPolicy(d=3, alpha=1.0, l2=1.0)
        x = np.array([1.0, 0.0, 0.0])
        s0 = policy.score(x, base=0.0, i=0)
        s1 = policy.score(x, base=0.0, i=1)
        assert isinstance(s0, float)
        assert isinstance(s1, float)
        assert 0 in policy.A
        assert 1 in policy.A

    def test_update_changes_matrices(self):
        policy = LinUCBPolicy(d=3, alpha=1.0, l2=1.0)
        policy._ensure(0)
        A_before = policy.A[0].copy()
        b_before = policy.b[0].copy()
        x = np.array([1.0, 2.0, 3.0])
        policy.update(0, x, 1.0)
        assert not np.allclose(policy.A[0], A_before)
        assert not np.allclose(policy.b[0], b_before)

    def test_score_with_base(self):
        policy = LinUCBPolicy(d=3, alpha=1.0, l2=1.0)
        x = np.array([1.0, 0.0, 0.0])
        s_no_base = policy.score(x, base=0.0, i=0)
        s_with_base = policy.score(x, base=5.0, i=0)
        assert abs(s_with_base - s_no_base - 5.0) < 1e-6

    def test_learning_improves_predictions(self):
        policy = LinUCBPolicy(d=3, alpha=0.1, l2=0.1)
        x_good = np.array([1.0, 0.0, 0.0])
        x_bad = np.array([0.0, 1.0, 0.0])
        # Train: arm 0 with x_good gets high reward, x_bad gets low reward
        for _ in range(20):
            policy.update(0, x_good, 1.0)
            policy.update(0, x_bad, 0.0)
        # After training, score for x_good should exceed score for x_bad
        s_good = policy.score(x_good, base=0.0, i=0)
        s_bad = policy.score(x_bad, base=0.0, i=0)
        assert s_good > s_bad

    def test_alpha_controls_exploration(self):
        d = 3
        x = np.random.randn(d)
        # High alpha = more exploration bonus
        lo = LinUCBPolicy(d=d, alpha=0.01, l2=1.0)
        hi = LinUCBPolicy(d=d, alpha=10.0, l2=1.0)
        s_lo = lo.score(x, base=0.0, i=0)
        s_hi = hi.score(x, base=0.0, i=0)
        # Higher alpha should give higher score due to confidence bonus
        assert s_hi >= s_lo

    def test_l2_regularization(self):
        policy = LinUCBPolicy(d=3, alpha=1.0, l2=100.0)
        x = np.array([1.0, 0.0, 0.0])
        policy.update(0, x, 1.0)
        score = policy.score(x, base=0.0, i=0)
        # With very high L2, the learned weight should be small
        assert abs(score) < 5.0


# ─────────────────────────────────────────────
# BootTS (deeper tests)
# ─────────────────────────────────────────────
class TestBootTSDetailed:
    def test_construction_initializes_heads(self):
        policy = BootTS(d=5, heads=7, l2=1.0, rng=42)
        assert len(policy.As) == 7
        assert len(policy.bs) == 7
        assert policy.As[0].shape == (5, 5)
        assert policy.bs[0].shape == (5,)

    def test_score_vec_stochastic(self):
        """Scores should vary across calls due to head randomization + noise."""
        policy = BootTS(d=5, heads=10, l2=1.0, rng=42)
        x = np.random.randn(5).astype(np.float64)
        scores = [policy.score_vec(x) for _ in range(20)]
        # Not all scores should be identical (random head selection + noise)
        assert len(set(round(s, 6) for s in scores)) > 1

    def test_update_modifies_selected_heads(self):
        policy = BootTS(d=3, heads=5, l2=1.0, rng=42)
        A_before = [a.copy() for a in policy.As]
        x = np.array([1.0, 2.0, 3.0])
        policy.update(x, 1.0, k=2)
        changed = sum(1 for i in range(5) if not np.allclose(policy.As[i], A_before[i]))
        assert changed == 2  # k=2 heads should be updated

    def test_update_k_capped_at_H(self):
        policy = BootTS(d=3, heads=3, l2=1.0, rng=42)
        A_before = [a.copy() for a in policy.As]
        x = np.array([1.0, 0.0, 0.0])
        policy.update(x, 1.0, k=10)  # k > H, should update at most H=3
        changed = sum(1 for i in range(3) if not np.allclose(policy.As[i], A_before[i]))
        assert changed == 3

    def test_reproducibility_with_same_seed(self):
        x = np.random.randn(5).astype(np.float64)
        p1 = BootTS(d=5, heads=3, l2=1.0, rng=99)
        p2 = BootTS(d=5, heads=3, l2=1.0, rng=99)
        scores1 = [p1.score_vec(x) for _ in range(5)]
        scores2 = [p2.score_vec(x) for _ in range(5)]
        assert scores1 == scores2


# ─────────────────────────────────────────────
# JSONLLogger
# ─────────────────────────────────────────────
class TestJSONLLoggerDetailed:
    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "subdir" / "nested" / "test.jsonl"
            logger = JSONLLogger(log_path)
            assert logger.path.parent.exists()

    def test_log_numpy_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JSONLLogger(log_path)
            record = {
                "float32": np.float32(1.5),
                "int64": np.int64(42),
                "array": np.array([1, 2, 3]),
                "bool": np.bool_(True),
            }
            logger.log(record)
            with open(log_path) as f:
                data = json.loads(f.readline())
            assert data["float32"] == 1.5
            assert data["int64"] == 42
            assert data["array"] == [1, 2, 3]
            assert data["bool"] is True

    def test_log_torch_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JSONLLogger(log_path)
            record = {"tensor": torch.tensor([1.0, 2.0, 3.0])}
            logger.log(record)
            with open(log_path) as f:
                data = json.loads(f.readline())
            assert data["tensor"] == [1.0, 2.0, 3.0]

    def test_log_nested_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JSONLLogger(log_path)
            record = {"outer": {"inner": np.float64(3.14), "list": [np.int32(1)]}}
            logger.log(record)
            with open(log_path) as f:
                data = json.loads(f.readline())
            assert abs(data["outer"]["inner"] - 3.14) < 1e-6
            assert data["outer"]["list"] == [1]

    def test_log_truncates_large_user_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JSONLLogger(log_path, max_feature_dump=5)
            features = {f"feat_{i}": float(i) for i in range(100)}
            record = {"inputs": {"user_features": features}}
            logger.log(record)
            with open(log_path) as f:
                data = json.loads(f.readline())
            assert "user_features" not in data["inputs"]
            assert "user_features_sample" in data["inputs"]
            assert len(data["inputs"]["user_features_sample"]) == 5
            assert data["inputs"]["user_features_count"] == 100

    def test_log_small_user_features_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JSONLLogger(log_path, max_feature_dump=256)
            features = {f"feat_{i}": float(i) for i in range(10)}
            record = {"inputs": {"user_features": features}}
            logger.log(record)
            with open(log_path) as f:
                data = json.loads(f.readline())
            assert "user_features" in data["inputs"]
            assert len(data["inputs"]["user_features"]) == 10

    def test_log_non_serializable_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JSONLLogger(log_path)
            record = {"custom": object()}
            logger.log(record)
            with open(log_path) as f:
                data = json.loads(f.readline())
            assert isinstance(data["custom"], str)

    def test_multiple_writes_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            logger = JSONLLogger(log_path)
            for i in range(5):
                logger.log({"idx": i})
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 5
            for i, line in enumerate(lines):
                assert json.loads(line)["idx"] == i


# ─────────────────────────────────────────────
# _TimingRecorder
# ─────────────────────────────────────────────
class TestTimingRecorder:
    def test_disabled_when_no_path(self):
        t = _TimingRecorder(None, 10)
        assert not t.enabled

    def test_disabled_when_zero_rounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            t = _TimingRecorder(Path(tmpdir) / "t.jsonl", 0)
            assert not t.enabled

    def test_enabled_with_valid_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.jsonl"
            t = _TimingRecorder(path, 5)
            assert t.enabled

    def test_begin_and_finish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.jsonl"
            t = _TimingRecorder(path, 5)
            t.begin(0)
            t.finish()
            assert path.exists()
            with open(path) as f:
                entry = json.loads(f.readline())
            assert entry["round"] == 0
            assert "total" in entry

    def test_phase_timing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.jsonl"
            t = _TimingRecorder(path, 5)
            t.begin(0)
            with t.phase("infer"):
                pass  # instant
            with t.phase("decide"):
                pass
            t.finish()
            with open(path) as f:
                entry = json.loads(f.readline())
            assert "infer" in entry["phases"]
            assert "decide" in entry["phases"]

    def test_finish_with_extras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.jsonl"
            t = _TimingRecorder(path, 5)
            t.begin(0)
            t.finish(extras={"precision": 0.8})
            with open(path) as f:
                entry = json.loads(f.readline())
            assert entry["metrics"]["precision"] == 0.8

    def test_respects_max_rounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.jsonl"
            t = _TimingRecorder(path, 2)
            for i in range(5):
                t.begin(i)
                t.finish()
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # only first 2 rounds recorded

    def test_phase_noop_when_disabled(self):
        t = _TimingRecorder(None, 0)
        # Should not raise
        with t.phase("test"):
            pass
        t.begin(0)
        t.finish()  # no-op

    def test_phase_accumulates_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.jsonl"
            t = _TimingRecorder(path, 5)
            t.begin(0)
            with t.phase("work"):
                import time
                time.sleep(0.01)
            t.finish()
            with open(path) as f:
                entry = json.loads(f.readline())
            assert entry["phases"]["work"] > 0.0


# ─────────────────────────────────────────────
# DualRecommender
# ─────────────────────────────────────────────
class TestDualRecommender:
    """Tests for DualRecommender wrapper."""

    @pytest.fixture
    def teacher_student(self):
        from orchid_ranker.agents.two_tower import TwoTowerRecommender
        device = torch.device("cpu")
        teacher = TwoTowerRecommender(
            num_users=5, num_items=10, user_dim=4, item_dim=4,
            emb_dim=8, device=device,
        ).to(device)
        student = TwoTowerRecommender(
            num_users=5, num_items=10, user_dim=4, item_dim=4,
            emb_dim=8, device=device,
        ).to(device)
        return teacher, student, device

    def test_construction(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        assert dual.teacher is teacher
        assert dual.student is student
        assert dual._student_weight == 0.0

    def test_warm_start_copies_params(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device, warm_start=True)
        # Teacher params should match student params
        for t_param, s_param in zip(dual.teacher.parameters(), dual.student.parameters()):
            assert torch.allclose(t_param, s_param)

    def test_infer_teacher_only_when_weight_zero(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        assert dual._student_weight == 0.0
        user_vec = torch.randn(1, 4, device=device)
        item_matrix = torch.randn(10, 4, device=device)
        user_ids = torch.tensor([0], dtype=torch.long, device=device)
        item_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        logits = dual.infer(
            user_vec=user_vec, item_matrix=item_matrix,
            user_ids=user_ids, item_ids=item_ids,
        )
        assert logits.shape == (1, 3)

    def test_novelty_bonus_property(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        dual.novelty_bonus = 0.5
        assert dual.novelty_bonus == 0.5

    def test_mmr_lambda_property(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        dual.mmr_lambda = 0.4
        assert dual.mmr_lambda == 0.4

    def test_dp_settings_passthrough(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        # dp_settings is read from student
        result = dual.dp_settings
        assert result is getattr(student, "dp_settings", None)

    def test_eps_cum_passthrough(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        assert dual.eps_cum == 0.0

    def test_decide_delegates_to_teacher(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        user_vec = torch.randn(1, 4, device=device)
        item_matrix = torch.randn(10, 4, device=device)
        user_ids = torch.tensor([0], dtype=torch.long, device=device)
        item_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device)
        logits = dual.infer(
            user_vec=user_vec, item_matrix=item_matrix,
            user_ids=user_ids, item_ids=item_ids,
        )
        chosen, meta = dual.decide(
            logits=logits, item_ids=item_ids, top_k=2,
            user_id=0, engagement=0.5, trust=0.5,
            difficulty_map={}, knowledge=0.5, zpd_delta=0.5,
        )
        assert isinstance(chosen, list)
        assert len(chosen) <= 2

    def test_after_student_update_increments_weight(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        initial_weight = dual._student_weight
        dual._after_student_update()
        assert dual._student_weight > initial_weight

    def test_student_weight_capped_at_one(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        for _ in range(100):
            dual._after_student_update()
        assert dual._student_weight <= 1.0

    def test_infer_policy_teacher(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        user_vec = torch.randn(1, 4, device=device)
        item_matrix = torch.randn(10, 4, device=device)
        user_ids = torch.tensor([0], dtype=torch.long, device=device)
        item_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        logits = dual.infer_policy(
            policy="teacher",
            user_vec=user_vec, item_matrix=item_matrix,
            user_ids=user_ids, item_ids=item_ids,
        )
        assert logits.shape == (1, 3)

    def test_infer_policy_student(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)
        user_vec = torch.randn(1, 4, device=device)
        item_matrix = torch.randn(10, 4, device=device)
        user_ids = torch.tensor([0], dtype=torch.long, device=device)
        item_ids = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        logits = dual.infer_policy(
            policy="student",
            user_vec=user_vec, item_matrix=item_matrix,
            user_ids=user_ids, item_ids=item_ids,
        )
        assert logits.shape == (1, 3)

    def test_replay_buffer(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device, replay_size=10, replay_steps=1)
        assert dual._replay_buf is not None
        assert dual._replay_steps == 1

    def test_student_teacher_anchor_tracks_wrapper_teacher(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)

        with torch.no_grad():
            for param in dual.student.parameters():
                param.add_(0.5)

        dual._after_student_update()

        for anchor_param, teacher_param in zip(dual.student.teacher.parameters(), dual.teacher.parameters()):
            assert torch.allclose(anchor_param, teacher_param)

    def test_train_step_replay_samples_buffer(self, teacher_student, monkeypatch):
        from orchid_ranker.agents.dual_recommender import DualRecommender

        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device, replay_size=10, replay_steps=1)

        calls = []

        def fake_train_step(batch):
            calls.append(batch["tag"])
            return {"loss": 0.0}

        monkeypatch.setattr(student, "train_step", fake_train_step)
        dual._replay_buf.extend([{"tag": "oldest"}, {"tag": "sampled"}])
        monkeypatch.setattr(np.random, "randint", lambda n: 1)

        dual.train_step({"tag": "current"})

        assert calls == ["current", "sampled"]

    def test_update_no_method_returns_noop(self):
        """When student has no update/train_step, returns noop."""
        from orchid_ranker.agents.dual_recommender import DualRecommender

        class Bare(nn.Module):
            def forward(self, x):
                return x

        teacher, student = Bare(), Bare()
        dual = DualRecommender(teacher, student)
        result = dual.update()
        assert result.get("note") == "no-update-implemented"

    def test_call_with_supported_args_filters(self, teacher_student):
        from orchid_ranker.agents.dual_recommender import DualRecommender
        teacher, student, device = teacher_student
        dual = DualRecommender(teacher, student, device=device)

        def fn(a, b):
            return a + b
        result = dual._call_with_supported_args(fn, a=1, b=2, c=3)
        assert result == 3


# ─────────────────────────────────────────────
# RecShim
# ─────────────────────────────────────────────
class TestRecShim:
    def test_construction(self):
        from orchid_ranker.agents.rec_shim import RecShim
        from orchid_ranker.agents.two_tower import TwoTowerRecommender
        device = torch.device("cpu")
        rec = TwoTowerRecommender(
            num_users=5, num_items=10, user_dim=4, item_dim=4, device=device,
        ).to(device)
        shim = RecShim(rec)
        assert shim.device == device
        assert shim.eps_cum == 0.0

    def test_think_passthrough(self):
        from orchid_ranker.agents.rec_shim import RecShim
        from orchid_ranker.agents.two_tower import TwoTowerRecommender
        device = torch.device("cpu")
        rec = TwoTowerRecommender(
            num_users=5, num_items=10, user_dim=4, item_dim=4,
            state_dim=4, device=device,
        ).to(device)
        shim = RecShim(rec)
        user_vec = torch.randn(1, 4)
        item_matrix = torch.randn(10, 4)
        user_ids = torch.tensor([0], dtype=torch.long)
        item_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        state_vec = torch.randn(1, 4)
        logits = shim.think(
            user_vec=user_vec, item_matrix=item_matrix,
            user_ids=user_ids, item_ids=item_ids, state_vec=state_vec,
        )
        assert logits.shape == (1, 3)

    def test_mmr_lambda_property(self):
        from orchid_ranker.agents.rec_shim import RecShim
        from orchid_ranker.agents.two_tower import TwoTowerRecommender
        device = torch.device("cpu")
        rec = TwoTowerRecommender(
            num_users=5, num_items=10, user_dim=4, item_dim=4, device=device,
        ).to(device)
        shim = RecShim(rec)
        shim.mmr_lambda = 0.5
        assert shim.mmr_lambda == 0.5

    def test_update_normalizes_external_feedback_ids(self):
        from orchid_ranker.agents.rec_shim import RecShim

        class DummyRec:
            def __init__(self):
                self.device = torch.device("cpu")
                self.dp_cfg = {"enabled": False}
                self.eps_cum = 0.0
                self.pos2id_map = {0: 100, 1: 200}
                self.received_feedback = None

            def update(self, **kwargs):
                self.received_feedback = dict(kwargs["feedback"])
                return {"loss": 0.0}

        rec = DummyRec()
        shim = RecShim(rec)
        shim.update(
            feedback={100: 1, 200: 0},
            user_vec=[0.0],
            state_vec=[0.0],
            user_ids=[0],
            item_matrix=[[0.0]],
            item_ids=[0, 1],
            epochs=1,
        )

        assert rec.received_feedback == {0: 1, 1: 0}


# ─────────────────────────────────────────────
# MultiUserOrchestrator (core functionality)
# ─────────────────────────────────────────────
class TestMultiUserOrchestrator:
    """Unit tests for MultiUserOrchestrator initialization and helpers."""

    @pytest.fixture
    def orchestrator_setup(self):
        from orchid_ranker.agents.two_tower import TwoTowerRecommender
        from orchid_ranker.agents.orchestrator import MultiUserOrchestrator

        device = torch.device("cpu")
        num_users, num_items = 3, 10
        user_dim, item_dim = 4, 4

        rec = TwoTowerRecommender(
            num_users=num_users, num_items=num_items,
            user_dim=user_dim, item_dim=item_dim,
            emb_dim=8, device=device,
        ).to(device)

        item_matrix = torch.randn(num_items, item_dim, device=device)
        item_ids = torch.arange(num_items, dtype=torch.long, device=device)
        pos2id = list(range(num_items))
        id2pos = {i: i for i in range(num_items)}
        item_meta = {i: {"difficulty": 0.5, "topic": "math"} for i in range(num_items)}

        users = []
        for uid in range(num_users):
            vec = torch.randn(1, user_dim, device=device)
            ctx = UserCtx(user_id=uid, user_idx=uid, student=None, user_vec=vec, profile="default")
            users.append(ctx)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MultiConfig(log_path=os.path.join(tmpdir, "test.jsonl"), rounds=3)
            orch = MultiUserOrchestrator(
                rec=rec, users=users,
                item_matrix_normal=item_matrix,
                item_matrix_sanitized=None,
                item_ids_pos=item_ids,
                pos2id=pos2id, id2pos=id2pos,
                item_meta_by_id=item_meta,
                cfg=cfg, device=device,
                mode_label="test",
            )
            yield orch, tmpdir

    def test_construction(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        assert len(orch.users) == 3
        assert orch._mode_label == "test"
        assert orch.cfg.rounds == 3

    def test_online_state_initialized(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        assert isinstance(orch.state, OnlineState)

    def test_difficulty_map_populated(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        assert len(orch._difficulty_map) == 10
        for iid, diff in orch._difficulty_map.items():
            assert diff == 0.5

    def test_seen_by_user_initialized(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        assert len(orch._seen_by_user) == 3
        for uid in range(3):
            assert orch._seen_by_user[uid] == set()

    def test_accepted_by_user_initialized(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        for uid in range(3):
            assert orch._accepted_by_user[uid] == []

    def test_reset_round_counters(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        orch._round["shown"] = 100
        orch._round["accepted"] = 50
        orch._reset_round_counters()
        assert orch._round["shown"] == 0
        assert orch._round["accepted"] == 0

    def test_is_adaptive_false_for_plain_rec(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        assert orch._is_adaptive is False

    def test_logger_created(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        assert isinstance(orch.logger, JSONLLogger)

    def test_pop_tracking_initialized(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        assert orch._pop_expose == {}
        assert orch._pop_accept == {}

    def test_ewma_states_initialized(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        for uid in range(3):
            assert uid in orch._khat
            assert uid in orch._ehat
            assert uid in orch._uunc

    def test_user_profiles_tracked(self, orchestrator_setup):
        orch, _ = orchestrator_setup
        for uid in range(3):
            assert orch._user_profiles[uid] == "default"


class TestMultiUserOrchestratorAdaptive:
    """Tests for adaptive mode with DualRecommender."""

    @pytest.fixture
    def adaptive_setup(self):
        from orchid_ranker.agents.two_tower import TwoTowerRecommender
        from orchid_ranker.agents.dual_recommender import DualRecommender
        from orchid_ranker.agents.orchestrator import MultiUserOrchestrator

        device = torch.device("cpu")
        num_users, num_items = 2, 8
        user_dim, item_dim = 4, 4

        teacher = TwoTowerRecommender(
            num_users=num_users, num_items=num_items,
            user_dim=user_dim, item_dim=item_dim,
            emb_dim=8, device=device,
        ).to(device)
        student = TwoTowerRecommender(
            num_users=num_users, num_items=num_items,
            user_dim=user_dim, item_dim=item_dim,
            emb_dim=8, device=device,
        ).to(device)
        rec = DualRecommender(teacher, student, device=device)

        item_matrix = torch.randn(num_items, item_dim, device=device)
        item_ids = torch.arange(num_items, dtype=torch.long, device=device)
        pos2id = list(range(num_items))
        id2pos = {i: i for i in range(num_items)}
        item_meta = {i: {"difficulty": 0.3 + 0.1 * i} for i in range(num_items)}

        users = []
        for uid in range(num_users):
            vec = torch.randn(1, user_dim, device=device)
            ctx = UserCtx(user_id=uid, user_idx=uid, student=None, user_vec=vec)
            users.append(ctx)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MultiConfig(
                log_path=os.path.join(tmpdir, "test.jsonl"),
                rounds=2,
                alpha_bounds=(0.1, 0.8),
                k_bounds=(2, 6),
            )
            orch = MultiUserOrchestrator(
                rec=rec, users=users,
                item_matrix_normal=item_matrix,
                item_matrix_sanitized=None,
                item_ids_pos=item_ids,
                pos2id=pos2id, id2pos=id2pos,
                item_meta_by_id=item_meta,
                cfg=cfg, device=device,
                mode_label="adaptive",
            )
            yield orch, tmpdir

    def test_is_adaptive_true(self, adaptive_setup):
        orch, _ = adaptive_setup
        assert orch._is_adaptive is True

    def test_policy_state_per_user(self, adaptive_setup):
        orch, _ = adaptive_setup
        assert len(orch._policy_state) == 2
        for uid in range(2):
            ps = orch._policy_state[uid]
            assert isinstance(ps, PolicyState)
            assert 0.1 <= ps.alpha <= 0.8
            assert 2 <= ps.top_k <= 6

    def test_init_policy_state_midpoints(self, adaptive_setup):
        orch, _ = adaptive_setup
        ps = orch._init_policy_state()
        # alpha should be midpoint of (0.1, 0.8) = 0.45
        assert abs(ps.alpha - 0.45) < 0.01
        assert ps.rounds == 0

    def test_apply_policy_sets_knobs(self, adaptive_setup):
        orch, _ = adaptive_setup
        ps = PolicyState(alpha=0.6, lam=0.4, top_k=4, zpd_delta=0.15, novelty=0.2)
        orch._apply_policy(ps)
        # Check that the model knobs got set
        if hasattr(orch.rec.teacher, "mmr_lambda"):
            assert orch.rec.teacher.mmr_lambda == 0.4
        if hasattr(orch.rec.student, "mmr_lambda"):
            assert orch.rec.student.mmr_lambda == 0.4

    def test_difficulty_map_varies(self, adaptive_setup):
        orch, _ = adaptive_setup
        diffs = list(orch._difficulty_map.values())
        assert len(set(diffs)) > 1  # not all the same

    def test_warmup_buffer_exists(self, adaptive_setup):
        orch, _ = adaptive_setup
        assert hasattr(orch, "_warmup_buffer")
        assert len(orch._warmup_buffer) == 0
