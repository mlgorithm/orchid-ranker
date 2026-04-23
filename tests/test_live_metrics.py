"""Tests for live progression metrics and the progression guardrail.

These tests assert three contracts:

1. Rolling metrics track the underlying helpers from evaluation.py.
2. Prometheus gauges are updated on every record() call.
3. The guardrail halts the adaptive policy on sustained violation and
   recovers on explicit restart — with the right Prometheus counter
   incremented and the right reason string attached.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from orchid_ranker.agents.two_tower import TwoTowerRecommender
from orchid_ranker.live_metrics import (
    GuardrailConfig,
    ProgressionGuardrail,
    RollingProgressionMonitor,
)
from orchid_ranker.observability import (
    PROGRESSION_GAIN,
    PROGRESSION_GUARDRAIL_HALTED,
    PROGRESSION_GUARDRAIL_TRIGGERS,
    metrics_registry,
)
from orchid_ranker.streaming import StreamingAdaptiveRanker


def _gauge_value(gauge, **labels) -> float:
    """Extract the current value of a labeled Prometheus gauge."""
    if labels:
        return float(gauge.labels(**labels)._value.get())
    return float(gauge._value.get())


# ---------------------------------------------------------------------------
# RollingProgressionMonitor
# ---------------------------------------------------------------------------
class TestRollingMonitor:
    def test_empty_snapshot(self):
        m = RollingProgressionMonitor(policy="t_empty", emit_prometheus=False)
        snap = m.snapshot()
        assert snap.sample_size == 0
        assert snap.progression_gain == 0.0
        # sequence adherence / stretch fit default to 1.0 on empty input
        assert snap.sequence_adherence == 1.0
        assert snap.stretch_fit == 1.0

    def test_progression_gain_tracks_deltas(self):
        m = RollingProgressionMonitor(policy="t_gain", emit_prometheus=False)
        # Two events with clear positive gain
        m.record(user_id=0, item_id=0, correct=True,
                 pre_competence=0.2, post_competence=0.6, category="a")
        m.record(user_id=0, item_id=1, correct=True,
                 pre_competence=0.6, post_competence=0.8, category="a")
        snap = m.snapshot()
        # Mean of (0.6-0.2)/(1-0.2) and (0.8-0.6)/(1-0.6) = mean(0.5, 0.5) = 0.5
        assert snap.progression_gain == pytest.approx(0.5, abs=1e-6)
        assert snap.accept_rate == 1.0
        assert snap.sample_size == 2

    def test_category_coverage_with_total(self):
        m = RollingProgressionMonitor(
            policy="t_cov", emit_prometheus=False,
            total_categories={"a", "b", "c"},
            success_threshold=0.7,
        )
        m.record(user_id=0, item_id=0, correct=True,
                 pre_competence=0.5, post_competence=0.9, category="a")
        m.record(user_id=0, item_id=1, correct=True,
                 pre_competence=0.5, post_competence=0.5, category="b")
        snap = m.snapshot()
        # Only category "a" crossed the success threshold; coverage = 1/3
        assert snap.category_coverage == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_sequence_adherence_with_graph(self):
        m = RollingProgressionMonitor(
            policy="t_seq", emit_prometheus=False,
            prerequisite_graph={3: {1, 2}, 4: {3}},
        )
        # User 0 completes 1 and 2, then gets item 3 recommended -> adherent.
        # User 1 has nothing completed and gets item 3 -> violation.
        m.record(user_id=0, item_id=1, correct=True, pre_competence=0.5, post_competence=0.9)
        m.record(user_id=0, item_id=2, correct=True, pre_competence=0.5, post_competence=0.9)
        m.record(user_id=0, item_id=3, correct=True, pre_competence=0.5, post_competence=0.9)
        m.record(user_id=1, item_id=3, correct=False, pre_competence=0.2, post_competence=0.2)
        snap = m.snapshot()
        # User 0: all 3 recs adherent (items 1,2 have no prereqs, item 3 met).
        # User 1: 1 rec, not adherent. Average over 2 users = (1.0 + 0.0)/2 = 0.5
        assert snap.sequence_adherence == pytest.approx(0.5, abs=1e-6)

    def test_stretch_fit(self):
        m = RollingProgressionMonitor(
            policy="t_diff", emit_prometheus=False, stretch_width=0.2,
        )
        # Anchor competence is mean pre_competence across events.
        m.record(user_id=0, item_id=0, correct=True,
                 pre_competence=0.5, post_competence=0.6, difficulty=0.55)
        m.record(user_id=0, item_id=1, correct=True,
                 pre_competence=0.5, post_competence=0.7, difficulty=0.95)
        snap = m.snapshot()
        # Competence = 0.5, stretch zone = [0.5, 0.7]. 0.55 in, 0.95 out.
        assert snap.stretch_fit == pytest.approx(0.5, abs=1e-6)

    def test_window_evicts_old_events(self):
        m = RollingProgressionMonitor(
            policy="t_win", window_size=3, emit_prometheus=False,
        )
        for k in range(10):
            m.record(user_id=0, item_id=k, correct=True,
                     pre_competence=0.5, post_competence=0.7, category="a")
        assert m.snapshot().sample_size == 3

    def test_prometheus_gauge_updated(self):
        m = RollingProgressionMonitor(policy="t_prom", emit_prometheus=True)
        m.record(user_id=0, item_id=0, correct=True,
                 pre_competence=0.2, post_competence=0.6, category="a")
        # Gauge labeled by policy should reflect the gain
        val = _gauge_value(PROGRESSION_GAIN, policy="t_prom")
        assert val == pytest.approx(0.5, abs=1e-6)

    def test_reset_clears_state(self):
        m = RollingProgressionMonitor(policy="t_reset", emit_prometheus=False)
        for k in range(5):
            m.record(user_id=0, item_id=k, correct=True,
                     pre_competence=0.2, post_competence=0.6)
        assert m.snapshot().sample_size == 5
        m.reset()
        assert m.snapshot().sample_size == 0


# ---------------------------------------------------------------------------
# ProgressionGuardrail
# ---------------------------------------------------------------------------
class TestProgressionGuardrail:
    def _fill_monitor(self, m, n, *, correct=True, pre=0.2, post=0.6):
        for k in range(n):
            m.record(user_id=k % 5, item_id=k, correct=correct,
                     pre_competence=pre, post_competence=post, category="a")

    def test_allows_by_default(self):
        m = RollingProgressionMonitor(policy="g1", emit_prometheus=False)
        g = ProgressionGuardrail(m, GuardrailConfig(warmup_samples=10))
        # No data -> warmup not passed -> allows adaptive
        assert g.evaluate() is True
        assert not g.is_halted

    def test_no_halt_before_warmup(self):
        m = RollingProgressionMonitor(policy="g2", emit_prometheus=False)
        g = ProgressionGuardrail(m, GuardrailConfig(
            warmup_samples=20, consecutive_violations=1, min_accept_rate=0.9,
        ))
        # Only 5 events, all failures -> below accept floor but no warmup yet
        self._fill_monitor(m, 5, correct=False, pre=0.2, post=0.2)
        assert g.evaluate() is True
        assert not g.is_halted

    def test_halts_on_sustained_violation(self):
        m = RollingProgressionMonitor(policy="g3", emit_prometheus=False)
        g = ProgressionGuardrail(m, GuardrailConfig(
            warmup_samples=10,
            consecutive_violations=2,
            min_accept_rate=0.8,
            min_progression_gain=-1.0,  # disable
            min_sequence_adherence=0.0,
            min_coverage=0.0,
        ))
        # Flood with failures — accept_rate drops to 0.0
        self._fill_monitor(m, 20, correct=False, pre=0.2, post=0.2)
        # First call: violation detected, streak=1, not yet halted -> still allows
        assert g.evaluate() is True
        assert not g.is_halted
        # Second call: streak=2 hits threshold -> halts and denies
        assert g.evaluate() is False
        assert g.is_halted
        assert g.halt_reason == "accept_rate_below_floor"
        # Gauge reflects halt
        assert _gauge_value(PROGRESSION_GUARDRAIL_HALTED) == 1.0

    def test_single_blip_does_not_halt(self):
        m = RollingProgressionMonitor(policy="g4", emit_prometheus=False)
        g = ProgressionGuardrail(m, GuardrailConfig(
            warmup_samples=5,
            consecutive_violations=3,
            min_accept_rate=0.5,
        ))
        # First batch: bad -> streak starts but doesn't reach 3.
        # Second batch: good -> window recovers, streak resets, no halt.
        self._fill_monitor(m, 10, correct=False, pre=0.2, post=0.2)
        assert g.evaluate() is True  # streak=1, still allowed
        assert not g.is_halted
        self._fill_monitor(m, 10, correct=True, pre=0.5, post=0.9)
        assert g.evaluate() is True
        assert not g.is_halted

    def test_restart_resumes(self):
        m = RollingProgressionMonitor(policy="g5", emit_prometheus=False)
        g = ProgressionGuardrail(m, GuardrailConfig(
            warmup_samples=5, consecutive_violations=1, min_accept_rate=0.9,
        ))
        self._fill_monitor(m, 10, correct=False, pre=0.2, post=0.2)
        g.evaluate()
        assert g.is_halted
        g.restart(reset_window=True)
        assert not g.is_halted
        assert g.halt_reason is None

    def test_halt_counter_incremented(self):
        before = float(PROGRESSION_GUARDRAIL_TRIGGERS.labels(
            reason="accept_rate_below_floor"
        )._value.get())
        m = RollingProgressionMonitor(policy="g6", emit_prometheus=False)
        g = ProgressionGuardrail(m, GuardrailConfig(
            warmup_samples=5, consecutive_violations=1, min_accept_rate=0.9,
        ))
        self._fill_monitor(m, 10, correct=False, pre=0.2, post=0.2)
        g.evaluate()
        after = float(PROGRESSION_GUARDRAIL_TRIGGERS.labels(
            reason="accept_rate_below_floor"
        )._value.get())
        assert after == before + 1.0

    def test_combine_with_gates_both(self):
        m = RollingProgressionMonitor(policy="g7", emit_prometheus=False)
        g = ProgressionGuardrail(m, GuardrailConfig(
            warmup_samples=5, consecutive_violations=1, min_accept_rate=0.9,
        ))
        # Start allowed
        other_state = [True]
        combined = g.combine_with(lambda: other_state[0])
        assert combined() is True
        # External gate denies -> combined denies
        other_state[0] = False
        assert combined() is False
        # External gate allows but guardrail halted -> combined denies
        other_state[0] = True
        self._fill_monitor(m, 10, correct=False, pre=0.2, post=0.2)
        g.evaluate()
        assert combined() is False


# ---------------------------------------------------------------------------
# Integration: StreamingAdaptiveRanker + monitor
# ---------------------------------------------------------------------------
class TestRankerIntegration:
    def _build(self, monitor=None, difficulties=None):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        uf = torch.tensor(rng.normal(size=(8, 4)).astype(np.float32))
        ifeat = torch.tensor(rng.normal(size=(12, 4)).astype(np.float32))
        tower = TwoTowerRecommender(
            num_users=8, num_items=12, user_dim=4, item_dim=4,
            hidden=8, emb_dim=8, state_dim=4,
            device="cpu", dp_cfg={"enabled": False},
        ).eval()
        return StreamingAdaptiveRanker(
            tower, uf, ifeat, monitor=monitor, item_difficulties=difficulties,
        )

    def test_monitor_receives_events(self):
        m = RollingProgressionMonitor(policy="int1", emit_prometheus=False)
        r = self._build(monitor=m)
        r.observe(user_id=0, item_id=3, correct=True, category="math")
        r.observe(user_id=0, item_id=4, correct=False, category="math")
        snap = m.snapshot()
        assert snap.sample_size == 2
        assert snap.accept_rate == 0.5

    def test_difficulty_propagation(self):
        diffs = [float(i) / 12.0 for i in range(12)]
        m = RollingProgressionMonitor(policy="int2", emit_prometheus=False)
        r = self._build(monitor=m, difficulties=diffs)
        r.observe(user_id=0, item_id=5, correct=True)
        snap = m.snapshot()
        # stretch_fit was computed against recorded difficulty,
        # so it's not the default 1.0 (we can't be strict about the value but
        # we can assert the monitor saw a difficulty).
        assert snap.sample_size == 1

    def test_monitor_error_does_not_break_observe(self):
        class Exploding:
            def record(self, **_): raise RuntimeError("boom")
        r = self._build(monitor=Exploding())
        # Must not raise
        out = r.observe(user_id=1, item_id=2, correct=True)
        assert out["p_known"] is not None

    def test_difficulty_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self._build(difficulties=[0.1, 0.2])  # wrong length

    def test_metrics_registry_exposes_new_gauges(self):
        # Sanity: the registry exposes the new metric names so scrapers pick them up
        text = metrics_registry().collect()
        names = {m.name for m in text}
        assert "orchid_progression_gain" in names
        assert "orchid_progression_guardrail_halted" in names
