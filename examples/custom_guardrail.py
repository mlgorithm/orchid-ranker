#!/usr/bin/env python3
"""Safety pattern: monitor + guardrail + frozen fallback.

Run with: python examples/custom_guardrail.py
Requires: pip install orchid-ranker[ml]
"""
import numpy as np
import pandas as pd
from orchid_ranker import OrchidRecommender
from orchid_ranker.live_metrics import (
    GuardrailConfig, ProgressionGuardrail, RollingProgressionMonitor,
)

# --- Fit a baseline recommender ---
rng = np.random.default_rng(99)
interactions = pd.DataFrame({
    "user_id": rng.integers(0, 30, 1500),
    "item_id": rng.integers(0, 60, 1500),
})
rec = OrchidRecommender.from_interactions(interactions, strategy="als")

# --- Set up progression monitor and guardrail ---
monitor = RollingProgressionMonitor(
    window_size=500, total_categories={"algebra", "geometry", "calculus"},
    emit_prometheus=False,
)
guardrail = ProgressionGuardrail(monitor, cfg=GuardrailConfig(
    min_progression_gain=0.0, min_accept_rate=0.3,
    warmup_samples=10, consecutive_violations=2,
))

# --- Simulate declining outcomes (user struggling) ---
for i in range(20):
    monitor.record(
        user_id=0, item_id=i, correct=False, category="algebra",
        pre_competence=0.3, post_competence=0.28, difficulty=0.8,
    )

# --- Check the guardrail before serving ---
allowed = guardrail.evaluate()
print(f"Adaptive allowed: {allowed}")
print(f"Halted: {guardrail.is_halted}  Reason: {guardrail.halt_reason}")

if not guardrail.should_allow_adaptive():
    print("\nGuardrail fired -- falling back to frozen baseline:")
    fallback = rec.baseline_rank(user_id=0, top_k=5)
    for r in fallback:
        print(f"  item {r.item_id}: score {r.score:.4f}")
else:
    print("\nAdaptive policy is clear to serve.")
