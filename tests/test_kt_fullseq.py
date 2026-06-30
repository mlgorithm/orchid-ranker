"""Mechanism and safety tests for the full-sequence KT rewrite (phase 1).

These tests assert the *mechanism* of the full-sequence migration -- causal
no-leakage, per-position output, recurrence, and serving consistency -- not just
that training runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from orchid_ranker.kt import DKTTracer, SAKTTracer


def _events() -> pd.DataFrame:
    rows = []
    for user_id, ability in [(1, 0.25), (2, 0.45), (3, 0.65), (4, 0.85)]:
        for step, (item_id, difficulty) in enumerate(
            [(10, 0.20), (20, 0.35), (30, 0.55), (40, 0.70), (50, 0.85)]
        ):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "correct": int(ability + 0.15 >= difficulty),
                    "timestamp": step,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("tracer_cls", [SAKTTracer, DKTTracer])
def test_no_leakage_flipping_own_response_does_not_change_prediction(tracer_cls):
    """Core correctness guarantee: the prediction at position ``t`` must NOT
    depend on the response at ``t`` (the model never sees its own answer).

    For each position ``t`` we flip ONLY ``correct[t]`` and re-run. The logit at
    position ``t`` must be unchanged -- it is causally restricted to interactions
    strictly before ``t``. (Predictions at positions > t legitimately CAN change,
    since later positions are allowed to see interaction ``t``'s response.)
    """
    tracer = tracer_cls(
        max_seq_len=6,
        d_model=16,
        n_heads=2,
        epochs=2,
        batch_size=4,
        random_state=7,
        device="cpu",
    ).fit(_events(), timestamp_col="timestamp")
    model = tracer.model
    assert model is not None
    model.eval()

    # A right-padded length-4 sequence (positions 4,5 are padding).
    items = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
    correct = torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        baseline = model(items, correct, items)
    assert baseline.shape == (1, 6)

    for t in range(4):  # real positions only
        flipped = correct.clone()
        flipped[0, t] = 1 - flipped[0, t]
        with torch.no_grad():
            out = model(items, flipped, items)
        assert out[0, t] == pytest.approx(float(baseline[0, t]), abs=1e-6), (
            f"prediction at position {t} depends on its own response -> leakage"
        )


@pytest.mark.parametrize("tracer_cls", [SAKTTracer, DKTTracer])
def test_model_produces_prediction_for_every_position(tracer_cls):
    """One forward pass yields a logit for every sequence position (B, L)."""
    tracer = tracer_cls(
        max_seq_len=5,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=3,
        device="cpu",
    ).fit(_events(), timestamp_col="timestamp")
    model = tracer.model
    assert model is not None
    model.eval()

    batch = 3
    items = torch.randint(1, 6, (batch, 5))
    correct = torch.randint(0, 2, (batch, 5))
    with torch.no_grad():
        out = model(items, correct, items)
    assert out.shape == (batch, 5)


def test_dkt_module_is_recurrent():
    """DKT must be recurrent: its module contains an nn.LSTM (paper fidelity)."""
    tracer = DKTTracer(
        max_seq_len=5, d_model=16, epochs=1, batch_size=4, random_state=5, device="cpu"
    ).fit(_events(), timestamp_col="timestamp")
    model = tracer.model
    assert model is not None
    assert any(isinstance(m, torch.nn.LSTM) for m in model.modules())


@pytest.mark.parametrize("tracer_cls", [SAKTTracer, DKTTracer])
def test_serving_consistency_and_determinism(tracer_cls):
    """predict_many returns probabilities in [0,1] for all candidates and is
    deterministic given the same random_state."""
    candidates = [10, 20, 30, 40, 50]

    def fit_and_predict():
        tracer = tracer_cls(
            max_seq_len=5,
            d_model=16,
            n_heads=2,
            epochs=2,
            batch_size=4,
            random_state=123,
            device="cpu",
        ).fit(_events(), timestamp_col="timestamp")
        tracer.observe("live", 10, correct=True)
        tracer.observe("live", 20, correct=False)
        return tracer.predict_many("live", candidates)

    first = fit_and_predict()
    second = fit_and_predict()

    assert set(first) == set(candidates)
    assert all(0.0 <= p <= 1.0 for p in first.values())
    for item_id in candidates:
        assert first[item_id] == pytest.approx(second[item_id], abs=1e-6)
