"""Mechanism and safety tests for the full-sequence KT rewrite (phase 1).

These tests assert the *mechanism* of the full-sequence migration -- causal
no-leakage, per-position output, recurrence, and serving consistency -- not just
that training runs.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from orchid_ranker.kt import (
    AKTTracer,
    DKTTracer,
    DKVMNTracer,
    SAINTPlusTracer,
    SAINTTracer,
    SAKTTracer,
)


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
                    "difficulty": difficulty,
                    "timestamp": step,
                }
            )
    return pd.DataFrame(rows)


def _fit_tracer(tracer_cls, **kwargs):
    """Fit any KT tracer on ``_events`` with the inputs each model needs.

    AKT additionally consumes an item-difficulty column (warm-starting the Rasch
    scalar); SAINT+ consumes timestamps for its decoder time features. All other
    tracers ignore the extras.
    """
    tracer = tracer_cls(epochs=2, batch_size=4, random_state=7, device="cpu", **kwargs)
    if tracer_cls is AKTTracer:
        return tracer.fit(_events(), timestamp_col="timestamp", item_difficulty_col="difficulty")
    return tracer.fit(_events(), timestamp_col="timestamp")


_ALL_TRACERS = [
    SAKTTracer,
    DKTTracer,
    AKTTracer,
    SAINTTracer,
    SAINTPlusTracer,
    DKVMNTracer,
]


@pytest.mark.parametrize("tracer_cls", _ALL_TRACERS)
def test_no_leakage_all_models(tracer_cls):
    """No-leakage gate for every full-sequence model: flipping ``correct[t]``
    must not change the logit at position ``t`` (AKT, SAINT, SAINT+, DKVMN
    included alongside the phase-1 SAKT/DKT)."""
    tracer = _fit_tracer(tracer_cls, max_seq_len=6, d_model=16)
    model = tracer.model
    assert model is not None
    model.eval()

    items = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
    correct = torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.long)

    def run(corr):
        # SAINT+ forward takes optional elapsed/lag; default zeros are fine here.
        with torch.no_grad():
            return model(items, corr, items)

    baseline = run(correct)
    assert baseline.shape == (1, 6)
    for t in range(4):
        flipped = correct.clone()
        flipped[0, t] = 1 - flipped[0, t]
        out = run(flipped)
        assert out[0, t] == pytest.approx(float(baseline[0, t]), abs=1e-5), (
            f"{tracer_cls.__name__} prediction at position {t} depends on its own "
            f"response -> leakage"
        )


def test_akt_has_rasch_difficulty_and_learned_decay():
    """AKT mechanism: learned Rasch difficulty (num_items, 1), learned per-head
    decay (gammas), and a CONTENT-dependent (not position-only) attention."""
    tracer = _fit_tracer(AKTTracer, max_seq_len=6, d_model=16, n_heads=2)
    model = tracer.model
    assert model is not None

    # Padding plus one reserved OOV embedding are included with learned item
    # difficulties, so registered catalog items can receive live feedback.
    num_items = len(tracer.item_ids_)
    assert model.difficulty.weight.shape == (num_items + 2, 1)
    assert model.difficulty.weight.requires_grad
    # Learned per-head decay parameter.
    assert isinstance(model.gammas, torch.nn.Parameter)
    assert model.gammas.shape[0] == model.n_heads
    assert model.gammas.requires_grad

    # Content-dependence: two inputs with IDENTICAL exercise positions but
    # DIFFERENT interaction content must give different attention/output at a
    # fixed late query position (a pure positional ramp could not).
    model.eval()
    items = torch.tensor([[1, 2, 3, 4, 5, 0]], dtype=torch.long)
    correct_a = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    correct_b = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        out_a = model(items, correct_a, items)
        out_b = model(items, correct_b, items)
    # Query at position 4 attends to interactions 0..3, whose content differs.
    assert abs(float(out_a[0, 4]) - float(out_b[0, 4])) > 1e-5


@pytest.mark.parametrize("tracer_cls", _ALL_TRACERS)
def test_reserved_oov_embeddings_use_learned_mean_priors(tracer_cls):
    """Catalog OOV rows must not retain untrained random initialization."""
    tracer = _fit_tracer(tracer_cls, max_seq_len=6, d_model=16)
    model = tracer.model
    assert model is not None
    unknown = tracer._unknown_item_idx
    num_items = tracer._model_num_items
    item_embedding_names = {
        "interaction_emb",
        "query_item_emb",
        "concept_emb",
        "variation_emb",
        "difficulty",
        "response_emb",
        "response_variation_emb",
        "exercise_emb",
        "exercise_key_emb",
        "interaction_value_emb",
    }

    checked = 0
    for name, module in model.named_modules():
        if name not in item_embedding_names or not isinstance(module, torch.nn.Embedding):
            continue
        if module.num_embeddings == num_items + 1:
            assert torch.allclose(module.weight[unknown], module.weight[1:unknown].mean(dim=0))
            checked += 1
        elif module.num_embeddings == 2 * num_items + 1:
            assert torch.allclose(module.weight[unknown], module.weight[1:unknown].mean(dim=0))
            response_unknown = num_items + unknown
            assert torch.allclose(
                module.weight[response_unknown], module.weight[num_items + 1 : response_unknown].mean(dim=0)
            )
            checked += 1
    assert checked > 0


def test_dkvmn_has_key_value_memory_that_evolves():
    """DKVMN mechanism: key matrix (N, d_k), value-memory matrix (N, d_v), and
    a working value memory that mutates as timesteps are processed."""
    tracer = _fit_tracer(DKVMNTracer, max_seq_len=6, d_model=16)
    model = tracer.model
    assert model is not None

    assert isinstance(model.key_matrix, torch.nn.Parameter)
    assert model.key_matrix.shape == (model.memory_size, model.d_k)
    assert isinstance(model.value_matrix_init, torch.nn.Parameter)
    assert model.value_matrix_init.shape == (model.memory_size, model.d_v)

    # Replicate the forward write loop and confirm the working memory evolves.
    model.eval()
    items = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    correct = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
    interaction_codes = items + correct * model.num_items
    write_values = model.interaction_value_emb(interaction_codes)
    q_keys = model.exercise_key_emb(items)
    mv = model.value_matrix_init.unsqueeze(0).clone()
    snapshots = [mv.clone()]
    with torch.no_grad():
        for t in range(items.shape[1]):
            w = torch.softmax(q_keys[:, t, :] @ model.key_matrix.t(), dim=-1)
            v = write_values[:, t, :]
            e = torch.sigmoid(model.erase(v))
            a = torch.tanh(model.add(v))
            wc = w.unsqueeze(-1)
            mv = mv * (1.0 - wc * e.unsqueeze(1)) + wc * a.unsqueeze(1)
            snapshots.append(mv.clone())
    # State after processing all interactions differs from the initial memory.
    assert not torch.allclose(snapshots[0], snapshots[-1])


def test_saint_has_distinct_encoder_and_decoder():
    """SAINT routing: distinct encoder (exercise stream) and decoder (response
    stream) submodules, both present and used."""
    tracer = _fit_tracer(SAINTTracer, max_seq_len=6, d_model=16, n_heads=2)
    model = tracer.model
    assert model is not None
    assert isinstance(model.encoder, torch.nn.TransformerEncoder)
    assert isinstance(model.decoder, torch.nn.TransformerDecoder)
    assert model.encoder is not model.decoder


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
