"""Tests for the experimental SAKT-style knowledge tracer."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from orchid_ranker.kt import (
    AKTTracer,
    DKTTracer,
    SAINTPlusTracer,
    SAINTTracer,
    SAKTTracer,
    build_sakt_examples,
)


def _small_learning_events() -> pd.DataFrame:
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


def test_build_sakt_examples_uses_only_prior_events():
    events = pd.DataFrame(
        {
            "user_id": [7, 7, 7, 7],
            "item_id": [10, 20, 30, 40],
            "correct": [1, 0, 1, 0],
            "timestamp": [1, 2, 3, 4],
        }
    )

    examples = build_sakt_examples(events, timestamp_col="timestamp", max_seq_len=2)

    assert len(examples) == 3
    assert examples[0].query_item_id == 20
    assert examples[0].history_item_ids == (10,)
    assert examples[1].query_item_id == 30
    assert examples[1].history_item_ids == (10, 20)
    assert examples[2].query_item_id == 40
    assert examples[2].history_item_ids == (20, 30)
    assert 40 not in examples[2].history_item_ids


def test_build_sakt_examples_includes_temporal_features_when_timestamped():
    events = pd.DataFrame(
        {
            "user_id": [7, 7, 7],
            "item_id": [10, 20, 30],
            "correct": [1, 0, 1],
            "timestamp": [10, 15, 30],
        }
    )

    examples = build_sakt_examples(events, timestamp_col="timestamp", max_seq_len=3)

    assert examples[1].query_item_id == 30
    assert examples[1].history_elapsed == (0.0, 5.0)
    assert examples[1].history_lag == (20.0, 15.0)


def test_sakt_tracer_fit_predict_and_state_vector_shape():
    tracer = SAKTTracer(
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=3,
        device="cpu",
    ).fit(_small_learning_events(), timestamp_col="timestamp")

    prob = tracer.predict_correct(1, 30)
    state = tracer.state_vector(1, [10, 20, 30])

    assert tracer.is_fitted
    assert 0.0 <= prob <= 1.0
    assert state.shape == (3,)
    assert np.all((0.0 <= state) & (state <= 1.0))
    assert tracer.result_["num_examples"] > 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"epochs": 0}, "epochs"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"correct_threshold": 1.5}, "correct_threshold"),
    ],
)
def test_sakt_tracer_rejects_invalid_training_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SAKTTracer(max_seq_len=3, d_model=16, n_heads=2, device="cpu", **kwargs)


def test_sakt_tracer_unknown_user_gets_cold_state():
    tracer = SAKTTracer(
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=5,
        device="cpu",
    ).fit(_small_learning_events(), timestamp_col="timestamp")

    state = tracer.state_vector("new-learner", [10, 20])

    assert state.shape == (2,)
    assert np.all((0.0 <= state) & (state <= 1.0))


def test_sakt_observe_updates_history_and_prediction_context():
    tracer = SAKTTracer(
        max_seq_len=4,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=7,
        device="cpu",
    ).fit(_small_learning_events(), timestamp_col="timestamp")

    before = tracer.state_vector("live-user", [30, 40])
    length = tracer.observe("live-user", 20, correct=True)
    after = tracer.state_vector("live-user", [30, 40])

    assert length == 1
    assert tracer.history_for("live-user") == [(20, 1)]
    assert not np.allclose(before, after)


def test_sakt_recommend_practice_prefers_target_correctness():
    tracer = SAKTTracer(
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=11,
        device="cpu",
    ).fit(_small_learning_events(), timestamp_col="timestamp")

    recs = tracer.recommend_practice(2, [10, 20, 30, 40, 50], top_k=3, target_correct=0.7)

    assert len(recs) == 3
    assert all(0.0 <= rec.p_correct <= 1.0 for rec in recs)
    assert recs == sorted(recs, key=lambda rec: (rec.score, rec.p_correct, str(rec.item_id)), reverse=True)


def test_sakt_predict_unknown_item_raises_key_error():
    tracer = SAKTTracer(
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=13,
        device="cpu",
    ).fit(_small_learning_events(), timestamp_col="timestamp")

    with pytest.raises(KeyError):
        tracer.predict_correct(1, 999)


def test_akt_tracer_fit_predict_with_item_difficulty_column():
    tracer = AKTTracer(
        max_seq_len=3,
        d_model=16,
        epochs=1,
        batch_size=4,
        random_state=21,
        device="cpu",
    ).fit(_small_learning_events(), timestamp_col="timestamp", item_difficulty_col="difficulty")

    prob = tracer.predict_correct(1, 30)
    state = tracer.state_vector(1, [10, 20, 30])

    assert tracer.is_fitted
    assert 0.0 <= prob <= 1.0
    assert state.shape == (3,)
    assert tracer.item_difficulty_[30] == 0.55


def test_akt_tracer_rejects_invalid_difficulty():
    events = _small_learning_events()
    events.loc[events["item_id"] == 30, "difficulty"] = 1.5
    tracer = AKTTracer(max_seq_len=3, d_model=16, epochs=1, batch_size=4, device="cpu")

    with pytest.raises(ValueError):
        tracer.fit(events, timestamp_col="timestamp", item_difficulty_col="difficulty")


@pytest.mark.parametrize("tracer_cls", [SAINTTracer, SAINTPlusTracer])
def test_saint_tracers_fit_predict_and_update(tracer_cls):
    tracer = tracer_cls(
        max_seq_len=3,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=31,
        device="cpu",
    ).fit(_small_learning_events(), timestamp_col="timestamp")

    before = tracer.state_vector("live-saint", [20, 30])
    with warnings.catch_warnings():
        # SAINTPlusTracer fit with timestamps intentionally warns when observe()
        # is called without one (see FIX 3). That behavior is exercised directly
        # elsewhere; here we only care about prediction/update mechanics.
        warnings.simplefilter("ignore", UserWarning)
        length = tracer.observe("live-saint", 10, correct=True)
    after = tracer.state_vector("live-saint", [20, 30])

    assert tracer.is_fitted
    assert 0.0 <= tracer.predict_correct(1, 30) <= 1.0
    assert length == 1
    assert before.shape == after.shape == (2,)
    assert not np.allclose(before, after)


def test_dkt_short_history_prediction_depends_on_last_response():
    """Regression test for DKT reading the wrong (left-padded) hidden state.

    Sequences are left-padded, so the most recent interaction lives at the final
    timestep. The old code read ``outputs[:, n-1]`` where ``n`` is the history
    length, which lands in the leading padding region for histories shorter than
    ``max_seq_len`` -- so the latest response had no effect on the prediction.
    With ``max_seq_len`` deliberately larger than the live history, flipping the
    most recent response must change the predicted probability.
    """
    tracer = DKTTracer(
        max_seq_len=8,
        d_model=16,
        epochs=3,
        batch_size=4,
        random_state=17,
        device="cpu",
    ).fit(_small_learning_events(), timestamp_col="timestamp")

    # Two live users with identical short histories except the final response.
    # The history (length 2) is far shorter than max_seq_len (8), so the buggy
    # indexing would read padding and ignore the last interaction entirely.
    for user_id in ("dkt-pos", "dkt-neg"):
        tracer.observe(user_id, 10, correct=True)
    tracer.observe("dkt-pos", 20, correct=True)
    tracer.observe("dkt-neg", 20, correct=False)

    prob_pos = tracer.predict_correct("dkt-pos", 30)
    prob_neg = tracer.predict_correct("dkt-neg", 30)

    assert 0.0 <= prob_pos <= 1.0
    assert 0.0 <= prob_neg <= 1.0
    # The only difference is the most recent response; it must influence the
    # prediction. This assertion fails with the pre-fix padding-region indexing.
    assert abs(prob_pos - prob_neg) > 1e-4


def _fit_saintplus_with_timestamps(random_state: int = 41) -> SAINTPlusTracer:
    return SAINTPlusTracer(
        max_seq_len=4,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=random_state,
        device="cpu",
    ).fit(_small_learning_events(), timestamp_col="timestamp")


def test_saintplus_observe_without_timestamp_warns_when_fit_with_timestamps():
    """Regression: warn that synthesized time gaps may not match training."""
    tracer = _fit_saintplus_with_timestamps()

    with pytest.warns(UserWarning, match="timestamp"):
        tracer.observe("live-user", 10, correct=True)

    # The warning fires at most once per instance.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # No exception means no warning was raised on the second call.
        tracer.observe("live-user", 20, correct=False)


def test_saintplus_observe_with_timestamp_does_not_warn():
    """Passing a real timestamp keeps temporal features consistent: no warning."""
    tracer = _fit_saintplus_with_timestamps(random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracer.observe("live-user", 10, correct=True, timestamp=5)
        tracer.observe("live-user", 20, correct=False, timestamp=9)


def test_saintplus_observe_without_timestamps_when_fit_without_timestamps_is_silent():
    """If fit without timestamps, the missing-timestamp warning must not fire."""
    tracer = SAINTPlusTracer(
        max_seq_len=4,
        d_model=16,
        n_heads=2,
        epochs=1,
        batch_size=4,
        random_state=43,
        device="cpu",
    ).fit(_small_learning_events())  # no timestamp_col

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracer.observe("live-user", 10, correct=True)
