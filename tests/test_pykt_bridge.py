from __future__ import annotations

import pandas as pd
import pytest

from orchid_ranker.learning_policy import KTValuePolicy
from orchid_ranker.pykt_bridge import (
    PyKTPredictionAdapter,
    export_pykt_sequences,
    load_pykt_sequences,
    pykt_sequences_to_interactions,
)


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "item_id": ["q1", "q2", "q3", "q1", "q2", "q3"],
            "concept_id": ["c1", "c1", "c2", "c1", "c1", "c2"],
            "correct": [1, 0, 1, 0, 1, 1],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "duration": [10, 11, 12, 13, 14, 15],
        }
    )


def test_export_and_load_pykt_sequences_round_trip(tmp_path):
    path = tmp_path / "pykt_sequences.txt"
    written = export_pykt_sequences(
        _interactions(),
        path,
        concept_col="concept_id",
        timestamp_col="timestamp",
        duration_col="duration",
    )
    loaded = load_pykt_sequences(path)

    assert len(written) == 2
    assert loaded == written
    assert loaded[0].user_id == "u1"
    assert loaded[0].questions == ("q1", "q2", "q3")
    assert loaded[0].concepts == ("c1", "c1", "c2")
    assert loaded[0].responses == (1, 0, 1)
    assert path.read_text().splitlines()[0] == "u1,3"


def test_pykt_sequences_to_interactions_returns_orchid_schema(tmp_path):
    path = tmp_path / "pykt_sequences.txt"
    export_pykt_sequences(_interactions(), path, concept_col="concept_id", timestamp_col="timestamp")

    frame = pykt_sequences_to_interactions(load_pykt_sequences(path))

    assert list(frame.columns) == ["user_id", "item_id", "correct", "timestamp", "concept_id", "duration"]
    assert len(frame) == 6
    assert frame["correct"].sum() == 4


def test_export_pykt_sequences_filters_short_sequences(tmp_path):
    data = _interactions().query("user_id == 'u1'").head(1)
    path = tmp_path / "pykt_sequences.txt"

    sequences = export_pykt_sequences(data, path, min_seq_len=2)

    assert sequences == []
    assert path.read_text() == ""


def test_pykt_prediction_adapter_feeds_kt_value_policy():
    predictions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "item_id": ["q1", "q2", "q3", "q1"],
            "p_correct": [0.95, 0.69, 0.35, 0.4],
        }
    )
    adapter = PyKTPredictionAdapter(predictions, fallback="item_mean")
    policy = KTValuePolicy(adapter, target_correct=0.70)

    recs = policy.rank("u1", ["q1", "q2", "q3"], top_k=3)
    fallback = adapter.predict_correct("new-user", "q1")

    assert recs[0].item_id == "q2"
    assert fallback == pytest.approx((0.95 + 0.4) / 2)


def test_pykt_prediction_adapter_raise_fallback():
    adapter = PyKTPredictionAdapter(
        pd.DataFrame({"user_id": ["u1"], "item_id": ["q1"], "p_correct": [0.5]}),
        fallback="raise",
    )

    with pytest.raises(KeyError):
        adapter.predict_correct("u2", "q1")
