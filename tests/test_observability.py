from orchid_ranker import record_training, export_metrics, metrics_content_type, metrics_registry


def test_record_training_updates_metrics():
    registry = metrics_registry()
    initial = export_metrics()
    record_training(2.5, epsilon=0.8)
    payload = export_metrics()
    assert payload != initial
    text = payload.decode("utf-8")
    assert "orchid_training_runs_total" in text
    assert "orchid_dp_epsilon_cumulative" in text
    assert metrics_content_type().startswith("text/plain")

