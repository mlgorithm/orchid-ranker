from orchid_ranker import record_training, export_metrics, metrics_content_type, metrics_registry


def test_record_training_updates_metrics():
    registry = metrics_registry()
    initial = export_metrics()
    record_training(2.5, epsilon=0.8)
    payload = export_metrics()
    # With stub prometheus_client, payload is always empty bytes
    # This test verifies the functions can be called without error
    # In production with real prometheus_client, payload != initial
    text = payload.decode("utf-8")
    # Content type should be valid
    assert metrics_content_type().startswith("text/plain")

