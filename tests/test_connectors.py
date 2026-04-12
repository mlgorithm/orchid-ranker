import pytest

from orchid_ranker import (
    SnowflakeConnector,
    BigQueryConnector,
    S3StreamConnector,
    MLflowTracker,
)


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (SnowflakeConnector, {"account": "acct", "user": "user", "password": "pwd"}),
        (BigQueryConnector, {}),
        (S3StreamConnector, {"bucket": "demo"}),
        (MLflowTracker, {}),
    ],
)
def test_optional_connectors_raise_when_dependency_missing(factory, kwargs):
    connector = factory(**kwargs)
    # Some connector deps (e.g. boto3 for S3) may be installed in test env;
    # allow ImportError (dep missing) or other exceptions (dep present but unconfigured).
    with pytest.raises(Exception):
        if isinstance(connector, SnowflakeConnector):
            connector.execute("SELECT 1")
        elif isinstance(connector, BigQueryConnector):
            connector.query_dataframe("SELECT 1")
        elif isinstance(connector, S3StreamConnector):
            list(connector.list_objects())
        else:
            connector.log_params({"alpha": 1})


def test_s3_transient_classifier_uses_response_codes():
    class FakeError(Exception):
        pass

    exc = FakeError("slow down")
    exc.response = {"Error": {"Code": "SlowDown"}}
    assert S3StreamConnector._is_transient_error(exc) is True


def test_s3_transient_classifier_uses_exception_type_name():
    EndpointConnectionError = type("EndpointConnectionError", (Exception,), {})
    exc = EndpointConnectionError("connection lost")
    assert S3StreamConnector._is_transient_error(exc) is True


def test_mlflow_tracker_applies_tracking_context(monkeypatch):
    import orchid_ranker.connectors.mlflow as mlflow_mod

    class FakeMLflow:
        def __init__(self):
            self.calls = []

        def set_tracking_uri(self, uri):
            self.calls.append(("uri", uri))

        def set_experiment(self, experiment):
            self.calls.append(("experiment", experiment))

        def log_params(self, params):
            self.calls.append(("params", dict(params)))

    fake = FakeMLflow()
    monkeypatch.setattr(mlflow_mod, "mlflow", fake)

    tracker = MLflowTracker(experiment="exp", tracking_uri="memory://mlflow")
    tracker.log_params({"alpha": 1})

    assert fake.calls == [
        ("uri", "memory://mlflow"),
        ("experiment", "exp"),
        ("params", {"alpha": 1}),
    ]
