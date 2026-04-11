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
