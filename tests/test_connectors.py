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
    with pytest.raises(ImportError):
        if isinstance(connector, SnowflakeConnector):
            connector.execute("SELECT 1")
        elif isinstance(connector, BigQueryConnector):
            connector.query_dataframe("SELECT 1")
        elif isinstance(connector, S3StreamConnector):
            list(connector.list_objects())
        else:
            connector.log_params({"alpha": 1})
