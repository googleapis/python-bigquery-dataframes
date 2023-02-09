from unittest import mock

import google.cloud.bigquery
import pytest

import bigframes


@pytest.fixture(autouse=True)
def mock_bigquery_client(monkeypatch):
    mock_client = mock.create_autospec(google.cloud.bigquery.Client)
    # Constructor returns the mock itself, so this mock can be treated as the
    # constructor or the instance.
    mock_client.return_value = mock_client
    mock_client.project = "test-project"
    monkeypatch.setattr(google.cloud.bigquery, "Client", mock_client)
    mock_client.reset_mock()

    return mock_client


@pytest.fixture
def session(mock_bigquery_client) -> bigframes.Session:
    return bigframes.Session()
