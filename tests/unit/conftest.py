import math
from unittest import mock

import google.auth
import google.cloud.bigquery
import ibis
import ibis.expr.types as ibis_types
import pandas
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


@pytest.fixture
def scalars_pandas_df() -> pandas.DataFrame:
    # Note: as of 2023-02-07, using nullable dtypes with the ibis pandas
    # backend requires running ibis at HEAD. See:
    # https://github.com/ibis-project/ibis/pull/5345
    return pandas.DataFrame(
        {
            "bool_col": pandas.Series(
                [True, None, False, True, None, False, True, None, False, True],
                dtype="boolean",
            ),
            "int64_col": pandas.Series(
                [1, 2, 3, None, 0, -1, -2, 2**63 - 1, -(2**63), None], dtype="Int64"
            ),
            "float64_col": pandas.Series(
                [
                    None,
                    1,
                    math.pi,
                    math.e * 1e10,
                    0,
                    float("nan"),
                    float("inf"),
                    float("-inf"),
                    -2.23e-308,
                    1.8e308,
                ],
                dtype="Float64",
            ),
            "string_col": pandas.Series(
                [
                    "abc",
                    "XYZ",
                    "aBcDeFgHiJkLmNoPqRsTuVwXyZ",
                    "1_2-3+4=5~6*7/8&9%10#11@12$" "",
                    None,
                    "こんにちは",
                    "你好",
                    "வணக்கம்",
                    "שלום",
                ],
                dtype="string[pyarrow]",
            ),
        }
    )


@pytest.fixture
def scalars_ibis_table(scalars_pandas_df) -> ibis_types.Table:
    ibis_engine = ibis.pandas.connect(
        dictionary={
            "scalars_table": scalars_pandas_df,
        }
    )
    return ibis_engine.table("scalars_table")
