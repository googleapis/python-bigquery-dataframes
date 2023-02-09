import math
from typing import Union
from unittest import mock

import google.api_core.exceptions
import google.auth
import google.cloud.bigquery
import ibis
import ibis.expr.types as ibis_types
import pandas
import pytest
from google.cloud.bigquery.table import TableReference

import bigframes


@pytest.fixture(autouse=True)
def mock_bigquery_client(monkeypatch):
    mock_client = mock.create_autospec(google.cloud.bigquery.Client)
    # Constructor returns the mock itself, so this mock can be treated as the
    # constructor or the instance.
    mock_client.return_value = mock_client
    mock_client.project = "default-project"
    mock_client.get_table = mock_bigquery_client_get_table
    monkeypatch.setattr(google.cloud.bigquery, "Client", mock_client)
    mock_client.reset_mock()

    return mock_client


def mock_bigquery_client_get_table(table_ref: Union[TableReference, str]):
    if isinstance(table_ref, TableReference):
        table_name = table_ref.__str__()
    else:
        table_name = table_ref

    if table_name == "project.dataset.table":
        return google.cloud.bigquery.Table(
            table_name, [{"mode": "NULLABLE", "name": "int64_col", "type": "INTEGER"}]
        )
    elif table_name == "default-project.dataset.table":
        return google.cloud.bigquery.Table(table_name)
    else:
        raise google.api_core.exceptions.NotFound("Not Found Table")


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
