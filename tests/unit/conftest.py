import math
from typing import Mapping, Optional, Union
from unittest import mock

import google.api_core.exceptions
import google.auth
import google.cloud.bigquery as bigquery
import google.cloud.bigquery.table
import google.oauth2.credentials  # type: ignore
import ibis.expr.types as ibis_types
import pandas
import pytest

import bigframes
import bigframes.core

SCALARS_TABLE_ID = "project.dataset.scalars_table"


@pytest.fixture(autouse=True)
def mock_bigquery_client(
    monkeypatch, scalars_pandas_df: pandas.DataFrame
) -> bigquery.Client:
    mock_client = mock.create_autospec(bigquery.Client)
    # Constructor returns the mock itself, so this mock can be treated as the
    # constructor or the instance.
    mock_client.return_value = mock_client
    mock_client.project = "default-project"
    mock_client.get_table = mock_bigquery_client_get_table

    def mock_bigquery_client_query(
        sql: str, job_config: Optional[bigquery.QueryJobConfig] = None
    ) -> bigquery.QueryJob:
        def mock_result(max_results=None):
            mock_job = mock.create_autospec(bigquery.QueryJob)
            mock_job.total_rows = len(scalars_pandas_df.index)
            mock_job.schema = [
                bigquery.SchemaField(name=name, field_type="INT64")
                for name in scalars_pandas_df.columns
            ]
            # Use scalars_pandas_df instead of ibis_expr.execute() to preserve dtypes.
            mock_job.to_dataframe.return_value = scalars_pandas_df.head(n=max_results)
            return mock_job

        mock_job = mock.create_autospec(bigquery.QueryJob)
        mock_job.result = mock_result
        return mock_job

    mock_client.query = mock_bigquery_client_query
    monkeypatch.setattr(bigquery, "Client", mock_client)
    mock_client.reset_mock()
    return mock_client


def mock_bigquery_client_get_table(
    table_ref: Union[google.cloud.bigquery.table.TableReference, str]
):
    if isinstance(table_ref, google.cloud.bigquery.table.TableReference):
        table_name = table_ref.__str__()
    else:
        table_name = table_ref

    if table_name == "project.dataset.table":
        return bigquery.Table(
            table_name, [{"mode": "NULLABLE", "name": "int64_col", "type": "INTEGER"}]
        )
    elif table_name == "default-project.dataset.table":
        return bigquery.Table(table_name)
    elif table_name == SCALARS_TABLE_ID:
        return bigquery.Table(
            table_name,
            [
                {"mode": "NULLABLE", "name": "bool_col", "type": "BOOL"},
                {"mode": "NULLABLE", "name": "int64_col", "type": "INTEGER"},
                {"mode": "NULLABLE", "name": "float64_col", "type": "FLOAT"},
                {"mode": "NULLABLE", "name": "string_col", "type": "STRING"},
            ],
        )
    else:
        raise google.api_core.exceptions.NotFound("Not Found Table")


@pytest.fixture
def session() -> bigframes.Session:
    return bigframes.Session(
        context=bigframes.Context(
            credentials=mock.create_autospec(google.oauth2.credentials.Credentials),
            project="unit-test-project",
        )
    )


@pytest.fixture
def scalars_df(session) -> bigframes.DataFrame:
    return session.read_gbq(SCALARS_TABLE_ID)


@pytest.fixture
def session_tables(scalars_pandas_df) -> Mapping[str, pandas.DataFrame]:
    return {
        SCALARS_TABLE_ID: scalars_pandas_df,
    }


@pytest.fixture
def scalars_pandas_df() -> pandas.DataFrame:
    # Note: as of 2023-02-07, using nullable dtypes with the ibis pandas
    # backend requires running ibis at HEAD. See:
    # https://github.com/ibis-project/ibis/pull/5345
    return pandas.DataFrame(
        {
            "bool_col": pandas.Series(
                [
                    True,
                    None,
                    False,
                    True,
                    None,
                    False,
                    True,
                    None,
                    False,
                    True,
                ],
                dtype="boolean",
            ),
            "int64_col": pandas.Series(
                [
                    1,
                    2,
                    3,
                    None,
                    0,
                    -1,
                    -2,
                    2**63 - 1,
                    -(2**63),
                    None,
                ],
                dtype="Int64",
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
def scalars_ibis_table(session) -> ibis_types.Table:
    return session.ibis_client.table(SCALARS_TABLE_ID)
