# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Tuple

import db_dtypes  # type:ignore
import google.api_core.exceptions
import pandas as pd
import pandas.testing
import pyarrow as pa
import pytest

from tests.system import utils

try:
    import pandas_gbq  # type: ignore
except ImportError:  # pragma: NO COVER
    # TODO(b/332758806): Run system tests without "extras"
    pandas_gbq = None

import typing

import bigframes
import bigframes.dataframe
import bigframes.features
import bigframes.pandas as bpd


def test_sql_executes(scalars_df_default_index, bigquery_client):
    """Test that DataFrame.sql returns executable SQL.

    DF.sql is used in public documentation such as
    https://cloud.google.com/blog/products/data-analytics/using-bigquery-dataframes-with-carto-geospatial-tools
    as a way to pass a DataFrame on to carto without executing the SQL
    immediately.

    Make sure that this SQL can be run outside of BigQuery DataFrames (assuming
    similar credentials / access to the referenced tables).
    """
    # Do some operations to make for more complex SQL.
    df = (
        scalars_df_default_index.drop(columns=["geography_col"])
        .groupby("string_col")
        .max()
    )
    df.index.name = None  # Don't include unnamed indexes.
    query = df.sql

    bf_result = df.to_pandas().sort_values("rowindex").reset_index(drop=True)
    bq_result = (
        bigquery_client.query_and_wait(query)
        .to_dataframe()
        .sort_values("rowindex")
        .reset_index(drop=True)
    )
    pandas.testing.assert_frame_equal(bf_result, bq_result, check_dtype=False)


def test_sql_executes_and_includes_named_index(
    scalars_df_default_index, bigquery_client
):
    """Test that DataFrame.sql returns executable SQL.

    DF.sql is used in public documentation such as
    https://cloud.google.com/blog/products/data-analytics/using-bigquery-dataframes-with-carto-geospatial-tools
    as a way to pass a DataFrame on to carto without executing the SQL
    immediately.

    Make sure that this SQL can be run outside of BigQuery DataFrames (assuming
    similar credentials / access to the referenced tables).
    """
    # Do some operations to make for more complex SQL.
    df = (
        scalars_df_default_index.drop(columns=["geography_col"])
        .groupby("string_col")
        .max()
    )
    query = df.sql

    bf_result = df.to_pandas().sort_values("rowindex")
    bq_result = (
        bigquery_client.query_and_wait(query)
        .to_dataframe()
        .set_index("string_col")
        .sort_values("rowindex")
    )
    pandas.testing.assert_frame_equal(
        bf_result, bq_result, check_dtype=False, check_index_type=False
    )


def test_sql_executes_and_includes_named_multiindex(
    scalars_df_default_index, bigquery_client
):
    """Test that DataFrame.sql returns executable SQL.

    DF.sql is used in public documentation such as
    https://cloud.google.com/blog/products/data-analytics/using-bigquery-dataframes-with-carto-geospatial-tools
    as a way to pass a DataFrame on to carto without executing the SQL
    immediately.

    Make sure that this SQL can be run outside of BigQuery DataFrames (assuming
    similar credentials / access to the referenced tables).
    """
    # Do some operations to make for more complex SQL.
    df = (
        scalars_df_default_index.drop(columns=["geography_col"])
        .groupby(["string_col", "bool_col"])
        .max()
    )
    query = df.sql

    bf_result = df.to_pandas().sort_values("rowindex")
    bq_result = (
        bigquery_client.query_and_wait(query)
        .to_dataframe()
        .set_index(["string_col", "bool_col"])
        .sort_values("rowindex")
    )
    pandas.testing.assert_frame_equal(
        bf_result, bq_result, check_dtype=False, check_index_type=False
    )


def test_to_arrow(scalars_df_default_index, scalars_pandas_df_default_index):
    """Verify to_arrow() APIs returns the expected data."""
    expected = pa.Table.from_pandas(
        scalars_pandas_df_default_index.drop(columns=["geography_col"])
    )

    with pytest.warns(
        bigframes.exceptions.PreviewWarning,
        match="to_arrow",
    ):
        actual = scalars_df_default_index.drop(columns=["geography_col"]).to_arrow()

    # Make string_col match type. Otherwise, pa.Table.from_pandas uses
    # LargeStringArray. LargeStringArray is unnecessary because our strings are
    # less than 2 GB.
    expected = expected.set_column(
        expected.column_names.index("string_col"),
        pa.field("string_col", pa.string()),
        expected["string_col"].cast(pa.string()),
    )

    # Note: the final .equals assertion covers all these checks, but these
    # finer-grained assertions are easier to debug.
    assert actual.column_names == expected.column_names
    for column in actual.column_names:
        assert actual[column].equals(expected[column])
    assert actual.equals(expected)


def test_to_arrow_multiindex(scalars_df_index, scalars_pandas_df_index):
    scalars_df_multiindex = scalars_df_index.set_index(["string_col", "int64_col"])
    scalars_pandas_df_multiindex = scalars_pandas_df_index.set_index(
        ["string_col", "int64_col"]
    )
    expected = pa.Table.from_pandas(
        scalars_pandas_df_multiindex.drop(columns=["geography_col"])
    )

    with pytest.warns(
        bigframes.exceptions.PreviewWarning,
        match="to_arrow",
    ):
        actual = scalars_df_multiindex.drop(columns=["geography_col"]).to_arrow()

    # Make string_col match type. Otherwise, pa.Table.from_pandas uses
    # LargeStringArray. LargeStringArray is unnecessary because our strings are
    # less than 2 GB.
    expected = expected.set_column(
        expected.column_names.index("string_col"),
        pa.field("string_col", pa.string()),
        expected["string_col"].cast(pa.string()),
    )

    # Note: the final .equals assertion covers all these checks, but these
    # finer-grained assertions are easier to debug.
    assert actual.column_names == expected.column_names
    for column in actual.column_names:
        assert actual[column].equals(expected[column])
    assert actual.equals(expected)


def test_to_pandas_w_correct_dtypes(scalars_df_default_index):
    """Verify to_pandas() APIs returns the expected dtypes."""
    actual = scalars_df_default_index.to_pandas().dtypes
    expected = scalars_df_default_index.dtypes

    pd.testing.assert_series_equal(actual, expected)


def test_to_pandas_array_struct_correct_result(session):
    """In future, we should support arrays with arrow types.
    For now we fall back to the current connector behavior of converting
    to Python objects"""
    df = session.read_gbq(
        """SELECT
        [1, 3, 2] AS array_column,
        STRUCT(
            "a" AS string_field,
            1.2 AS float_field) AS struct_column"""
    )

    result = df.to_pandas()
    expected = pd.DataFrame(
        {
            "array_column": pd.Series(
                [[1, 3, 2]],
                dtype=(
                    pd.ArrowDtype(pa.list_(pa.int64()))
                    if bigframes.features.PANDAS_VERSIONS.is_arrow_list_dtype_usable
                    else "object"
                ),
            ),
            "struct_column": pd.Series(
                [{"string_field": "a", "float_field": 1.2}],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            ("string_field", pa.string()),
                            ("float_field", pa.float64()),
                        ]
                    )
                ),
            ),
        }
    )
    expected.index = expected.index.astype("Int64")
    pd.testing.assert_series_equal(result.dtypes, expected.dtypes)
    pd.testing.assert_series_equal(result["array_column"], expected["array_column"])
    # assert_series_equal not implemented for struct columns yet. Compare
    # values as Python objects, instead.
    pd.testing.assert_series_equal(
        result["struct_column"].astype("O"), expected["struct_column"].astype("O")
    )


def test_load_json_w_unboxed_py_value(session):
    sql = """
        SELECT 0 AS id, JSON_OBJECT('boolean', True) AS json_col,
        UNION ALL
        SELECT 1, JSON_OBJECT('int', 100),
        UNION ALL
        SELECT 2, JSON_OBJECT('float', 0.98),
        UNION ALL
        SELECT 3, JSON_OBJECT('string', 'hello world'),
        UNION ALL
        SELECT 4, JSON_OBJECT('array', [8, 9, 10]),
        UNION ALL
        SELECT 5, JSON_OBJECT('null', null),
        UNION ALL
        SELECT
            6,
            JSON_OBJECT(
                'dict',
                JSON_OBJECT(
                    'int', 1,
                    'array', [JSON_OBJECT('bar', 'hello'), JSON_OBJECT('foo', 1)]
                )
            ),
    """
    df = session.read_gbq(sql, index_col="id")

    assert df.dtypes["json_col"] == db_dtypes.JSONDtype()
    assert isinstance(df["json_col"][0], dict)

    assert df["json_col"][0]["boolean"]
    assert df["json_col"][1]["int"] == 100
    assert math.isclose(df["json_col"][2]["float"], 0.98)
    assert df["json_col"][3]["string"] == "hello world"
    assert df["json_col"][4]["array"] == [8, 9, 10]
    assert df["json_col"][5]["null"] is None
    assert df["json_col"][6]["dict"] == {
        "int": 1,
        "array": [{"bar": "hello"}, {"foo": 1}],
    }


def test_load_json_to_pandas_has_correct_result(session):
    df = session.read_gbq("SELECT JSON_OBJECT('foo', 10, 'bar', TRUE) AS json_col")
    assert df.dtypes["json_col"] == db_dtypes.JSONDtype()
    result = df.to_pandas()

    # The order of keys within the JSON object shouldn't matter for equality checks.
    pd_df = pd.DataFrame(
        {"json_col": [{"bar": True, "foo": 10}]},
        dtype=db_dtypes.JSONDtype(),
    )
    pd_df.index = pd_df.index.astype("Int64")
    pd.testing.assert_series_equal(result.dtypes, pd_df.dtypes)
    pd.testing.assert_series_equal(result["json_col"], pd_df["json_col"])


def test_load_json_in_struct(session):
    """Avoid regressions for internal issue 381148539."""
    sql = """
        SELECT 0 AS id, STRUCT(JSON_OBJECT('boolean', True) AS data, 1 AS number) AS struct_col
        UNION ALL
        SELECT 1, STRUCT(JSON_OBJECT('int', 100), 2),
        UNION ALL
        SELECT 2, STRUCT(JSON_OBJECT('float', 0.98), 3),
        UNION ALL
        SELECT 3, STRUCT(JSON_OBJECT('string', 'hello world'), 4),
        UNION ALL
        SELECT 4, STRUCT(JSON_OBJECT('array', [8, 9, 10]), 5),
        UNION ALL
        SELECT 5, STRUCT(JSON_OBJECT('null', null), 6),
        UNION ALL
        SELECT
            6,
            STRUCT(JSON_OBJECT(
                'dict',
                JSON_OBJECT(
                    'int', 1,
                    'array', [JSON_OBJECT('bar', 'hello'), JSON_OBJECT('foo', 1)]
                )
            ), 7),
    """
    df = session.read_gbq(sql, index_col="id")

    assert isinstance(df.dtypes["struct_col"], pd.ArrowDtype)
    assert isinstance(df.dtypes["struct_col"].pyarrow_dtype, pa.StructType)

    data = df["struct_col"].struct.field("data")
    assert data.dtype == db_dtypes.JSONDtype()

    assert data[0]["boolean"]
    assert data[1]["int"] == 100
    assert math.isclose(data[2]["float"], 0.98)
    assert data[3]["string"] == "hello world"
    assert data[4]["array"] == [8, 9, 10]
    assert data[5]["null"] is None
    assert data[6]["dict"] == {
        "int": 1,
        "array": [{"bar": "hello"}, {"foo": 1}],
    }


def test_load_json_in_array(session):
    sql = """
        SELECT
            0 AS id,
            [
                JSON_OBJECT('boolean', True),
                JSON_OBJECT('int', 100),
                JSON_OBJECT('float', 0.98),
                JSON_OBJECT('string', 'hello world'),
                JSON_OBJECT('array', [8, 9, 10]),
                JSON_OBJECT('null', null),
                JSON_OBJECT(
                    'dict',
                    JSON_OBJECT(
                        'int', 1,
                        'array', [JSON_OBJECT('bar', 'hello'), JSON_OBJECT('foo', 1)]
                    )
                )
            ] AS array_col,
    """
    df = session.read_gbq(sql, index_col="id")

    assert isinstance(df.dtypes["array_col"], pd.ArrowDtype)
    assert isinstance(df.dtypes["array_col"].pyarrow_dtype, pa.ListType)

    data = df["array_col"].list
    assert data.len()[0] == 7
    assert data[0].dtype == db_dtypes.JSONDtype()

    assert data[0][0]["boolean"]
    assert data[1][0]["int"] == 100
    assert math.isclose(data[2][0]["float"], 0.98)
    assert data[3][0]["string"] == "hello world"
    assert data[4][0]["array"] == [8, 9, 10]
    assert data[5][0]["null"] is None
    assert data[6][0]["dict"] == {
        "int": 1,
        "array": [{"bar": "hello"}, {"foo": 1}],
    }


def test_to_pandas_batches_w_correct_dtypes(scalars_df_default_index):
    """Verify to_pandas_batches() APIs returns the expected dtypes."""
    expected = scalars_df_default_index.dtypes
    for df in scalars_df_default_index.to_pandas_batches():
        actual = df.dtypes
        pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    ("index",),
    [(True,), (False,)],
)
def test_to_csv_index(
    scalars_dfs: Tuple[bigframes.dataframe.DataFrame, pd.DataFrame],
    gcs_folder: str,
    index: bool,
):
    if pd.__version__.startswith("1."):
        pytest.skip("date_format parameter not supported in pandas 1.x.")
    """Test the `to_csv` API with the `index` parameter."""
    scalars_df, scalars_pandas_df = scalars_dfs
    index_col = None
    path = gcs_folder + f"test_index_df_to_csv_index_{index}*.csv"
    if index:
        index_col = typing.cast(str, scalars_df.index.name)

    # TODO(swast): Support "date_format" parameter and make sure our
    # DATETIME/TIMESTAMP column export is the same format as pandas by default.
    scalars_df.to_csv(path, index=index)

    # Pandas dataframes dtypes from read_csv are not fully compatible with
    # BigQuery-backed dataframes, so manually convert the dtypes specifically
    # here.
    dtype = scalars_df.reset_index().dtypes.to_dict()
    dtype.pop("geography_col")
    dtype.pop("rowindex")
    # read_csv will decode into bytes inproperly, convert_pandas_dtypes will encode properly from string
    dtype.pop("bytes_col")
    gcs_df = pd.read_csv(
        utils.get_first_file_from_wildcard(path),
        dtype=dtype,
        date_format={"timestamp_col": "YYYY-MM-DD HH:MM:SS Z"},
        index_col=index_col,
    )
    utils.convert_pandas_dtypes(gcs_df, bytes_col=True)
    gcs_df.index.name = scalars_df.index.name

    scalars_pandas_df = scalars_pandas_df.copy()
    scalars_pandas_df.index = scalars_pandas_df.index.astype("int64")
    # Ordering should be maintained for tables smaller than 1 GB.
    pd.testing.assert_frame_equal(gcs_df, scalars_pandas_df)


def test_to_csv_tabs(
    scalars_dfs: Tuple[bigframes.dataframe.DataFrame, pd.DataFrame],
    gcs_folder: str,
):
    if pd.__version__.startswith("1."):
        pytest.skip("date_format parameter not supported in pandas 1.x.")
    """Test the `to_csv` API with the `sep` parameter."""
    scalars_df, scalars_pandas_df = scalars_dfs
    index_col = typing.cast(str, scalars_df.index.name)
    path = gcs_folder + "test_to_csv_tabs*.csv"

    # TODO(swast): Support "date_format" parameter and make sure our
    # DATETIME/TIMESTAMP column export is the same format as pandas by default.
    scalars_df.to_csv(path, sep="\t", index=True)

    # Pandas dataframes dtypes from read_csv are not fully compatible with
    # BigQuery-backed dataframes, so manually convert the dtypes specifically
    # here.
    dtype = scalars_df.reset_index().dtypes.to_dict()
    dtype.pop("geography_col")
    dtype.pop("rowindex")
    # read_csv will decode into bytes inproperly, convert_pandas_dtypes will encode properly from string
    dtype.pop("bytes_col")
    gcs_df = pd.read_csv(
        utils.get_first_file_from_wildcard(path),
        sep="\t",
        dtype=dtype,
        date_format={"timestamp_col": "YYYY-MM-DD HH:MM:SS Z"},
        index_col=index_col,
    )
    utils.convert_pandas_dtypes(gcs_df, bytes_col=True)
    gcs_df.index.name = scalars_df.index.name

    scalars_pandas_df = scalars_pandas_df.copy()
    scalars_pandas_df.index = scalars_pandas_df.index.astype("int64")

    # Ordering should be maintained for tables smaller than 1 GB.
    pd.testing.assert_frame_equal(gcs_df, scalars_pandas_df)


@pytest.mark.parametrize(
    ("index"),
    [True, False],
)
@pytest.mark.skipif(pandas_gbq is None, reason="required by pd.read_gbq")
def test_to_gbq_index(scalars_dfs, dataset_id, index):
    """Test the `to_gbq` API with the `index` parameter."""
    scalars_df, scalars_pandas_df = scalars_dfs
    destination_table = f"{dataset_id}.test_index_df_to_gbq_{index}"
    df_in = scalars_df.copy()
    if index:
        index_col = "index"
        df_in.index.name = index_col
    else:
        index_col = None

    df_in.to_gbq(destination_table, if_exists="replace", index=index)
    df_out = pd.read_gbq(destination_table, index_col=index_col)

    if index:
        df_out = df_out.sort_index()
    else:
        df_out = df_out.sort_values("rowindex_2").reset_index(drop=True)

    utils.convert_pandas_dtypes(df_out, bytes_col=False)
    # pd.read_gbq interprets bytes_col as object, reconvert to pyarrow binary
    df_out["bytes_col"] = df_out["bytes_col"].astype(pd.ArrowDtype(pa.binary()))
    expected = scalars_pandas_df.copy()
    expected.index.name = index_col
    pd.testing.assert_frame_equal(df_out, expected, check_index_type=False)


@pytest.mark.parametrize(
    ("if_exists", "expected_index"),
    [
        pytest.param("replace", 1),
        pytest.param("append", 2),
        pytest.param(
            "fail",
            0,
            marks=pytest.mark.xfail(
                raises=google.api_core.exceptions.Conflict,
            ),
        ),
        pytest.param(
            "unknown",
            0,
            marks=pytest.mark.xfail(
                raises=ValueError,
            ),
        ),
    ],
)
@pytest.mark.skipif(pandas_gbq is None, reason="required by pd.read_gbq")
def test_to_gbq_if_exists(
    scalars_df_default_index,
    scalars_pandas_df_default_index,
    dataset_id,
    if_exists,
    expected_index,
):
    """Test the `to_gbq` API with the `if_exists` parameter."""
    destination_table = f"{dataset_id}.test_to_gbq_if_exists_{if_exists}"

    scalars_df_default_index.to_gbq(destination_table)
    scalars_df_default_index.to_gbq(destination_table, if_exists=if_exists)

    gcs_df = pd.read_gbq(destination_table)
    assert len(gcs_df.index) == expected_index * len(
        scalars_pandas_df_default_index.index
    )
    pd.testing.assert_index_equal(
        gcs_df.columns, scalars_pandas_df_default_index.columns
    )


def test_to_gbq_w_duplicate_column_names(
    scalars_df_index, scalars_pandas_df_index, dataset_id
):
    """Test the `to_gbq` API when dealing with duplicate column names."""
    destination_table = f"{dataset_id}.test_to_gbq_w_duplicate_column_names"

    # Renaming 'int64_too' to 'int64_col', which will result in 'int64_too'
    # becoming 'int64_col_1' after deduplication.
    scalars_df_index = scalars_df_index.rename(columns={"int64_too": "int64_col"})
    scalars_df_index.to_gbq(destination_table, if_exists="replace")

    bf_result = bpd.read_gbq(destination_table, index_col="rowindex").to_pandas()

    pd.testing.assert_series_equal(
        scalars_pandas_df_index["int64_col"], bf_result["int64_col"]
    )
    pd.testing.assert_series_equal(
        scalars_pandas_df_index["int64_too"],
        bf_result["int64_col_1"],
        check_names=False,
    )


def test_to_gbq_w_None_column_names(
    scalars_df_index, scalars_pandas_df_index, dataset_id
):
    """Test the `to_gbq` API with None as a column name."""
    destination_table = f"{dataset_id}.test_to_gbq_w_none_column_names"

    scalars_df_index = scalars_df_index.rename(columns={"int64_too": None})
    scalars_df_index.to_gbq(destination_table, if_exists="replace")

    bf_result = bpd.read_gbq(destination_table, index_col="rowindex").to_pandas()

    pd.testing.assert_series_equal(
        scalars_pandas_df_index["int64_col"], bf_result["int64_col"]
    )
    pd.testing.assert_series_equal(
        scalars_pandas_df_index["int64_too"],
        bf_result["bigframes_unnamed_column"],
        check_names=False,
    )


@pytest.mark.parametrize(
    "clustering_columns",
    [
        pytest.param(["int64_col", "geography_col"]),
        pytest.param(
            ["float64_col"],
            marks=pytest.mark.xfail(raises=google.api_core.exceptions.BadRequest),
        ),
        pytest.param(
            ["int64_col", "int64_col"],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_to_gbq_w_clustering(
    scalars_df_default_index,
    dataset_id,
    bigquery_client,
    clustering_columns,
):
    """Test the `to_gbq` API for creating clustered tables."""
    destination_table = (
        f"{dataset_id}.test_to_gbq_clustering_{'_'.join(clustering_columns)}"
    )

    scalars_df_default_index.to_gbq(
        destination_table, clustering_columns=clustering_columns
    )
    table = bigquery_client.get_table(destination_table)

    assert list(table.clustering_fields) == clustering_columns
    assert table.expires is None


def test_to_gbq_w_clustering_no_destination(
    scalars_df_default_index,
    bigquery_client,
):
    """Test the `to_gbq` API for creating clustered tables without destination."""
    clustering_columns = ["int64_col", "geography_col"]
    destination_table = scalars_df_default_index.to_gbq(
        clustering_columns=clustering_columns
    )
    table = bigquery_client.get_table(destination_table)

    assert list(table.clustering_fields) == clustering_columns
    assert table.expires is not None


def test_to_gbq_w_invalid_destination_table(scalars_df_index):
    with pytest.raises(ValueError):
        scalars_df_index.to_gbq("table_id")


def test_to_gbq_w_json(bigquery_client):
    """Test the `to_gbq` API can get a JSON column."""
    s1 = bpd.Series([1, 2, 3, 4])
    s2 = bpd.Series(
        ["a", 1, False, ["a", {"b": 1}], {"c": [1, 2, 3]}], dtype=db_dtypes.JSONDtype()
    )

    df = bpd.DataFrame({"id": s1, "json_col": s2})
    destination_table = df.to_gbq()
    table = bigquery_client.get_table(destination_table)

    assert table.schema[1].name == "json_col"
    assert table.schema[1].field_type == "JSON"


@pytest.mark.parametrize(
    ("index"),
    [True, False],
)
def test_to_json_index_invalid_orient(
    scalars_dfs: Tuple[bigframes.dataframe.DataFrame, pd.DataFrame],
    gcs_folder: str,
    index: bool,
):
    scalars_df, _ = scalars_dfs
    path = gcs_folder + f"test_index_df_to_json_index_{index}*.jsonl"
    with pytest.raises(ValueError):
        scalars_df.to_json(path, index=index, lines=True)


@pytest.mark.parametrize(
    ("index"),
    [True, False],
)
def test_to_json_index_invalid_lines(
    scalars_dfs: Tuple[bigframes.dataframe.DataFrame, pd.DataFrame],
    gcs_folder: str,
    index: bool,
):
    scalars_df, _ = scalars_dfs
    path = gcs_folder + f"test_index_df_to_json_index_{index}.jsonl"
    with pytest.raises(NotImplementedError):
        scalars_df.to_json(path, index=index)


@pytest.mark.parametrize(
    ("index"),
    [True, False],
)
def test_to_json_index_records_orient(
    scalars_dfs: Tuple[bigframes.dataframe.DataFrame, pd.DataFrame],
    gcs_folder: str,
    index: bool,
):
    """Test the `to_json` API with the `index` parameter.

    Uses the scalable options orient='records' and lines=True.
    """
    scalars_df, scalars_pandas_df = scalars_dfs
    path = gcs_folder + f"test_index_df_to_json_index_{index}*.jsonl"

    scalars_df.to_json(path, index=index, orient="records", lines=True)

    gcs_df = pd.read_json(
        utils.get_first_file_from_wildcard(path),
        lines=True,
        convert_dates=["datetime_col"],
    )
    utils.convert_pandas_dtypes(gcs_df, bytes_col=True)
    if index and scalars_df.index.name is not None:
        gcs_df = gcs_df.set_index(scalars_df.index.name)

    assert len(gcs_df.index) == len(scalars_pandas_df.index)
    pd.testing.assert_index_equal(gcs_df.columns, scalars_pandas_df.columns)

    gcs_df.index.name = scalars_df.index.name
    gcs_df.index = gcs_df.index.astype("Int64")
    scalars_pandas_df.index = scalars_pandas_df.index.astype("Int64")

    # Ordering should be maintained for tables smaller than 1 GB.
    pd.testing.assert_frame_equal(gcs_df, scalars_pandas_df)


@pytest.mark.parametrize(
    ("index"),
    [True, False],
)
def test_to_parquet_index(scalars_dfs, gcs_folder, index):
    """Test the `to_parquet` API with the `index` parameter."""
    scalars_df, scalars_pandas_df = scalars_dfs
    scalars_pandas_df = scalars_pandas_df.copy()
    path = gcs_folder + f"test_index_df_to_parquet_{index}*.parquet"

    # TODO(b/268693993): Type GEOGRAPHY is not currently supported for parquet.
    scalars_df = scalars_df.drop(columns="geography_col")
    scalars_pandas_df = scalars_pandas_df.drop(columns="geography_col")

    # TODO(swast): Do a bit more processing on the input DataFrame to ensure
    # the exported results are from the generated query, not just the source
    # table.
    scalars_df.to_parquet(path, index=index)

    gcs_df = pd.read_parquet(utils.get_first_file_from_wildcard(path))
    utils.convert_pandas_dtypes(gcs_df, bytes_col=False)
    if index and scalars_df.index.name is not None:
        gcs_df = gcs_df.set_index(scalars_df.index.name)

    assert len(gcs_df.index) == len(scalars_pandas_df.index)
    pd.testing.assert_index_equal(gcs_df.columns, scalars_pandas_df.columns)

    gcs_df.index.name = scalars_df.index.name
    gcs_df.index = gcs_df.index.astype("Int64")
    scalars_pandas_df.index = scalars_pandas_df.index.astype("Int64")

    # Ordering should be maintained for tables smaller than 1 GB.
    pd.testing.assert_frame_equal(
        gcs_df.drop("bytes_col", axis=1), scalars_pandas_df.drop("bytes_col", axis=1)
    )


def test_to_sql_query_unnamed_index_included(
    session: bigframes.Session,
    scalars_df_default_index: bpd.DataFrame,
    scalars_pandas_df_default_index: pd.DataFrame,
):
    bf_df = scalars_df_default_index.reset_index(drop=True)
    sql, idx_ids, idx_labels = bf_df._to_sql_query(include_index=True)
    assert len(idx_labels) == 1
    assert len(idx_ids) == 1
    assert idx_labels[0] is None
    assert idx_ids[0].startswith("bigframes")

    pd_df = scalars_pandas_df_default_index.reset_index(drop=True)
    roundtrip = session.read_gbq(sql, index_col=idx_ids)
    roundtrip.index.names = [None]
    utils.assert_pandas_df_equal(roundtrip.to_pandas(), pd_df, check_index_type=False)


def test_to_sql_query_named_index_included(
    session: bigframes.Session,
    scalars_df_default_index: bpd.DataFrame,
    scalars_pandas_df_default_index: pd.DataFrame,
):
    bf_df = scalars_df_default_index.set_index("rowindex_2", drop=True)
    sql, idx_ids, idx_labels = bf_df._to_sql_query(include_index=True)
    assert len(idx_labels) == 1
    assert len(idx_ids) == 1
    assert idx_labels[0] == "rowindex_2"
    assert idx_ids[0] == "rowindex_2"

    pd_df = scalars_pandas_df_default_index.set_index("rowindex_2", drop=True)
    roundtrip = session.read_gbq(sql, index_col=idx_ids)
    utils.assert_pandas_df_equal(roundtrip.to_pandas(), pd_df)


def test_to_sql_query_unnamed_index_excluded(
    session: bigframes.Session,
    scalars_df_default_index: bpd.DataFrame,
    scalars_pandas_df_default_index: pd.DataFrame,
):
    bf_df = scalars_df_default_index.reset_index(drop=True)
    sql, idx_ids, idx_labels = bf_df._to_sql_query(include_index=False)
    assert len(idx_labels) == 0
    assert len(idx_ids) == 0

    pd_df = scalars_pandas_df_default_index.reset_index(drop=True)
    roundtrip = session.read_gbq(sql)
    utils.assert_pandas_df_equal(
        roundtrip.to_pandas(), pd_df, check_index_type=False, ignore_order=True
    )


def test_to_sql_query_named_index_excluded(
    session: bigframes.Session,
    scalars_df_default_index: bpd.DataFrame,
    scalars_pandas_df_default_index: pd.DataFrame,
):
    bf_df = scalars_df_default_index.set_index("rowindex_2", drop=True)
    sql, idx_ids, idx_labels = bf_df._to_sql_query(include_index=False)
    assert len(idx_labels) == 0
    assert len(idx_ids) == 0

    pd_df = scalars_pandas_df_default_index.set_index(
        "rowindex_2", drop=True
    ).reset_index(drop=True)
    roundtrip = session.read_gbq(sql)
    utils.assert_pandas_df_equal(
        roundtrip.to_pandas(), pd_df, check_index_type=False, ignore_order=True
    )
