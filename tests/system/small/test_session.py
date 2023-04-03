from typing import Tuple, Union

import google.api_core.exceptions
import pandas as pd
import pytest

import bigframes
import bigframes.core.indexes.index
import bigframes.dataframe
import bigframes.dtypes


def test_read_gbq(session: bigframes.Session, scalars_table_id, scalars_schema):
    df = session.read_gbq(scalars_table_id)
    # TODO(swast): Test against public properties like columns or dtypes. Also,
    # check the names and data types match up.
    assert len(df._block.expr._columns) == len(scalars_schema)


def test_read_gbq_w_col_order(session, scalars_table_id, scalars_schema):
    columns = list(column.name for column in scalars_schema)
    df = session.read_gbq(scalars_table_id, col_order=columns)
    assert len(df._block.expr._columns) == len(scalars_schema)

    df = session.read_gbq(scalars_table_id, col_order=[columns[0]])
    assert len(df._block.expr._columns) == 1

    with pytest.raises(ValueError):
        df = session.read_gbq(scalars_table_id, col_order=["unknown"])


def test_read_gbq_sql(
    session, scalars_dfs: Tuple[bigframes.dataframe.DataFrame, pd.DataFrame]
):
    scalars_df, scalars_pandas_df = scalars_dfs
    df_len = len(scalars_pandas_df.index)

    index_cols: Union[Tuple[str], Tuple] = ()
    if isinstance(scalars_df.index, bigframes.core.indexes.index.Index):
        sql = """SELECT
                t.rowindex AS rowindex,
                t.float64_col * 2 AS my_floats,
                CONCAT(t.string_col, "_2") AS my_strings,
                t.int64_col > 0 AS my_bools
            FROM ({subquery}) AS t
            ORDER BY t.rowindex""".format(
            subquery=scalars_df.sql
        )
        index_cols = ("rowindex",)
    else:
        sql = """SELECT
                t.float64_col * 2 AS my_floats,
                CONCAT(t.string_col, "_2") AS my_strings,
                t.int64_col > 0 AS my_bools
            FROM ({subquery}) AS t
            ORDER BY t.rowindex""".format(
            subquery=scalars_df.sql
        )
        index_cols = ()

    df = session.read_gbq(sql, index_cols=index_cols)
    result = df.compute()

    expected = pd.DataFrame(
        {
            "my_floats": pd.Series(scalars_pandas_df["float64_col"] * 2),
            "my_strings": pd.Series(
                scalars_pandas_df["string_col"].str.cat(["_2"] * df_len)
            ),
            "my_bools": pd.Series(scalars_pandas_df["int64_col"] > 0),
        },
    )
    pd.testing.assert_frame_equal(result, expected)


def test_read_gbq_sql_w_col_order(session):
    sql = """SELECT 1 AS my_int_col, "hi" AS my_string_col, 0.2 AS my_float_col"""
    df = session.read_gbq(sql, col_order=["my_float_col", "my_string_col"])
    result = df.compute()
    expected: pd.DataFrame = pd.concat(
        [
            pd.Series([0.2], name="my_float_col", dtype=pd.Float64Dtype()),
            pd.Series(["hi"], name="my_string_col", dtype=pd.StringDtype()),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_read_pandas(session, scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    df = session.read_pandas(scalars_pandas_df)
    result = df.compute()
    expected = scalars_df.compute()

    # TODO(osmanamjad): remove when ordering is supported.
    result = result.sort_values(list(result.columns))
    expected = expected.sort_values(list(expected.columns))

    # TODO(chelsealin): Check the dtypes after supporting all dtypes.
    pd.testing.assert_frame_equal(
        result,
        expected,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )


def test_read_pandas_multi_index_throws_error(session, scalars_pandas_df_multi_index):
    with pytest.raises(NotImplementedError, match="MultiIndex not supported."):
        session.read_pandas(scalars_pandas_df_multi_index)


def test_read_csv(session, scalars_dfs, gcs_folder):
    scalars_df, _ = scalars_dfs
    if scalars_df.index.name is not None:
        path = gcs_folder + "test_to_csv_w_index.csv"
    else:
        path = gcs_folder + "test_to_csv_wo_index.csv"
    scalars_df.to_csv(path)
    gcs_df = session.read_csv(path)

    # TODO(chelsealin): If we serialize the index, can more easily compare values.
    pd.testing.assert_index_equal(gcs_df.columns, scalars_df.columns)

    # In the read_csv() API, the BQ load job auto detects the "byte_col" as the STRING type,
    # and the `numeric_col` as the FLOAT type in BigQuery table.
    # TODO(chelsealin): check the number of rows is expected with the Daframes.count() API.
    gcs_df = gcs_df.drop(["bytes_col", "numeric_col"])
    scalars_df = scalars_df.drop(["bytes_col", "numeric_col"])
    pd.testing.assert_series_equal(gcs_df.dtypes, scalars_df.dtypes)


def test_read_csv_w_header(session, scalars_dfs, gcs_folder):
    scalars_df, _ = scalars_dfs
    if scalars_df.index.name is not None:
        path = gcs_folder + "test_to_csv_w_header_w_index.csv"
    else:
        path = gcs_folder + "test_to_csv_w_header_wo_index.csv"
    scalars_df.to_csv(path)

    # Skip the header and the 1st data rows. Without provided schema, the column names
    # would be like `bool_field_0`, `string_field_1` and etc.
    # TODO(chelsealin): check the number of rows is expected with the Daframes.count() API.
    gcs_df = session.read_csv(path, header=2)
    assert len(gcs_df.columns) == len(scalars_df.columns)


def test_session_id(session):
    assert session._session_id is not None

    # BQ client always runs query within the opened session.
    query_job = session.bqclient.query("SELECT 1")
    assert query_job.session_info.session_id == session._session_id

    # TODO(chelsealin): Verify the session id can be binded with a load job.


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_to_close_session(session):
    assert session._session_id is not None
    session.close()
    assert session._session_id is None

    # Session has expired and is no longer available.
    with pytest.raises(google.api_core.exceptions.BadRequest):
        query_job = session.bqclient.query("SELECT 1")
        query_job.result()  # blocks until finished
