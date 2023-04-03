import google.api_core.exceptions
import pandas
import pytest


def test_to_pandas_w_correct_dtypes(scalars_df_no_index):
    """Verify to_pandas() APIs returns the expected dtypes."""
    # TODO(chelsealin): Check scalars_df_index.
    # TODO(chelsealin): Use this check instead after b/275417413.
    # pd.testing.assert_series_equal(result.dtypes, scalars_df_no_index.dtypes)

    # For now, manually check passing and failing cases:
    actual = scalars_df_no_index.to_pandas().dtypes
    expected = scalars_df_no_index.dtypes

    assert actual["bool_col"] == expected["bool_col"]
    assert actual["bytes_col"] == expected["bytes_col"]
    assert actual["date_col"] == expected["date_col"]
    assert actual["geography_col"] == expected["geography_col"]
    assert actual["int64_col"] == expected["int64_col"]
    assert actual["int64_too"] == expected["int64_too"]
    assert actual["numeric_col"] == expected["numeric_col"]
    assert actual["rowindex"] == expected["rowindex"]
    assert actual["time_col"] == expected["time_col"]
    assert actual["float64_col"] == expected["float64_col"]
    assert actual["string_col"] == expected["string_col"]

    # TODO: should be microsecond units (not nanosecond)
    # assert actual["datetime_col"] == expected["datetime_col"]

    # TODO: should be pd.DatetimeTZDtype(unit="us", tz="UTC")
    # not np.dtype("datetime64[ns]"). Note that both sides are
    # wrong here.
    # assert actual["timestamp_col"] == expected["timestamp_col"]


def test_to_csv(scalars_dfs, gcs_folder: str):
    scalars_df, scalars_pandas_df = scalars_dfs
    if scalars_df.index.name is not None:
        path = gcs_folder + "test_to_csv_w_index.csv"
    else:
        path = gcs_folder + "test_to_csv_wo_index.csv"

    scalars_df.to_csv(path)

    # TODO(swast): Load pandas DataFrame from exported parquet file to
    # check that it matches the expected output.
    gcs_df = pandas.read_csv(path)

    # TODO(swast): If we serialize the index, can more easily compare values.
    assert len(gcs_df.index) == len(scalars_pandas_df.index)
    pandas.testing.assert_index_equal(gcs_df.columns, scalars_pandas_df.columns)


def test_to_gbq(scalars_dfs, dataset_id):
    scalars_df, scalars_pandas_df = scalars_dfs
    if scalars_df.index.name is not None:
        table_id = "test_to_gbq_w_index"
    else:
        table_id = "test_to_gbq_wo_index"

    destination_table = f"{dataset_id}.{table_id}"
    scalars_df.to_gbq(destination_table)

    # TODO(chelsealin): If we serialize the index, can more easily compare values.
    gcs_df = pandas.read_gbq(destination_table)
    assert len(gcs_df.index) == len(scalars_pandas_df.index)
    pandas.testing.assert_index_equal(gcs_df.columns, scalars_pandas_df.columns)


def test_to_gbq_if_exists(scalars_df_index, scalars_pandas_df_index, dataset_id):
    destination_table = f"{dataset_id}.test_to_gbq_if_exists"
    scalars_df_index.to_gbq(destination_table)

    with pytest.raises(google.api_core.exceptions.Conflict):
        scalars_df_index.to_gbq(destination_table, if_exists="fail")

    scalars_df_index.to_gbq(destination_table, if_exists="append")
    gcs_df = pandas.read_gbq(destination_table)
    assert len(gcs_df.index) == 2 * len(scalars_pandas_df_index.index)
    pandas.testing.assert_index_equal(gcs_df.columns, scalars_pandas_df_index.columns)

    scalars_df_index.to_gbq(destination_table, if_exists="replace")
    gcs_df = pandas.read_gbq(destination_table)
    assert len(gcs_df.index) == len(scalars_pandas_df_index.index)
    pandas.testing.assert_index_equal(gcs_df.columns, scalars_pandas_df_index.columns)


def test_to_gbq_w_invalid_destination_table(scalars_df_index):
    with pytest.raises(ValueError):
        scalars_df_index.to_gbq("table_id")


def test_to_gbq_w_invalid_if_exists(scalars_df_index, dataset_id):
    destination_table = f"{dataset_id}.test_to_gbq_w_invalid_if_exists"
    with pytest.raises(ValueError):
        scalars_df_index.to_gbq(destination_table, if_exists="unknown")


def test_to_parquet(scalars_dfs, gcs_folder: str):
    scalars_df, scalars_pandas_df = scalars_dfs
    if scalars_df.index.name is not None:
        path = gcs_folder + "test_to_parquet_w_index.csv"
    else:
        path = gcs_folder + "test_to_parquet_wo_index.csv"

    # TODO(b/268693993): Type GEOGRAPHY is not currently supported for parquet.
    scalars_df = scalars_df.drop("geography_col")
    scalars_pandas_df = scalars_pandas_df.drop("geography_col", axis=1)

    # TODO(swast): Do a bit more processing on the input DataFrame to ensure
    # the exported results are from the generated query, not just the source
    # table.
    scalars_df.to_parquet(path)

    # TODO(swast): Load pandas DataFrame from exported parquet file to
    # check that it matches the expected output.
    gcs_df = pandas.read_parquet(path)

    # TODO(swast): If we serialize the index, can more easily compare values.
    assert len(gcs_df.index) == len(scalars_pandas_df.index)
    pandas.testing.assert_index_equal(gcs_df.columns, scalars_pandas_df.columns)
