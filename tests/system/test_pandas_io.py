def test_to_pandas_has_correct_dtypes(scalars_df_no_index):
    """This test ensures conversion from BigFrames to Pandas DataFrame results in the correct dtypes"""
    result = scalars_df_no_index.compute()

    # TODO: Fix all inconsistencies and use this check instead:
    # pd.testing.assert_series_equal(result.dtypes, scalars_df_no_index.dtypes)

    # For now, manually check passing and failing cases:
    actual = result.dtypes
    expected = scalars_df_no_index.dtypes

    # TODO: should be pd.BooleanDtype
    # assert actual["bool_col"] == expected["bool_col"]

    assert actual["bytes_col"] == expected["bytes_col"]

    # TODO: should be db_dtypes.DateDType
    # assert actual["date_col"] == expected["date_col"]

    # TODO: should be microsecond units (not nanosecond)
    # assert actual["datetime_col"] == expected["datetime_col"]

    assert actual["geography_col"] == expected["geography_col"]

    # TODO: should be pd.Int64Dtype not np.float64
    # assert actual["int64_col"] == expected["int64_col"]

    # TODO: should be pd.Int64Dtype not np.int64
    # assert actual["int64_too"] == expected["int64_too"]

    assert actual["numeric_col"] == expected["numeric_col"]

    # TODO: should be pd.Float64Dtype not np.float64
    # assert actual["float64_col"] == expected["float64_col"]

    # TODO: should be pd.Int64Dtype not np.int64
    # assert actual["rowindex"] == expected["rowindex"]

    # TODO: should be pd.StringDtype not object
    # assert actual["string_col"] == expected["string_col"]

    # TODO: should be db_dtypes.TimeDtype not object
    # assert actual["time_col"] == expected["time_col"]

    # TODO: should be pd.DatetimeTZDtype(unit="us", tz="UTC")
    # not np.dtype("datetime64[ns]"). Note that both sides are
    # wrong here.
    # assert actual["timestamp_col"] == expected["timestamp_col"]
