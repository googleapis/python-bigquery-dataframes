def test_to_pandas_has_correct_dtypes(scalars_df_no_index):
    """This test ensures conversion from BigFrames to Pandas DataFrame results in the correct dtypes"""
    result = scalars_df_no_index.compute()

    # TODO(chelsealin): Use this check instead after b/275417413.
    # pd.testing.assert_series_equal(result.dtypes, scalars_df_no_index.dtypes)

    # For now, manually check passing and failing cases:
    actual = result.dtypes
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
