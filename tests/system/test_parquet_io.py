import pandas.testing


def test_to_parquet_writes_file(
    scalars_df_index,
    scalars_pandas_df_index,
    gcs_folder: str,
):
    path = gcs_folder + "test_to_parquet_writes_file.parquet"
    # TODO(b/268693993): Type GEOGRAPHY is not currently supported for parquet.
    bf_df = scalars_df_index.drop("geography_col")
    pd_df = scalars_pandas_df_index.drop("geography_col", axis=1)
    # TODO(swast): Do a bit more processing on the input DataFrame to ensure
    # the exported results are from the generated query, not just the source
    # table.
    bf_df.to_parquet(path)
    # TODO(swast): Load pandas DataFrame from exported parquet file to
    # check that it matches the expected output.
    gcs_df = pandas.read_parquet(path)
    # TODO(swast): If we serialize the index, can more easily compare values.
    assert len(gcs_df.index) == len(pd_df.index)
    pandas.testing.assert_index_equal(gcs_df.columns, pd_df.columns)
