def test_to_pandas_override_global_option(scalars_df_index):
    bf_series = scalars_df_index["int64_col"]
    # Direct call to_pandas uses global default setting (allow_large_results=True),
    # table has 'bqdf' prefix.
    bf_series.to_pandas()
    assert bf_series._query_job.destination.table_id.startswith("bqdf")

    # When allow_large_results=False, a destination table is implicitly created,
    # table has 'anon' prefix.
    bf_series.to_pandas(allow_large_results=False)
    assert bf_series._query_job.destination.table_id.startswith("anon")
