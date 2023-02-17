import pandas as pd


def test_get_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    series = scalars_df[col_name]
    bf_result = series.compute()
    pd_result = scalars_pandas_df[col_name]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(bf_result, pd_result)


def test_hasattr(scalars_dfs):
    scalars_df, _ = scalars_dfs
    assert hasattr(scalars_df, "int64_col")
    assert hasattr(scalars_df, "head")
    assert not hasattr(scalars_df, "not_exist")


def test_get_column_by_attr(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    series = scalars_df.int64_col
    bf_result = series.compute()
    pd_result = scalars_pandas_df.int64_col

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(bf_result, pd_result)


def test_get_columns(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_names = ["bool_col", "float64_col", "int64_col"]
    df_subset = scalars_df[col_names]
    df_pandas = df_subset.compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df[col_names].columns
    )


def test_drop_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    df_pandas = scalars_df.drop(col_name).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.drop(columns=col_name).columns
    )


def test_drop_columns(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_names = ["int64_col", "geography_col", "time_col"]
    df_pandas = scalars_df.drop(col_names).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.drop(columns=col_names).columns
    )


def test_rename(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name_dict = {"bool_col": "boolean_col"}
    df_pandas = scalars_df.rename(col_name_dict).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.rename(columns=col_name_dict).columns
    )


def test_assign_new_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    kwargs = {"new_col": 2}
    df = scalars_df.assign(**kwargs)
    bf_result = df.compute()
    pd_result = scalars_pandas_df.assign(**kwargs)

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values("rowindex", ignore_index=True)
        pd_result = pd_result.sort_values("rowindex", ignore_index=True)

    # dtype discrepencies of Int64(ibis) vs int64(pandas)
    # TODO(garrettwu): enable check_type once BF type issue is solved.
    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )


def test_assign_existing_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    kwargs = {"int64_col": 2}
    df = scalars_df.assign(**kwargs)
    bf_result = df.compute()
    pd_result = scalars_pandas_df.assign(**kwargs)

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values("rowindex", ignore_index=True)
        pd_result = pd_result.sort_values("rowindex", ignore_index=True)

    # dtype discrepencies of Int64(ibis) vs int64(pandas)
    # TODO(garrettwu): enable check_type once BF type issue is solved.
    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )
