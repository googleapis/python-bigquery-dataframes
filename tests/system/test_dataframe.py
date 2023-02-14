import pandas as pd


def test_get_column(scalars_df, scalars_pandas_df):
    col_name = "int64_col"
    series = scalars_df[col_name]
    series_pandas = series.compute()
    pd.testing.assert_series_equal(series_pandas, scalars_pandas_df[col_name])


def test_hasattr(scalars_df):
    assert hasattr(scalars_df, "int64_col")
    assert hasattr(scalars_df, "head")
    assert not hasattr(scalars_df, "not_exist")


def test_get_column_by_attr(scalars_df, scalars_pandas_df):
    series = scalars_df.int64_col
    series_pandas = series.compute()
    pd.testing.assert_series_equal(series_pandas, scalars_pandas_df.int64_col)


def test_get_columns(scalars_df, scalars_pandas_df):
    col_names = ["bool_col", "float64_col", "int64_col"]
    df_subset = scalars_df[col_names]
    df_pandas = df_subset.compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df[col_names].columns
    )


def test_drop_column(scalars_df, scalars_pandas_df):
    col_name = "int64_col"
    df_pandas = scalars_df.drop(col_name).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.drop(columns=col_name).columns
    )


def test_drop_columns(scalars_df, scalars_pandas_df):
    col_names = ["int64_col", "geography_col", "time_col"]
    df_pandas = scalars_df.drop(col_names).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.drop(columns=col_names).columns
    )


def test_rename(scalars_df, scalars_pandas_df):
    col_name_dict = {"bool_col": "boolean_col"}
    df_pandas = scalars_df.rename(col_name_dict).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.rename(columns=col_name_dict).columns
    )


def test_assign_new_column(scalars_df, scalars_pandas_df):
    kwargs = {"new_col": 2}
    df = scalars_df.assign(**kwargs)
    df_pandas = df.compute()
    scalars_pandas_df = scalars_pandas_df.assign(**kwargs)
    pd.testing.assert_index_equal(df_pandas.columns, scalars_pandas_df.columns)
    for k in kwargs.keys():
        # dtype discrepencies of Int64(ibis) vs int64(pandas)
        # TODO(garrettwu): enable check_type once BF type issue is solved.
        pd.testing.assert_series_equal(
            df_pandas[k], scalars_pandas_df[k], check_dtype=False
        )


def test_assign_existing_column(scalars_df, scalars_pandas_df):
    kwargs = {"int64_col": 2}
    df = scalars_df.assign(**kwargs)
    df_pandas = df.compute()
    scalars_pandas_df = scalars_pandas_df.assign(**kwargs)
    pd.testing.assert_index_equal(df_pandas.columns, scalars_pandas_df.columns)
    for k in kwargs.keys():
        # dtype discrepencies of Int64(ibis) vs int64(pandas)
        # TODO(garrettwu): enable check_type once BF type issue is solved.
        pd.testing.assert_series_equal(
            df_pandas[k], scalars_pandas_df[k], check_dtype=False
        )
