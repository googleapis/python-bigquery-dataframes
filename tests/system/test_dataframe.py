import pandas as pd


def test_get_column(scalars_df, scalars_pandas_df):
    col_name = "int64_col"
    series = scalars_df[col_name]
    series_pandas = series.compute()
    pd.testing.assert_series_equal(series_pandas, scalars_pandas_df[col_name])


def test_get_columns(scalars_df, scalars_pandas_df):
    col_names = ["bool_col", "float64_col", "int64_col"]
    df_subset = scalars_df[col_names]
    df_pandas = df_subset.compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df[col_names].columns
    )
