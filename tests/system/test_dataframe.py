import pandas.testing


def test_get_columns(scalars_df):
    df_subset = scalars_df[["bool_col", "float64_col", "int64_col"]]
    df_pandas = df_subset.compute()
    pandas.testing.assert_index_equal(
        df_pandas.columns, pandas.Index(["bool_col", "float64_col", "int64_col"])
    )
