import pandas as pd


def test_repr_w_all_rows(scalars_df, scalars_pandas_df):
    # When there are 10 or fewer rows, the outputs should be identical.
    actual = repr(scalars_df.head(10))
    expected = repr(scalars_pandas_df.head(10))
    assert actual == expected


def test_get_dtypes(scalars_df):
    dtypes = scalars_df.dtypes
    pd.testing.assert_series_equal(
        dtypes,
        pd.Series(
            {
                "bool_col": pd.BooleanDtype(),
                "int64_col": pd.Int64Dtype(),
                "float64_col": pd.Float64Dtype(),
                "string_col": pd.StringDtype(),
            }
        ),
    )
