import pandas as pd


def assert_pandas_df_equal_ignore_ordering(df0, df1, rtol=None):
    # Sort by a column to get consistent results.
    if df0.index.name != "rowindex":
        df0 = df0.sort_values(list(df0.columns)).reset_index(drop=True)
        df1 = df1.sort_values(list(df1.columns)).reset_index(drop=True)
    else:
        df0 = df0.sort_index()
        df1 = df1.sort_index()

    # TODO(garrettwu): enable check_type once BF type issue is solved.
    pd.testing.assert_frame_equal(
        df0, df1, check_dtype=False, check_exact=(rtol is not None), rtol=rtol
    )


def assert_series_equal_ignoring_order(left: pd.Series, right: pd.Series, **kwargs):
    if left.index.name is None:
        left = left.sort_values().reset_index(drop=True)
        right = right.sort_values().reset_index(drop=True)
    else:
        left = left.sort_index()
        right = right.sort_index()

    pd.testing.assert_series_equal(left, right, **kwargs)
