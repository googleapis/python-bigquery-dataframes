import pandas as pd


def assert_pandas_df_equal_ignore_ordering(df0, df1, rtol=None):
    # Sort by a column to get consistent results.
    if df0.index.name != "rowindex":
        df0 = df0.sort_values(list(df0.columns)).reset_index(drop=True)
        df1 = df1.sort_values(list(df1.columns)).reset_index(drop=True)

    # TODO(garrettwu): enable check_type once BF type issue is solved.
    pd.testing.assert_frame_equal(
        df0, df1, check_dtype=False, check_exact=(rtol is not None), rtol=rtol
    )
