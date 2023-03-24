def test_dtype(scalars_df, scalars_pandas_df):
    for column in scalars_df.columns:
        assert scalars_df[column].dtype == scalars_pandas_df[column].dtype
