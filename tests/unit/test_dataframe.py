def test_repr_w_all_rows(scalars_df, scalars_pandas_df):
    # When there are 10 or fewer rows, the outputs should be identical.
    actual = repr(scalars_df.head(10))
    expected = repr(scalars_pandas_df.head(10))
    assert actual == expected
