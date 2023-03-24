import pandas as pd


def test_repr_w_all_rows(scalars_df, scalars_pandas_df):
    # When there are 10 or fewer rows, the outputs should be identical.
    actual = repr(scalars_df.head(10))
    expected = repr(scalars_pandas_df.head(10))
    assert actual == expected


def test_get_dtypes(scalars_df, scalars_pandas_df):
    dtypes = scalars_df.dtypes
    pd.testing.assert_series_equal(
        dtypes,
        scalars_pandas_df.dtypes,
    )


def test_get_columns(scalars_df, scalars_pandas_df):
    pd.testing.assert_index_equal(scalars_df.columns, scalars_pandas_df.columns)


def test_sql(scalars_df):
    # Note: Exact generated SQL depends on Ibis backend
    # so don't test it here
    assert "SELECT " in scalars_df.sql


def test_assign_coerces_literals_to_compatible_types(scalars_df):
    scalars_df = scalars_df.assign(new_int_col=2, new_float_col=3.0)
    assert scalars_df.new_int_col.dtype == pd.Int64Dtype()
    assert scalars_df.new_float_col.dtype == pd.Float64Dtype()
