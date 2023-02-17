import pandas


def test_repr(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    bf_series = scalars_df[col_name]
    pd_series = scalars_pandas_df[col_name].astype(pandas.Int64Dtype())
    bf_scalar = bf_series.sum()
    pd_scalar = pd_series.sum()
    assert repr(bf_scalar) == repr(pd_scalar)
