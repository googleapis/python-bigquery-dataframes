from tests.system.utils import assert_pandas_index_equal_ignore_index_type


def test_get_index(scalars_df_index, scalars_pandas_df_index):
    index = scalars_df_index.index
    bf_result = index.compute()
    pd_result = scalars_pandas_df_index.index

    assert_pandas_index_equal_ignore_index_type(bf_result, pd_result)
