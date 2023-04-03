import pandas as pd

import bigframes as bf


def test_concat_dataframe(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = bf.concat([scalars_df, scalars_df])
    bf_result = bf_result.compute()
    pd_result = pd.concat([scalars_pandas_df, scalars_pandas_df])

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values("rowindex", ignore_index=True)
        pd_result = pd_result.sort_values("rowindex", ignore_index=True)
    # TODO(garrettwu): Currently we can't control the ordering for concat. Since it isn't any sortings of the result, but the concatenation of the oritinal ordering. We will need ordering object to be added.
    else:
        bf_result = bf_result.sort_index(ignore_index=True)
        pd_result = pd_result.sort_index(ignore_index=True)

    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )
