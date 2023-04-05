# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

import bigframes.series

from ...utils import assert_series_equal_ignoring_order


def test_find(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_series: bigframes.series.Series = scalars_df[col_name]
    bf_result = bf_series.str.find("W").compute()
    pd_result = scalars_pandas_df[col_name].str.find("W")

    # One of type mismatches to be documented. Here, the `bf_result.dtype` is `Int64` but
    # the `pd_result.dtype` is `float64`: https://github.com/pandas-dev/pandas/issues/51948
    assert_series_equal_ignoring_order(
        pd_result.astype(pd.Int64Dtype()),
        bf_result,
    )
