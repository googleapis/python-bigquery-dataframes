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


def test_dataframe_groupby_sum(scalars_df_index, scalars_pandas_df_index):
    col_names = ["int64_too", "float64_col", "int64_col", "bool_col", "string_col"]
    bf_series = scalars_df_index[col_names].groupby("string_col").sum()
    pd_series = scalars_pandas_df_index[col_names].groupby("string_col").sum()
    bf_result = bf_series.compute()
    pd.testing.assert_frame_equal(
        pd_series,
        bf_result,
    )


def test_dataframe_groupby_multi_sum(scalars_df_index, scalars_pandas_df_index):
    col_names = ["int64_too", "float64_col", "int64_col", "bool_col", "string_col"]
    bf_series = (
        scalars_df_index[col_names]
        .groupby(["bool_col", "int64_col"], as_index=False)
        .sum()
    )
    pd_series = (
        scalars_pandas_df_index[col_names]
        .groupby(["bool_col", "int64_col"], as_index=False)
        .sum(numeric_only=True)
    )
    bf_result = bf_series.compute()

    # BigFrames default indices use nullable Int64 always
    pd_series.index = pd_series.index.astype("Int64")

    pd.testing.assert_frame_equal(
        pd_series,
        bf_result,
    )
