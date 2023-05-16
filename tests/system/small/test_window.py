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


def test_rolling_min(scalars_df_index, scalars_pandas_df_index):
    col_name = "int64_too"
    bf_series = scalars_df_index[col_name].rolling(3).min().compute()
    pd_series = scalars_pandas_df_index[col_name].rolling(3).min()

    # Pandas converts to float64, which is not desired
    pd_series = pd_series.astype("Int64")

    pd.testing.assert_series_equal(
        pd_series,
        bf_series,
    )


def test_rolling_max(scalars_df_index, scalars_pandas_df_index):
    col_name = "int64_too"
    bf_series = scalars_df_index[col_name].rolling(3).max().compute()
    pd_series = scalars_pandas_df_index[col_name].rolling(3).max()

    # Pandas converts to float64, which is not desired
    pd_series = pd_series.astype("Int64")

    pd.testing.assert_series_equal(
        pd_series,
        bf_series,
    )
