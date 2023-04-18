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


def assert_pandas_df_equal_ignore_ordering(df0, df1, rtol=None):
    # Sort by a column to get consistent results.
    if df0.index.name != "rowindex":
        df0 = df0.sort_values(list(df0.columns)).reset_index(drop=True)
        df1 = df1.sort_values(list(df1.columns)).reset_index(drop=True)
    else:
        df0 = df0.sort_index()
        df1 = df1.sort_index()

    # TODO(garrettwu): enable check_type once BF type issue is solved.
    pd.testing.assert_frame_equal(
        df0, df1, check_dtype=False, check_exact=(rtol is not None), rtol=rtol
    )


def assert_series_equal_ignoring_order(left: pd.Series, right: pd.Series, **kwargs):
    if left.index.name is None:
        left = left.sort_values().reset_index(drop=True)
        right = right.sort_values().reset_index(drop=True)
    else:
        left = left.sort_index()
        right = right.sort_index()

    pd.testing.assert_series_equal(left, right, **kwargs)
