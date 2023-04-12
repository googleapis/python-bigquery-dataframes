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

import bigframes as bf


def test_concat_dataframe(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = bf.concat(11 * [scalars_df])
    bf_result = bf_result.compute()
    pd_result = pd.concat(11 * [scalars_pandas_df])

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values("rowindex", ignore_index=True)
        pd_result = pd_result.sort_values("rowindex", ignore_index=True)

    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )
