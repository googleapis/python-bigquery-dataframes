# Copyright 2024 Google LLC
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
import pandas.testing

import bigframes.pandas as bpd


def test_polars_local_engine_add():
    pd_df = pd.DataFrame({"colA": [1, 2, 3], "colB": [10, 20, 30]})
    bf_df = bpd.DataFrame(pd_df)

    bf_result = (bf_df["colA"] + bf_df["colB"]).to_pandas(local_engine=True)
    pd_result = pd_df.colA + pd_df.colB
    pandas.testing.assert_series_equal(bf_result, pd_result)


def test_polars_local_engine_order_by():
    pd_df = pd.DataFrame({"colA": [1, None, 3], "colB": [3, 1, 2]})
    bf_df = bpd.DataFrame(pd_df)

    bf_result = bf_df.sort_values("colB").to_pandas(local_engine=True)
    pd_result = pd_df.sort_values("colB")
    pandas.testing.assert_frame_equal(bf_result, pd_result)


def test_polars_local_engine_filter():
    pd_df = pd.DataFrame({"colA": [1, None, 3], "colB": [3, 1, 2]})
    bf_df = bpd.DataFrame(pd_df)

    bf_result = bf_df.filter(bf_df["colB"] >= 1).to_pandas(local_engine=True)
    pd_result = pd_df.filter(pd_df["colB"] >= 1)  # type: ignore
    pandas.testing.assert_frame_equal(bf_result, pd_result)
