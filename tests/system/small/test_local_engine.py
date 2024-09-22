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
import pytest

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


def test_polars_local_engine_reset_index():
    pd_df = pd.DataFrame({"colA": [1, None, 3], "colB": [3, 1, 2]}, index=[3, 1, 2])
    bf_df = bpd.DataFrame(pd_df)

    bf_result = bf_df.reset_index().to_pandas(local_engine=True)
    pd_result = pd_df.reset_index()
    pandas.testing.assert_frame_equal(bf_result, pd_result)


def test_polars_local_engine_join_binop():
    pd_df_1 = pd.DataFrame({"colA": [1, None, 3], "colB": [3, 1, 2]}, index=[1, 2, 3])
    pd_df_2 = pd.DataFrame(
        {"colA": [100, 200, 300], "colB": [30, 10, 40]}, index=[2, 1, 4]
    )
    bf_df_1 = bpd.DataFrame(pd_df_1)
    bf_df_2 = bpd.DataFrame(pd_df_2)

    bf_result = (bf_df_1 + bf_df_2).to_pandas(local_engine=True)
    pd_result = pd_df_1 + pd_df_2
    # Sort by index because ordering logic isn't quite consistent yet
    pandas.testing.assert_frame_equal(bf_result.sort_index(), pd_result.sort_index())


@pytest.mark.parametrize(
    "join_type",
    ["inner", "left", "right", "outer"],
)
def test_polars_local_engine_joins(join_type):
    pd_df_1 = pd.DataFrame({"colA": [1, None, 3], "colB": [3, 1, 2]}, index=[1, 2, 3])
    pd_df_2 = pd.DataFrame(
        {"colC": [100, 200, 300], "colD": [30, 10, 40]}, index=[2, 1, 4]
    )
    bf_df_1 = bpd.DataFrame(pd_df_1)
    bf_df_2 = bpd.DataFrame(pd_df_2)

    # Sort by index because ordering logic isn't quite consistent yet
    bf_result = bf_df_1.join(bf_df_2, how=join_type).to_pandas(local_engine=True)
    pd_result = pd_df_1.join(pd_df_2, how=join_type)
    # Sort by index because ordering logic isn't quite consistent yet
    pandas.testing.assert_frame_equal(bf_result.sort_index(), pd_result.sort_index())


def test_polars_local_engine_agg():
    pd_df = pd.DataFrame(
        {"colA": [True, False, True, False, True], "colB": [1, 2, 3, 4, 5]}
    )
    bf_df = bpd.DataFrame(pd_df)

    bf_result = bf_df.agg(["sum", "count"]).to_pandas(local_engine=True)
    pd_result = pd_df.agg(["sum", "count"])
    # local engine appears to produce uint32
    pandas.testing.assert_frame_equal(bf_result, pd_result, check_dtype=False)  # type: ignore


def test_polars_local_engine_groupby_sum():
    pd_df = pd.DataFrame(
        {"colA": [True, False, True, False, True], "colB": [1, 2, 3, 4, 5]}
    )
    bf_df = bpd.DataFrame(pd_df)

    bf_result = bf_df.groupby("colA").sum().to_pandas(local_engine=True)
    pd_result = pd_df.groupby("colA").sum()
    pandas.testing.assert_frame_equal(bf_result, pd_result)


def test_polars_local_engine_cumsum():
    pd_df = pd.DataFrame({"colA": [1, 2, 3], "colB": [10, 20, 30]})
    bf_df = bpd.DataFrame(pd_df)

    bf_result = bf_df.cumsum().to_pandas(local_engine=True)
    pd_result = pd_df.cumsum()
    pandas.testing.assert_frame_equal(bf_result, pd_result)


def test_polars_local_engine_explode():
    pd_df = pd.DataFrame(
        {
            "colA": [[1, 2, 3], [4, 5, 6, 7], None],
            "colB": [[10, 20, 30], [40, 50, 60, 70], None],
            "colC": [True, False, True],
        }
    )
    bf_df = bpd.DataFrame(pd_df)

    bf_result = bf_df.explode(["colA", "colB"]).to_pandas(local_engine=True)
    pd_result = pd_df.explode(["colA", "colB"])
    pandas.testing.assert_frame_equal(bf_result, pd_result, check_dtype=False)
