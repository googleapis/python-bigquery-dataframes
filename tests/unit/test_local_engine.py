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
import pyarrow as pa
import pytest

import bigframes
import bigframes.pandas as bpd

from . import resources


@pytest.fixture(scope="session")
def polars_session():
    return resources.create_polars_session()


@pytest.fixture(scope="session")
def test_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "int1": pd.Series([1, 2, 3], dtype="Int64"),
            "int2": pd.Series([-10, 20, 30], dtype="Int64"),
            "bools": pd.Series([True, None, False], dtype="boolean"),
            "strings": pd.Series(["b", "aa", "ccc"], dtype="string[pyarrow]"),
            "intLists": pd.Series(
                [[1, 2, 3], [4, 5, 6, 7], None],
                dtype=pd.ArrowDtype(pa.list_(pa.int64())),
            ),
        },
    )
    df.index = df.index.astype("Int64")
    return df


# These tests should be unit tests, but Session object is tightly coupled to BigQuery client.
def test_polars_local_engine_add(
    test_frame: pd.DataFrame, polars_session: bigframes.Session
):
    pd_df = test_frame
    bf_df = bpd.DataFrame(pd_df, session=polars_session)

    bf_result = (bf_df["int1"] + bf_df["int2"]).to_pandas()
    pd_result = pd_df.int1 + pd_df.int2
    pandas.testing.assert_series_equal(bf_result, pd_result)


def test_polars_local_engine_order_by(test_frame: pd.DataFrame, polars_session):
    pd_df = test_frame
    bf_df = bpd.DataFrame(pd_df, session=polars_session)

    bf_result = bf_df.sort_values("strings").to_pandas()
    pd_result = pd_df.sort_values("strings")
    pandas.testing.assert_frame_equal(bf_result, pd_result)


def test_polars_local_engine_filter(test_frame: pd.DataFrame, polars_session):
    pd_df = test_frame
    bf_df = bpd.DataFrame(pd_df, session=polars_session)

    bf_result = bf_df.filter(bf_df["int2"] >= 1).to_pandas()
    pd_result = pd_df.filter(pd_df["int2"] >= 1)  # type: ignore
    pandas.testing.assert_frame_equal(bf_result, pd_result)


def test_polars_local_engine_reset_index(test_frame: pd.DataFrame, polars_session):
    pd_df = test_frame
    bf_df = bpd.DataFrame(pd_df, session=polars_session)

    bf_result = bf_df.reset_index().to_pandas()
    pd_result = pd_df.reset_index()
    # pd default index is int64, bf is Int64
    pandas.testing.assert_frame_equal(bf_result, pd_result, check_index_type=False)


def test_polars_local_engine_join_binop(polars_session):
    pd_df_1 = pd.DataFrame({"colA": [1, None, 3], "colB": [3, 1, 2]}, index=[1, 2, 3])
    pd_df_2 = pd.DataFrame(
        {"colA": [100, 200, 300], "colB": [30, 10, 40]}, index=[2, 1, 4]
    )
    bf_df_1 = bpd.DataFrame(pd_df_1, session=polars_session)
    bf_df_2 = bpd.DataFrame(pd_df_2, session=polars_session)

    bf_result = (bf_df_1 + bf_df_2).to_pandas()
    pd_result = pd_df_1 + pd_df_2
    # Sort by index because ordering logic isn't quite consistent yet
    pandas.testing.assert_frame_equal(
        bf_result.sort_index(),
        pd_result.sort_index(),
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    "join_type",
    ["inner", "left", "right", "outer"],
)
def test_polars_local_engine_joins(join_type, polars_session):
    pd_df_1 = pd.DataFrame(
        {"colA": [1, None, 3], "colB": [3, 1, 2]}, index=[1, 2, 3], dtype="Int64"
    )
    pd_df_2 = pd.DataFrame(
        {"colC": [100, 200, 300], "colD": [30, 10, 40]}, index=[2, 1, 4], dtype="Int64"
    )
    bf_df_1 = bpd.DataFrame(pd_df_1, session=polars_session)
    bf_df_2 = bpd.DataFrame(pd_df_2, session=polars_session)

    # Sort by index because ordering logic isn't quite consistent yet
    bf_result = bf_df_1.join(bf_df_2, how=join_type).to_pandas()
    pd_result = pd_df_1.join(pd_df_2, how=join_type)
    # Sort by index because ordering logic isn't quite consistent yet
    pandas.testing.assert_frame_equal(
        bf_result.sort_index(), pd_result.sort_index(), check_index_type=False
    )


def test_polars_local_engine_agg(polars_session):
    pd_df = pd.DataFrame(
        {"colA": [True, False, True, False, True], "colB": [1, 2, 3, 4, 5]}
    )
    bf_df = bpd.DataFrame(pd_df, session=polars_session)

    bf_result = bf_df.agg(["sum", "count"]).to_pandas()
    pd_result = pd_df.agg(["sum", "count"])
    # local engine appears to produce uint32
    pandas.testing.assert_frame_equal(bf_result, pd_result, check_dtype=False, check_index_type=False)  # type: ignore


def test_polars_local_engine_groupby_sum(polars_session):
    pd_df = pd.DataFrame(
        {"colA": [True, False, True, False, True], "colB": [1, 2, 3, 4, 5]}
    )
    bf_df = bpd.DataFrame(pd_df, session=polars_session)

    bf_result = bf_df.groupby("colA").sum().to_pandas()
    pd_result = pd_df.groupby("colA").sum()
    pandas.testing.assert_frame_equal(
        bf_result, pd_result, check_dtype=False, check_index_type=False
    )


def test_polars_local_engine_cumsum(test_frame, polars_session):
    pd_df = test_frame[["int1", "int2"]]
    bf_df = bpd.DataFrame(pd_df, session=polars_session)

    bf_result = bf_df.cumsum().to_pandas()
    pd_result = pd_df.cumsum()
    pandas.testing.assert_frame_equal(bf_result, pd_result)


def test_polars_local_engine_explode(test_frame, polars_session):
    pd_df = test_frame
    bf_df = bpd.DataFrame(pd_df, session=polars_session)

    bf_result = bf_df.explode(["intLists"]).to_pandas()
    pd_result = pd_df.explode(["intLists"])
    pandas.testing.assert_frame_equal(bf_result, pd_result, check_dtype=False)