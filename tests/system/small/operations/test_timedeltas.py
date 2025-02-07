# Copyright 2025 Google LLC
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


import datetime

import numpy as np
import pandas as pd
import pandas.testing
import pyarrow as pa
import pytest

from bigframes import dtypes


@pytest.fixture(scope="module")
def temporal_dfs(session):
    pandas_df = pd.DataFrame(
        {
            "datetime_col": [
                pd.Timestamp("2025-02-01 01:00:01"),
                pd.Timestamp("2019-01-02 02:00:00"),
            ],
            "timestamp_col": [
                pd.Timestamp("2023-01-01 01:00:01", tz="UTC"),
                pd.Timestamp("2024-01-02 02:00:00", tz="UTC"),
            ],
            "timedelta_col": [pd.Timedelta(3, "s"), pd.Timedelta(-4, "d")],
        }
    )

    bigframes_df = session.read_pandas(pandas_df)

    return bigframes_df, pandas_df


@pytest.mark.parametrize(
    ("column", "bf_dtype"),
    [
        ("datetime_col", dtypes.DATETIME_DTYPE),
        ("timestamp_col", dtypes.TIMESTAMP_DTYPE),
    ],
)
def test_timestamp_add__ts_series_plus_td_series(temporal_dfs, column, bf_dtype):
    bf_df, pd_df = temporal_dfs

    actual_result = bf_df[column] + bf_df["timedelta_col"]

    expected_result = (pd_df[column] + pd_df["timedelta_col"]).astype(bf_dtype)
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


@pytest.mark.parametrize(
    "literal",
    [
        pytest.param(pd.Timedelta(1, unit="s"), id="pandas"),
        pytest.param(datetime.timedelta(seconds=1), id="python-datetime"),
        pytest.param(np.timedelta64(1, "s"), id="numpy"),
    ],
)
def test_timestamp_add__ts_series_plus_td_literal(temporal_dfs, literal):
    bf_df, pd_df = temporal_dfs

    actual_result = bf_df["timestamp_col"] + literal

    expected_result = (pd_df["timestamp_col"] + literal).astype(dtypes.TIMESTAMP_DTYPE)
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


@pytest.mark.parametrize(
    ("column", "bf_dtype"),
    [
        ("datetime_col", dtypes.DATETIME_DTYPE),
        ("timestamp_col", dtypes.TIMESTAMP_DTYPE),
    ],
)
def test_timestamp_add__td_series_plus_ts_series(temporal_dfs, column, bf_dtype):
    bf_df, pd_df = temporal_dfs

    actual_result = bf_df["timedelta_col"] + bf_df[column]

    expected_result = (pd_df["timedelta_col"] + pd_df[column]).astype(bf_dtype)
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


def test_timestamp_add__td_literal_plus_ts_series(temporal_dfs):
    bf_df, pd_df = temporal_dfs
    timedelta = pd.Timedelta(1, unit="s")

    actual_result = timedelta + bf_df["datetime_col"]

    expected_result = (timedelta + pd_df["datetime_col"]).astype(dtypes.DATETIME_DTYPE)
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


def test_timestamp_add__ts_literal_plus_td_series(temporal_dfs):
    bf_df, pd_df = temporal_dfs
    timestamp = pd.Timestamp("2025-01-01", tz="UTC")

    actual_result = timestamp + bf_df["timedelta_col"]

    expected_result = (timestamp + pd_df["timedelta_col"]).astype(
        dtypes.TIMESTAMP_DTYPE
    )
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


@pytest.mark.parametrize(
    ("column", "bf_dtype"),
    [
        ("datetime_col", dtypes.DATETIME_DTYPE),
        ("timestamp_col", dtypes.TIMESTAMP_DTYPE),
    ],
)
def test_timestamp_add_with_numpy_op(temporal_dfs, column, bf_dtype):
    bf_df, pd_df = temporal_dfs

    actual_result = np.add(bf_df[column], bf_df["timedelta_col"])

    expected_result = np.add(pd_df[column], pd_df["timedelta_col"]).astype(bf_dtype)
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


def test_timestamp_add_dataframes(temporal_dfs):
    columns = ["datetime_col", "timestamp_col"]
    timedelta = pd.Timedelta(1, unit="s")
    bf_df, pd_df = temporal_dfs

    actual_result = bf_df[columns] + timedelta

    expected_result = pd_df[columns] + timedelta
    expected_result["datetime_col"] = expected_result["datetime_col"].astype(
        dtypes.DATETIME_DTYPE
    )
    expected_result["timestamp_col"] = expected_result["timestamp_col"].astype(
        dtypes.TIMESTAMP_DTYPE
    )
    pandas.testing.assert_frame_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )
