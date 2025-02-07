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


@pytest.mark.parametrize("column", ["datetime_col", "timestamp_col"])
def test_timestamp_add__ts_series_plus_td_series(scalars_dfs, column):
    bf_df, pd_df = scalars_dfs

    actual_result = bf_df[column] + bf_df["timedelta_col"]

    expected_result = pd_df[column] + pd_df["timedelta_col"]
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


@pytest.mark.parametrize(
    "literal",
    [
        pytest.param(pd.Timedelta(1, unit="s"), id="pandas"),
        pytest.param(datetime.timedelta(seconds=1), id="python-datetime"),
        pytest.param(np.timedelta64(1, "s"), id="numpy"),
        pytest.param(pa.scalar(1, type=pa.duration("s")), id="pyarrow"),
    ],
)
def test_timestamp_add__ts_series_plus_td_literal(scalars_dfs, literal):
    bf_df, pd_df = scalars_dfs

    actual_result = bf_df["datetime_col"] + literal

    expected_result = (pd_df["datetime_col"] + literal).astype("timestamp[us][pyarrow]")
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


@pytest.mark.parametrize("column", ["datetime_col", "timestamp_col"])
def test_timestamp_add__td_series_plus_ts_series(scalars_dfs, column):
    bf_df, pd_df = scalars_dfs

    actual_result = bf_df["timedelta_col"] + bf_df[column]

    expected_result = pd_df["timedelta_col"] + pd_df[column]
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


def test_timestamp_add__td_literal_plus_ts_series(scalars_dfs):
    bf_df, pd_df = scalars_dfs
    timedelta = pd.Timedelta(1, unit="s")

    actual_result = timedelta + bf_df["datetime_col"]

    expected_result = (timedelta + pd_df["datetime_col"]).astype(
        "timestamp[us][pyarrow]"
    )
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


def test_timestamp_add__ts_literal_plus_td_series(scalars_dfs):
    bf_df, pd_df = scalars_dfs
    timestamp = pd.Timestamp("2025-01-01", tz="UTC")

    actual_result = timestamp + bf_df["timedelta_col"]

    expected_result = timestamp + pd_df["timedelta_col"]
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


@pytest.mark.parametrize("column", ["datetime_col", "timestamp_col"])
def test_timestamp_add_with_numpy_op(scalars_dfs, column):
    bf_df, pd_df = scalars_dfs

    actual_result = np.add(bf_df[column], bf_df["timedelta_col"])

    expected_result = np.add(pd_df[column], pd_df["timedelta_col"])
    pandas.testing.assert_series_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )


def test_timestamp_add_dataframes(scalars_dfs):
    columns = ["datetime_col", "timestamp_col"]
    timedelta = pd.Timedelta(1, unit="s")
    bf_df, pd_df = scalars_dfs

    actual_result = bf_df[columns] + timedelta

    expected_result = pd_df[columns] + timedelta
    expected_result["datetime_col"] = expected_result["datetime_col"].astype(
        "timestamp[us][pyarrow]"
    )
    expected_result["timestamp_col"] = expected_result["timestamp_col"].astype(
        "timestamp[us, tz=UTC][pyarrow]"
    )
    pandas.testing.assert_frame_equal(
        actual_result.to_pandas(), expected_result, check_index_type=False
    )
