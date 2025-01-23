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

import typing

import pandas as pd

from bigframes import bigquery


def test_unix_seconds(scalars_dfs):
    bigframes_df, pandas_df = scalars_dfs

    actual_res = bigquery.unix_seconds(bigframes_df["timestamp_col"]).to_pandas()

    expected_res = (
        pandas_df["timestamp_col"]
        .apply(lambda ts: _to_unix_epoch(ts, "s"))
        .astype("Int64")
    )
    pd.testing.assert_series_equal(actual_res, expected_res)


def test_unix_millis(scalars_dfs):
    bigframes_df, pandas_df = scalars_dfs

    actual_res = bigquery.unix_millis(bigframes_df["timestamp_col"]).to_pandas()

    expected_res = (
        pandas_df["timestamp_col"]
        .apply(lambda ts: _to_unix_epoch(ts, "ms"))
        .astype("Int64")
    )
    pd.testing.assert_series_equal(actual_res, expected_res)


def test_unix_micros(scalars_dfs):
    bigframes_df, pandas_df = scalars_dfs

    actual_res = bigquery.unix_micros(bigframes_df["timestamp_col"]).to_pandas()

    expected_res = (
        pandas_df["timestamp_col"]
        .apply(lambda ts: _to_unix_epoch(ts, "us"))
        .astype("Int64")
    )
    pd.testing.assert_series_equal(actual_res, expected_res)


def _to_unix_epoch(
    ts: pd.Timestamp, unit: typing.Literal["s", "ms", "us"]
) -> typing.Optional[int]:
    if pd.isna(ts):
        return None
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(1, unit)
