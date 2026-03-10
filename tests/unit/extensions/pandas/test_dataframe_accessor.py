# Copyright 2026 Google LLC
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

import unittest.mock as mock

import pandas as pd

# Importing bigframes registers the accessor.
import bigframes  # noqa: F401


def test_dataframe_accessor_sql_scalar():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with mock.patch("bigframes.pandas.io.api.read_pandas") as mock_read_pandas:
        with mock.patch("bigframes.bigquery.sql_scalar") as mock_sql_scalar:
            mock_bf_df = mock.MagicMock()
            mock_bf_df.columns = ["a", "b"]
            mock_bf_df.__getitem__.side_effect = lambda x: f"series_{x}"
            mock_read_pandas.return_value = mock_bf_df

            mock_result_series = mock.MagicMock()
            mock_sql_scalar.return_value = mock_result_series
            mock_result_series.to_pandas.return_value = pd.Series([4, 6])

            # This should trigger the accessor
            result = df.bigquery.sql_scalar("ROUND({0} + {1})")

            mock_read_pandas.assert_called_once()
            # check it was called with df
            assert mock_read_pandas.call_args[0][0] is df

            mock_sql_scalar.assert_called_once_with(
                "ROUND({0} + {1})", ["series_a", "series_b"]
            )

            pd.testing.assert_series_equal(result, pd.Series([4, 6]))


def test_dataframe_accessor_sql_scalar_with_session():
    df = pd.DataFrame({"a": [1]})
    mock_session = mock.MagicMock()

    with mock.patch("bigframes.pandas.io.api.read_pandas") as mock_read_pandas:
        with mock.patch("bigframes.bigquery.sql_scalar") as mock_sql_scalar:
            mock_bf_df = mock.MagicMock()
            mock_bf_df.columns = ["a"]
            mock_bf_df.__getitem__.side_effect = lambda x: f"series_{x}"
            mock_read_pandas.return_value = mock_bf_df

            mock_result_series = mock.MagicMock()
            mock_sql_scalar.return_value = mock_result_series

            df.bigquery.sql_scalar("template", session=mock_session)

            mock_read_pandas.assert_called_once_with(df, session=mock_session)
