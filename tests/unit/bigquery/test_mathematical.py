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

import unittest.mock as mock

import bigframes.bigquery as bbq
import bigframes.dataframe as dataframe
import bigframes.dtypes as dtypes
import bigframes.operations as ops
import bigframes.series as series


def test_rand_calls_apply_nary_op():
    mock_series = mock.create_autospec(series.Series, instance=True)

    bbq.rand(mock_series)

    mock_series._apply_nary_op.assert_called_once()
    args, _ = mock_series._apply_nary_op.call_args
    op = args[0]
    assert isinstance(op, ops.SqlScalarOp)
    assert op.sql_template == "RAND()"
    assert op._output_type == dtypes.FLOAT_DTYPE
    assert op.deterministic is False
    assert args[1] == []


def test_rand_with_dataframe():
    mock_df = mock.create_autospec(dataframe.DataFrame, instance=True)
    # mock columns length > 0
    mock_df.columns = ["col1"]
    # mock iloc to return a series
    mock_series = mock.create_autospec(series.Series, instance=True)
    # Configure mock_df.iloc to return mock_series when indexed
    # iloc is indexable, so we mock __getitem__
    mock_indexer = mock.MagicMock()
    mock_indexer.__getitem__.return_value = mock_series
    type(mock_df).iloc = mock.PropertyMock(return_value=mock_indexer)

    bbq.rand(mock_df)

    mock_series._apply_nary_op.assert_called_once()
    args, _ = mock_series._apply_nary_op.call_args
    op = args[0]
    assert isinstance(op, ops.SqlScalarOp)
    assert op.sql_template == "RAND()"
