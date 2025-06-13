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

import pytest

import bigframes.bigquery as bbq
import bigframes.pandas as bpd
from bigframes import operations as ops
from bigframes import dtypes as bpd_dtypes
from bigframes.core import indexes as bpd_indexes
from bigframes.core.groupby import series_groupby as bpd_groupby


def test_json_set_w_invalid_json_path_value_pairs():
    mock_series = mock.create_autospec(bpd.pandas.Series, instance=True)
    with pytest.raises(ValueError, match="Incorrect format"):
        bbq.json_set(mock_series, json_path_value_pairs=[("$.a", 1, 100)])  # type: ignore


# Test 1: Default path, no value_dtype
def test_json_value_array_default_path():
    mock_series = mock.create_autospec(bpd.Series, instance=True)
    # When value_dtype is None, the series from _apply_unary_op is returned directly
    mock_series_after_op = mock.create_autospec(bpd.Series, instance=True)
    mock_series._apply_unary_op.return_value = mock_series_after_op

    result = bbq.json_value_array(mock_series)

    mock_series._apply_unary_op.assert_called_once()
    op_arg = mock_series._apply_unary_op.call_args[0][0]
    assert isinstance(op_arg, ops.JSONValueArray)
    assert op_arg.json_path == "$"
    assert result is mock_series_after_op # Ensure the direct result is returned

# Test 2: Custom path, no value_dtype
def test_json_value_array_custom_path():
    mock_series = mock.create_autospec(bpd.Series, instance=True)
    mock_series_after_op = mock.create_autospec(bpd.Series, instance=True)
    mock_series._apply_unary_op.return_value = mock_series_after_op
    custom_path = "$.data.items"

    result = bbq.json_value_array(mock_series, json_path=custom_path)

    mock_series._apply_unary_op.assert_called_once()
    op_arg = mock_series._apply_unary_op.call_args[0][0]
    assert isinstance(op_arg, ops.JSONValueArray)
    assert op_arg.json_path == custom_path
    assert result is mock_series_after_op

# Test 3: With value_dtype (e.g., Int64)
@mock.patch("bigframes.bigquery._operations.array.array_agg")
def test_json_value_array_with_value_dtype(mock_array_agg):
    mock_series_input = mock.create_autospec(bpd.Series, instance=True)
    mock_index = mock.create_autospec(bpd_indexes.Index, instance=True)
    mock_index.names = ["index_col"]
    mock_series_input.index = mock_index

    series_after_unary_op = mock.create_autospec(bpd.Series, instance=True)
    series_after_explode = mock.create_autospec(bpd.Series, instance=True)
    series_after_astype = mock.create_autospec(bpd.Series, instance=True)
    groupby_object_mock = mock.create_autospec(bpd_groupby.SeriesGroupBy, instance=True)
    final_aggregated_series_mock = mock.create_autospec(bpd.Series, instance=True)

    mock_series_input._apply_unary_op.return_value = series_after_unary_op
    series_after_unary_op.explode.return_value = series_after_explode
    series_after_explode.astype.return_value = series_after_astype
    series_after_astype.groupby.return_value = groupby_object_mock
    mock_array_agg.return_value = final_aggregated_series_mock

    result = bbq.json_value_array(mock_series_input, value_dtype='Int64')

    mock_series_input._apply_unary_op.assert_called_once_with(ops.JSONValueArray(json_path="$"))
    series_after_unary_op.explode.assert_called_once_with()
    series_after_explode.astype.assert_called_once_with('Int64')
    series_after_astype.groupby.assert_called_once_with(level=["index_col"], dropna=False)
    mock_array_agg.assert_called_once_with(groupby_object_mock)
    assert result is final_aggregated_series_mock

# Test 4: With bool value_dtype
@mock.patch("bigframes.bigquery._operations.array.array_agg")
def test_json_value_array_with_bool_dtype(mock_array_agg):
    mock_series_input = mock.create_autospec(bpd.Series, instance=True)
    mock_index = mock.create_autospec(bpd_indexes.Index, instance=True)
    mock_index.names = ["index_col"]
    mock_series_input.index = mock_index

    series_after_unary_op = mock.create_autospec(bpd.Series, instance=True)
    series_after_explode = mock.create_autospec(bpd.Series, instance=True)

    str_accessor_mock = mock.Mock()
    series_after_explode.str = str_accessor_mock
    series_after_lower = mock.create_autospec(bpd.Series, instance=True)
    str_accessor_mock.lower.return_value = series_after_lower

    series_after_comparison = mock.create_autospec(bpd.Series, instance=True)
    series_after_lower.__eq__.return_value = series_after_comparison

    groupby_object_mock = mock.create_autospec(bpd_groupby.SeriesGroupBy, instance=True)
    series_after_comparison.groupby.return_value = groupby_object_mock

    final_aggregated_series_mock = mock.create_autospec(bpd.Series, instance=True)
    mock_array_agg.return_value = final_aggregated_series_mock

    result = bbq.json_value_array(mock_series_input, value_dtype=bpd_dtypes.BOOL_DTYPE)

    mock_series_input._apply_unary_op.assert_called_once_with(ops.JSONValueArray(json_path="$"))
    series_after_unary_op.explode.assert_called_once_with()
    assert series_after_explode.str is str_accessor_mock
    str_accessor_mock.lower.assert_called_once_with()
    series_after_lower.__eq__.assert_called_once_with("true")
    series_after_comparison.groupby.assert_called_once_with(level=["index_col"], dropna=False)
    mock_array_agg.assert_called_once_with(groupby_object_mock)
    assert result is final_aggregated_series_mock
