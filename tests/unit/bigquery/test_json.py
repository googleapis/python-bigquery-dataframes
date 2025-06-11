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
import bigframes.operations as ops
import bigframes.dtypes as dtypes


def test_json_set_w_invalid_json_path_value_pairs():
    mock_series = mock.create_autospec(bpd.pandas.Series, instance=True)
    with pytest.raises(ValueError, match="Incorrect format"):
        bbq.json_set(mock_series, json_path_value_pairs=[("$.a", 1, 100)])  # type: ignore


def test_json_query_array_specific_path():
    mock_input_series = mock.create_autospec(bpd.Series, instance=True)
    # Ensure the mock series has a dtype that is_json_like
    mock_input_series.dtype = dtypes.STRING_DTYPE

    bbq.json_query_array(mock_input_series, json_path="$.items")

    mock_input_series._apply_unary_op.assert_called_once_with(
        ops.JSONQueryArray(json_path="$.items")
    )

def test_json_query_array_default_path():
    mock_input_series = mock.create_autospec(bpd.Series, instance=True)
    # Ensure the mock series has a dtype that is_json_like
    mock_input_series.dtype = dtypes.JSON_DTYPE

    bbq.json_query_array(mock_input_series) # Default path "$"

    mock_input_series._apply_unary_op.assert_called_once_with(
        ops.JSONQueryArray(json_path="$")
    )

def test_json_query_array_input_type_validation_passes_with_json_like():
    # This test is more about the op itself, but we can ensure the function doesn't break it.
    # Assumes the op's output_type method will be invoked during series operation.
    # This kind of test might be more suitable for operation tests if they exist.
    # For now, just ensure the call goes through.
    mock_input_series = mock.create_autospec(bpd.Series, instance=True)
    mock_input_series.dtype = dtypes.STRING_DTYPE
    bbq.json_query_array(mock_input_series)
    mock_input_series._apply_unary_op.assert_called_once()

    mock_input_series_json = mock.create_autospec(bpd.Series, instance=True)
    mock_input_series_json.dtype = dtypes.JSON_DTYPE
    bbq.json_query_array(mock_input_series_json)
    mock_input_series_json._apply_unary_op.assert_called_once()
