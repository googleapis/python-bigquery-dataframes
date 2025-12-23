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

from unittest import mock

import pandas as pd
import pytest

from bigframes.core import blocks


@pytest.fixture
def mock_session():
    session = mock.MagicMock()
    session.bqclient = None
    return session


def test_pd_index_to_array_value_with_empty_index_creates_columns(mock_session):
    """
    Tests that `_pd_index_to_array_value` correctly handles an empty pandas Index by creating
    an ArrayValue with the expected columns (index column + offset column).
    This prevents crashes in `unpivot` which expects these columns to exist.
    """
    empty_index = pd.Index([], name="test")

    array_val = blocks._pd_index_to_array_value(mock_session, empty_index)

    # Should be 2: one for index, one for offset
    assert len(array_val.column_ids) == 2


def test_pd_index_to_array_value_with_empty_multiindex_creates_columns(mock_session):
    """
    Tests that `_pd_index_to_array_value` correctly handles an empty pandas MultiIndex by creating
    an ArrayValue with the expected columns (one for each level + offset column).
    """
    empty_index = pd.MultiIndex.from_arrays([[], []], names=["a", "b"])

    array_val = blocks._pd_index_to_array_value(mock_session, empty_index)

    # Should have 3 columns: a, b, offset
    assert len(array_val.column_ids) == 3
