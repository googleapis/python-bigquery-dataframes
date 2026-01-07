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


def test_pd_index_to_array_value_with_empty_index_creates_no_columns(mock_session):
    """
    Tests that `_pd_index_to_array_value` with an empty pandas Index creates
    an ArrayValue with no columns.
    """
    empty_index = pd.Index([], name="test")

    array_val = blocks._pd_index_to_array_value(mock_session, empty_index)

    assert len(array_val.column_ids) == 0


def test_pd_index_to_array_value_with_empty_multiindex_creates_no_columns(mock_session):
    """
    Tests that `_pd_index_to_array_value` with an empty pandas MultiIndex creates
    an ArrayValue with no columns.
    """
    empty_index = pd.MultiIndex.from_arrays([[], []], names=["a", "b"])

    array_val = blocks._pd_index_to_array_value(mock_session, empty_index)

    assert len(array_val.column_ids) == 0


def test_unpivot_with_empty_row_labels(mock_session):
    """
    Tests that `unpivot` handles an empty `row_labels` index correctly.
    """
    import pyarrow as pa

    # Create a dummy ArrayValue
    df = pd.DataFrame({"a": [1, 2, 3]})
    pa_table = pa.Table.from_pandas(df)
    array_value = blocks.core.ArrayValue.from_pyarrow(pa_table, session=mock_session)

    # Call unpivot with an empty pd.Index
    unpivot_result, (index_cols, unpivot_cols, passthrough_cols) = blocks.unpivot(
        array_value,
        row_labels=pd.Index([]),
        unpivot_columns=[("a",)],
    )

    # The expected behavior is that the unpivot operation does nothing and returns
    # the original array_value and empty column tuples.
    assert unpivot_result is array_value
    assert index_cols == tuple()
    assert unpivot_cols == tuple()
    assert passthrough_cols == tuple()
