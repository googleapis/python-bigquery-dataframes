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

import pandas as pd
import pytest

import bigframes.dtypes


@pytest.fixture
def mock_df():
    df = mock.Mock()
    # Mock behavior for caching check (shape)
    df.shape = (100, 4)
    df.columns = ["A", "B", "C", "D"]
    # Use actual bigframes dtypes or compatible types that works with is_orderable
    df.dtypes = {
        "A": bigframes.dtypes.INT_DTYPE,
        "B": bigframes.dtypes.STRING_DTYPE,
        "C": bigframes.dtypes.FLOAT_DTYPE,
        "D": bigframes.dtypes.BOOL_DTYPE,
    }

    # Ensure to_pandas_batches returns an iterable
    df.to_pandas_batches.return_value = iter(
        [pd.DataFrame({"A": [1], "B": ["a"], "C": [1.0], "D": [True]})]
    )

    # Ensure sort_values returns the mock itself (so to_pandas_batches is still configured)
    df.sort_values.return_value = df
    return df


def test_init_raises_if_anywidget_not_installed():
    with mock.patch("bigframes.display.anywidget._ANYWIDGET_INSTALLED", False):
        with pytest.raises(ImportError):
            from bigframes.display.anywidget import TableWidget

            TableWidget(mock.Mock())


def test_init_initializes_attributes(mock_df):
    from bigframes.display.anywidget import TableWidget

    # Mock _initial_load to avoid execution
    with mock.patch.object(TableWidget, "_initial_load"):
        widget = TableWidget(mock_df)

    assert widget._dataframe is mock_df
    assert widget.page == 0
    assert widget.page_size > 0
    assert widget.orderable_columns == [
        "A",
        "B",
        "C",
        "D",
    ]  # Int, String, Float, Bool are orderable


def test_init_calls_initial_load(mock_df):
    from bigframes.display.anywidget import TableWidget

    with mock.patch.object(TableWidget, "_initial_load") as mock_load:
        TableWidget(mock_df, deferred=False)
        mock_load.assert_called_once()


def test_validate_page_clamping(mock_df):
    from bigframes.display.anywidget import TableWidget

    with mock.patch.object(TableWidget, "_initial_load"):
        widget = TableWidget(mock_df)
        widget.row_count = 100
        widget.page_size = 10

        # Valid page
        widget.page = 5
        assert widget.page == 5

        # Negative page
        with pytest.raises(ValueError):
            widget.page = -1

        # Page too high
        widget.page = 100
        assert widget.page == 9  # Max page is 9 (0-9)


def test_validate_page_size(mock_df):
    from bigframes.display.anywidget import TableWidget

    with mock.patch.object(TableWidget, "_initial_load"):
        widget = TableWidget(mock_df)

        # Valid page size
        widget.page_size = 50
        assert widget.page_size == 50

        # Negative/Zero page size (should be ignored/reset to previous)
        # Note: Traitlets validation returns the *value to set*.
        # Our validator returns self.page_size if input <= 0.
        original_size = widget.page_size
        widget.page_size = -5
        assert widget.page_size == original_size

        # Too large page size
        widget.page_size = 10000
        assert widget.page_size == 1000


def test_page_size_change_resets_page_and_sort(mock_df):
    from bigframes.display.anywidget import TableWidget

    with mock.patch.object(TableWidget, "_initial_load"):
        widget = TableWidget(mock_df)
        widget._initial_load_complete = True  # Enable observers
        widget.page = 5
        widget.sort_context = [{"column": "A", "ascending": True}]

        # Change page size
        widget.page_size = 20

        assert widget.page == 0
        assert widget.sort_context == []


def test_page_size_change_resets_batches(mock_df):
    from bigframes.display.anywidget import TableWidget

    with mock.patch.object(TableWidget, "_initial_load"):
        widget = TableWidget(mock_df)
        widget._initial_load_complete = True  # Enable observers

        # Trigger page size change
        widget.page_size = 50

    # to_pandas_batches called in _reset_batches_for_new_page_size
    mock_df.to_pandas_batches.assert_called()


def test_page_size_change_resets_sort(mock_df):
    from bigframes.display.anywidget import TableWidget

    with mock.patch.object(TableWidget, "_initial_load"):
        widget = TableWidget(mock_df)
        widget._initial_load_complete = True

        # Setup initial batches mock
        mock_df.to_pandas_batches.reset_mock()

        # Change sort
        widget.sort_context = [{"column": "B", "ascending": False}]

    # to_pandas_batches called again (reset)
    # Note: _initial_load is mocked, so this is the first call in this test setup
    assert mock_df.to_pandas_batches.call_count >= 1


def test_deferred_mode_initialization(mock_df):
    """Test that deferred mode does not load data initially."""
    from bigframes.display.anywidget import TableWidget

    with mock.patch.object(TableWidget, "_initial_load") as mock_load:
        widget = TableWidget(mock_df)

        assert widget.is_deferred_mode is True
        mock_load.assert_not_called()


def test_deferred_mode_execution(mock_df):
    """Test that setting start_execution triggers load and disables deferred mode."""
    from bigframes.display.anywidget import TableWidget

    # specific mock for _initial_load to avoid real execution but allow tracking calls
    # We need to make sure _initial_load exists on the class to patch it
    with mock.patch.object(TableWidget, "_initial_load") as mock_load:
        widget = TableWidget(mock_df)

        assert widget.is_deferred_mode is True
        mock_load.assert_not_called()

        # Simulate user clicking "Run"
        widget.start_execution = True

        # Verify load triggered
        mock_load.assert_called_once()
        assert widget.is_deferred_mode is False


def test_deferred_mode_execution_error(mock_df):
    """Test that error during deferred execution is handled."""
    from bigframes.display.anywidget import TableWidget

    with mock.patch.object(TableWidget, "_initial_load") as mock_load:
        mock_load.side_effect = RuntimeError("Query Failed")

        widget = TableWidget(mock_df)

        # Simulate user clicking "Run"
        widget.start_execution = True

        # Verify mode switched and error set
        assert widget.is_deferred_mode is False
        assert widget._error_message == "Query Failed"


def test_normal_mode_initialization(mock_df):
    """Test that normal mode loads data initially."""
    from bigframes.display.anywidget import TableWidget

    with mock.patch.object(TableWidget, "_initial_load") as mock_load:
        widget = TableWidget(mock_df, deferred=False)

        assert widget.is_deferred_mode is False
        mock_load.assert_called_once()
