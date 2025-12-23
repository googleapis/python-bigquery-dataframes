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

import signal
import unittest.mock as mock

import pandas as pd
import pytest

import bigframes.dataframe
import bigframes.display.anywidget as anywidget

# Skip if anywidget/traitlets not installed, though they should be in the dev env
pytest.importorskip("anywidget")
pytest.importorskip("traitlets")


class TestTableWidget:
    def test_navigation_to_invalid_page_resets_to_valid_page_without_deadlock(self):
        """
        Verifies that navigating to an invalid page resets to a valid page without deadlock.

        This behavior relies on _set_table_html releasing the lock before updating self.page,
        preventing re-entrancy issues where the observer triggers a new update on the same thread.
        """
        mock_df = mock.create_autospec(bigframes.dataframe.DataFrame, instance=True)
        mock_df.columns = ["col1"]
        mock_df.dtypes = {"col1": "object"}

        mock_block = mock.Mock()
        mock_block.has_index = False
        mock_df._block = mock_block

        # We mock _initial_load to avoid complex setup
        with mock.patch.object(anywidget.TableWidget, "_initial_load"):
            widget = anywidget.TableWidget(mock_df)

        # Simulate "loaded data but unknown total rows" state
        widget.page_size = 10
        widget.row_count = None
        widget._all_data_loaded = True

        # Populate cache with 1 page of data (10 rows). Page 0 is valid, page 1+ are invalid.
        widget._cached_batches = [pd.DataFrame({"col1": range(10)})]

        # Mark initial load as complete so observers fire
        widget._initial_load_complete = True

        # Setup timeout to fail fast if deadlock occurs
        def handler(signum, frame):
            raise TimeoutError("Deadlock detected!")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(2)  # 2 seconds timeout

        try:
            # Trigger navigation to page 5 (invalid), which should reset to page 0
            widget.page = 5

            assert widget.page == 0

        finally:
            signal.alarm(0)
