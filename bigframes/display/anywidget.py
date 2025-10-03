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

from __future__ import annotations

from importlib import resources
import functools
import math
from typing import Any, cast, Dict, Iterator, List, Optional, Type
import uuid

import pandas as pd

import bigframes
import bigframes.core.blocks
import bigframes.display.html
import bigframes.session.execution_spec

# anywidget and traitlets are optional dependencies. We don't want the import of
# this module to fail if they aren't installed, though. Instead, we try to
# limit the surface that these packages could affect. This makes unit testing
# easier and ensures we don't accidentally make these required packages.
try:
    import anywidget
    import traitlets

    ANYWIDGET_INSTALLED = True
except Exception:
    ANYWIDGET_INSTALLED = False

WIDGET_BASE: Type[Any]
if ANYWIDGET_INSTALLED:
    WIDGET_BASE = anywidget.AnyWidget
else:
    WIDGET_BASE = object


class TableWidget(WIDGET_BASE):
    """An interactive, paginated table widget for BigFrames DataFrames.

    This widget provides a user-friendly way to display and navigate through
    large BigQuery DataFrames within a Jupyter environment.
    """

    page = traitlets.Int(0).tag(sync=True)
    page_size = traitlets.Int(0).tag(sync=True)
    row_count = traitlets.Int(0).tag(sync=True)
    table_html = traitlets.Unicode().tag(sync=True)

    def __init__(self, dataframe: bigframes.dataframe.DataFrame):
        """Initialize the TableWidget.

        Args:
            dataframe: The Bigframes Dataframe to display in the widget.
        """
        if not ANYWIDGET_INSTALLED:
            raise ImportError(
                "Please `pip install anywidget traitlets` or "
                "`pip install 'bigframes[anywidget]'` to use TableWidget."
            )

        self._dataframe = dataframe

        super().__init__()

        # This flag prevents observers from firing during initialization.
        # When traitlets like `page` and `page_size` are set in `__init__`, we
        # don't want their corresponding `_..._changed` methods to execute
        # until the widget is fully constructed.
        self._initializing = True

        # Initialize attributes that might be needed by observers first
        self._table_id = str(uuid.uuid4())
        self._all_data_loaded = False
        self._batch_iter: Optional[Iterator[pd.DataFrame]] = None
        self._cached_batches: List[pd.DataFrame] = []

        # Respect display options for initial page size
        self.page_size = bigframes.options.display.max_rows

        # Force execution with explicit destination to get total_rows metadata
        execute_result = dataframe._block.session._executor.execute(
            dataframe._block.expr,
            execution_spec=bigframes.session.execution_spec.ExecutionSpec(
                ordered=True, promise_under_10gb=False
            ),
        )
        # The query issued by `to_pandas_batches()` already contains
        # metadata about how many results there were. Use that to avoid
        # doing an extra COUNT(*) query that `len(...)` would do.
        self.row_count = execute_result.total_rows or 0
        self._batches = cast(
            bigframes.core.blocks.PandasBatches,
            execute_result.to_pandas_batches(page_size=self.page_size),
        )

        self._set_table_html()
        self._initializing = False

    @functools.cached_property
    def _esm(self):
        """Load JavaScript code from external file."""
        return resources.read_text(bigframes.display, "table_widget.js")

    @functools.cached_property
    def _css(self):
        """Load CSS code from external file."""
        return resources.read_text(bigframes.display, "table_widget.css")

    @traitlets.validate("page")
    def _validate_page(self, proposal: Dict[str, Any]) -> int:
        """Validate and clamp the page number to a valid range.

        Args:
            proposal: A dictionary from the traitlets library containing the
                proposed change. The new value is in proposal["value"].

        Returns:
            The validated and clamped page number as an integer.
        """

        value = proposal["value"]
        if self.row_count == 0 or self.page_size == 0:
            return 0

        # Calculate the zero-indexed maximum page number.
        max_page = max(0, math.ceil(self.row_count / self.page_size) - 1)

        # Clamp the proposed value to the valid range [0, max_page].
        return max(0, min(value, max_page))

    @traitlets.validate("page_size")
    def _validate_page_size(self, proposal: Dict[str, Any]) -> int:
        """Validate page size to ensure it's positive and reasonable.

        Args:
            proposal: A dictionary from the traitlets library containing the
                proposed change. The new value is in proposal["value"].

        Returns:
            The validated page size as an integer.
        """
        value = proposal["value"]

        # Ensure page size is positive and within reasonable bounds
        if value <= 0:
            return self.page_size  # Keep current value

        # Cap at reasonable maximum to prevent performance issues
        max_page_size = 1000
        return min(value, max_page_size)

    def _get_next_batch(self) -> bool:
        """
        Gets the next batch of data from the generator and appends to cache.

        Returns:
            True if a batch was successfully loaded, False otherwise.
        """
        if self._all_data_loaded:
            return False

        try:
            iterator = self._batch_iterator
            batch = next(iterator)
            self._cached_batches.append(batch)
            return True
        except StopIteration:
            self._all_data_loaded = True
            return False

    @property
    def _batch_iterator(self) -> Iterator[pd.DataFrame]:
        """Lazily initializes and returns the batch iterator."""
        if self._batch_iter is None:
            self._batch_iter = iter(self._batches)
        return self._batch_iter

    @property
    def _cached_data(self) -> pd.DataFrame:
        """Combine all cached batches into a single DataFrame."""
        if not self._cached_batches:
            return pd.DataFrame(columns=self._dataframe.columns)
        return pd.concat(self._cached_batches, ignore_index=True)

    def _reset_batches_for_new_page_size(self) -> None:
        """Reset the batch iterator when page size changes."""
        # Execute with explicit destination for consistency with __init__
        execute_result = self._dataframe._block.session._executor.execute(
            self._dataframe._block.expr,
            execution_spec=bigframes.session.execution_spec.ExecutionSpec(
                ordered=True, promise_under_10gb=False
            ),
        )

        # Create pandas batches from the ExecuteResult
        self._batches = cast(
            bigframes.core.blocks.PandasBatches,
            execute_result.to_pandas_batches(page_size=self.page_size),
        )

        self._cached_batches = []
        self._batch_iter = None
        self._all_data_loaded = False

    def _set_table_html(self) -> None:
        """Sets the current html data based on the current page and page size."""
        start = self.page * self.page_size
        end = start + self.page_size

        # fetch more data if the requested page is outside our cache
        cached_data = self._cached_data
        while len(cached_data) < end and not self._all_data_loaded:
            if self._get_next_batch():
                cached_data = self._cached_data
            else:
                break

        # Get the data for the current page
        page_data = cached_data.iloc[start:end]

        # Generate HTML table
        self.table_html = bigframes.display.html.render_html(
            dataframe=page_data,
            table_id=f"table-{self._table_id}",
        )

    @traitlets.observe("page")
    def _page_changed(self, _change: Dict[str, Any]) -> None:
        """Handler for when the page number is changed from the frontend."""
        if self._initializing:
            return
        self._set_table_html()

    @traitlets.observe("page_size")
    def _page_size_changed(self, _change: Dict[str, Any]) -> None:
        """Handler for when the page size is changed from the frontend."""
        if self._initializing:
            return
        # Reset the page to 0 when page size changes to avoid invalid page states
        self.page = 0

        # Reset batches to use new page size for future data fetching
        self._reset_batches_for_new_page_size()

        # Update the table display
        self._set_table_html()
