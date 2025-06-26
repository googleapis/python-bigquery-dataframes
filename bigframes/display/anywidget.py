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
from typing import Iterator, TYPE_CHECKING
import uuid

import pandas as pd

import bigframes

# Follow the same pattern as Polars
anywidget_installed = True
if TYPE_CHECKING:
    import anywidget
    import traitlets
else:
    try:
        import anywidget
        import traitlets
    except Exception:
        anywidget_installed = False


class TableWidget(anywidget.AnyWidget):
    """
    An interactive, paginated table widget for BigFrames DataFrames.
    """

    def __init__(self, dataframe):
        """
        Initialize the TableWidget.
        Args:
            dataframe: The Bigframes Dataframe to display.
        """
        if not anywidget_installed:
            raise ValueError("Anywidget is not installed, cannot create TableWidget.")

        super().__init__()
        self._dataframe = dataframe

        # respect display options
        self.page_size = bigframes.options.display.max_rows

        # Initialize data fetching attributes.
        self._batches = dataframe.to_pandas_batches(page_size=self.page_size)
        self._cached_data = pd.DataFrame(columns=self._dataframe.columns)
        self._table_id = str(uuid.uuid4())
        self._all_data_loaded = False
        self._batch_iterator = None

        # len(dataframe) is expensive, since it will trigger a
        # SELECT COUNT(*) query. It is a must have however.
        self.row_count = len(dataframe)

        # get the initial page
        self._set_table_html()

    # Use functools.cached_property instead of @property for _esm
    @functools.cached_property
    def _esm(self):
        """Load JavaScript code from external file."""
        return resources.read_text(bigframes.display, "table_widget.js")

    page = traitlets.Int(0).tag(sync=True)
    page_size = traitlets.Int(25).tag(sync=True)
    row_count = traitlets.Int(0).tag(sync=True)
    table_html = traitlets.Unicode().tag(sync=True)

    @traitlets.validate("page")
    def _validate_page(self, proposal):
        """Validate and clamp page number to valid range."""
        value = proposal["value"]
        if self.row_count == 0 or self.page_size == 0:
            return 0
        max_page = max(0, math.ceil(self.row_count / self.page_size) - 1)
        return max(0, min(value, max_page))

    def _get_next_batch(self) -> bool:
        """
        Gets the next batch of data from the generator and appends to cache.
        Returns:
            bool: True if a batch was successfully loaded, False otherwise.
        """
        if self._all_data_loaded:
            return False

        try:
            iterator = self._get_batch_iterator()
            batch = next(iterator)
            self._cached_data = pd.concat([self._cached_data, batch], ignore_index=True)
            return True
        except StopIteration:
            self._all_data_loaded = True
            # update row count if we loaded all data
            if self.row_count == 0:
                self.row_count = len(self._cached_data)
            return False
        except Exception as e:
            raise RuntimeError(f"Error during batch processing: {str(e)}") from e

    def _get_batch_iterator(self) -> Iterator[pd.DataFrame]:
        """Lazily initializes and returns the batch iterator."""
        if self._batch_iterator is None:
            self._batch_iterator = iter(self._batches)
        return self._batch_iterator

    def _set_table_html(self):
        """Sets the current html data based on the current page and page size."""
        start = self.page * self.page_size
        end = start + self.page_size

        # fetch more data if the requested page is outside our cache
        while len(self._cached_data) < end and not self._all_data_loaded:
            self._get_next_batch()

        # Get the data for the current page
        page_data = self._cached_data.iloc[start:end]

        # Generate HTML table
        self.table_html = page_data.to_html(
            index=False,
            max_rows=None,
            table_id=f"table-{self._table_id}",
            classes="table table-striped table-hover",
            escape=False,
        )

    @traitlets.observe("page")
    def _page_changed(self, change):
        """Handler for when the page number is changed from the frontend."""
        self._set_table_html()
