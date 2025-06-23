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

import uuid

import anywidget  # type: ignore
import pandas as pd
import traitlets

import bigframes


class TableWidget(anywidget.AnyWidget):
    """
    An interactive, paginated table widget for BigFrames DataFrames.
    """

    _esm = """
    function render({ model, el }) {
        const container = document.createElement('div');
        container.innerHTML = model.get('table_html');

        const buttonContainer = document.createElement('div');
        const prevPage = document.createElement('button');
        const label = document.createElement('span');
        const nextPage = document.createElement('button');
        prevPage.type = 'button';
        nextPage.type = 'button';
        prevPage.textContent = 'Prev';
        nextPage.textContent = 'Next';

        // update button states and label
        function updateButtonStates() {
            const totalPages = Math.ceil(model.get('row_count') / model.get('page_size'));
            const currentPage = model.get('page');

            // Update label
            label.textContent = `Page ${currentPage + 1} of ${totalPages}`;

            // Update button states
            prevPage.disabled = currentPage === 0;
            nextPage.disabled = currentPage >= totalPages - 1;
        }

        // Initial button state setup
        updateButtonStates();

        prevPage.addEventListener('click', () => {
            let newPage = model.get('page') - 1;
            if (newPage < 0) {
              newPage = 0;
            }
            console.log(`Setting page to ${newPage}`)
            model.set('page', newPage);
            model.save_changes();
        });

        nextPage.addEventListener('click', () => {
            const newPage = model.get('page') + 1;
            console.log(`Setting page to ${newPage}`)
            model.set('page', newPage);
            model.save_changes();
        });

        model.on('change:table_html', () => {
            container.innerHTML = model.get('table_html');
            updateButtonStates(); // Update button states when table changes
        });

        buttonContainer.appendChild(prevPage);
        buttonContainer.appendChild(label);
        buttonContainer.appendChild(nextPage);
        el.appendChild(container);
        el.appendChild(buttonContainer);
    }
    export default { render };
    """

    page = traitlets.Int(0).tag(sync=True)
    page_size = traitlets.Int(25).tag(sync=True)
    row_count = traitlets.Int(0).tag(sync=True)
    table_html = traitlets.Unicode().tag(sync=True)

    def __init__(self, dataframe):
        """
        Initialize the TableWidget.

        Args:
            dataframe: The Bigframes Dataframe to display
        """
        super().__init__()
        self._dataframe = dataframe

        # respect display options
        self.page_size = bigframes.options.display.max_rows

        self._batches = dataframe.to_pandas_batches(page_size=self.page_size)
        self._cached_data = pd.DataFrame(columns=self._dataframe.columns)
        self._table_id = str(uuid.uuid4())
        self._all_data_loaded = False

        # store the iterator as an instance variable
        self._batch_iterator = None

        # len(dataframe) is expensive, since it will trigger a
        # SELECT COUNT(*) query. It is a must have however.
        self.row_count = len(dataframe)

        # get the initial page
        self._set_table_html()

    def _get_next_batch(self):
        """Gets the next batch of data from the batches generator."""
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

    def _get_batch_iterator(self):
        """Get batch Iterator."""
        if self._batch_iterator is None:
            self._batch_iterator = iter(self._batches)
        return self._batch_iterator

    def _set_table_html(self):
        """Sets the current html data based on the current page and page size."""
        start = self.page * self.page_size
        end = start + self.page_size

        # fetch more dat if the requested page is outside our cache
        while len(self._cached_data) < end:
            prev_len = len(self._cached_data)
            self._get_next_batch()
            if len(self._cached_data) == prev_len:
                break
        # Get the data fro the current page
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
        """Handler for when the page nubmer is changed from the frontend"""
        self._set_table_html()
