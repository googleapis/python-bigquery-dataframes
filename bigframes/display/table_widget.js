/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const ModelProperty = {
  TABLE_HTML: "table_html",
  ROW_COUNT: "row_count",
  PAGE_SIZE: "page_size",
  PAGE: "page",
};

const Event = {
  CHANGE: "change",
  CHANGE_TABLE_HTML: `change:${ModelProperty.TABLE_HTML}`,
  CLICK: "click",
};

/**
 * Renders a paginated table and its controls into a given element.
 * @param {{
 * model: !Backbone.Model,
 * el: !HTMLElement
 * }} options
 */
function render({ model, el }) {
  // Structure
  const container = document.createElement("div");
  const tableContainer = document.createElement("div");
  const footer = document.createElement("div");
  // Total rows label
  const rowCountLabel = document.createElement("div");
  // Pagination controls
  const paginationContainer = document.createElement("div");
  const prevPage = document.createElement("button");
  const paginationLabel = document.createElement("span");
  const nextPage = document.createElement("button");
  // Page size controls
  const pageSizeContainer = document.createElement("div");
  const pageSizeLabel = document.createElement("label");
  const pageSizeSelect = document.createElement("select");

  tableContainer.classList.add("table-container");
  footer.classList.add("footer");
  paginationContainer.classList.add("pagination");
  pageSizeContainer.classList.add("page-size");

  prevPage.type = "button";
  nextPage.type = "button";
  prevPage.textContent = "Prev";
  nextPage.textContent = "Next";

  pageSizeLabel.textContent = "Page Size";
  for (const size of [10, 25, 50, 100]) {
    const option = document.createElement('option');
    option.value = size;
    option.textContent = size;
    pageSizeSelect.appendChild(option);
  }
  pageSizeSelect.value = Number(model.get(ModelProperty.PAGE_SIZE));

  /** Updates the button states and page label based on the model. */
  function updateButtonStates() {
    const rowCount = model.get(ModelProperty.ROW_COUNT);
    rowCountLabel.textContent = `${rowCount.toLocaleString()} total rows`;

    const totalPages = Math.ceil(
      model.get(ModelProperty.ROW_COUNT) / model.get(ModelProperty.PAGE_SIZE),
    );
    const currentPage = model.get(ModelProperty.PAGE);

    paginationLabel.textContent = `Page ${currentPage + 1} of ${totalPages}`;
    prevPage.disabled = currentPage === 0;
    nextPage.disabled = currentPage >= totalPages - 1;
  }

  /**
   * Updates the page in the model.
   * @param {number} direction -1 for previous, 1 for next.
   */
  function handlePageChange(direction) {
    const currentPage = model.get(ModelProperty.PAGE);
    const newPage = Math.max(0, currentPage + direction);
    if (newPage !== currentPage) {
      model.set(ModelProperty.PAGE, newPage);
      model.save_changes();
    }
  }

  /** Handles the page_size in the model.
   * @param {number} size - new size to set
   */
  function handlePageSizeChange(size) {
    const currentSize = model.get(ModelProperty.PAGE_SIZE);
    if (size !== currentSize) {
      model.set(ModelProperty.PAGE_SIZE, size);
      model.save_changes();
    }
  }

  /** Updates the HTML in the table container **/
  function handleTableHTMLChange() {
    // Note: Using innerHTML can be a security risk if the content is
    // user-generated. Ensure 'table_html' is properly sanitized.
    tableContainer.innerHTML = model.get(ModelProperty.TABLE_HTML);
    updateButtonStates();
  }

  prevPage.addEventListener(Event.CLICK, () => handlePageChange(-1));
  nextPage.addEventListener(Event.CLICK, () => handlePageChange(1));
  pageSizeSelect.addEventListener(Event.CHANGE, (e) => {
    const newSize = Number(e.target.value);
    if (newSize) {
      handlePageSizeChange(newSize);
    }
  });
  model.on(Event.CHANGE_TABLE_HTML, handleTableHTMLChange);

  // Initial setup
  paginationContainer.appendChild(prevPage);
  paginationContainer.appendChild(paginationLabel);
  paginationContainer.appendChild(nextPage);
  pageSizeContainer.appendChild(pageSizeLabel);
  pageSizeContainer.appendChild(pageSizeSelect);
  footer.appendChild(rowCountLabel);
  footer.appendChild(paginationContainer);
  footer.appendChild(pageSizeContainer);
  container.appendChild(tableContainer);
  container.appendChild(footer);
  el.appendChild(container);
  handleTableHTMLChange();
}

export default { render };
