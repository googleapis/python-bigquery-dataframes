/**
 * @fileoverview Tests for the anywidget-based table widget.
 */

/*
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

import { jest } from '@jest/globals';

/**
 * Test suite for the TableWidget frontend component.
 */
describe('TableWidget', () => {
  /** @type {any} */
  let model;
  /** @type {HTMLElement} */
  let el;
  /** @type {Function} */
  let render;

  /**
   * Sets up the test environment before each test.
   * This includes resetting modules, creating a DOM element,
   * and mocking the widget model.
   */
  beforeEach(async () => {
    jest.resetModules();
    document.body.innerHTML = '<div></div>';
    el = document.body.querySelector('div');

    const tableWidget = (
      await import('../../bigframes/display/table_widget.js')
    ).default;
    render = tableWidget.render;

    model = {
      get: jest.fn(),
      set: jest.fn(),
      save_changes: jest.fn(),
      on: jest.fn(),
    };
  });

  it('should have a render function', () => {
    expect(render).toBeDefined();
  });

  /**
   * Tests for the render function of the widget.
   */
  describe('render', () => {
    it('should create the basic structure', () => {
      // Mock the initial state
      model.get.mockImplementation((property) => {
        if (property === 'table_html') {
          return '';
        }
        if (property === 'row_count') {
          return 100;
        }
        if (property === 'error_message') {
          return null;
        }
        if (property === 'page_size') {
          return 10;
        }
        if (property === 'page') {
          return 0;
        }
        return null;
      });

      render({ model, el });

      expect(el.classList.contains('bigframes-widget')).toBe(true);
      expect(el.querySelector('.error-message')).not.toBeNull();
      expect(el.querySelector('div')).not.toBeNull();
      expect(el.querySelector('div:nth-child(3)')).not.toBeNull();
    });

    /**
     * Verifies that clicking a sortable column header triggers a sort action
     * with the correct parameters.
     */
    it('should sort when a sortable column is clicked', () => {
      // Mock the initial state
      model.get.mockImplementation((property) => {
        if (property === 'table_html') {
          return '<table><thead><tr><th><div>col1</div></th></tr></thead></table>';
        }
        if (property === 'orderable_columns') {
          return ['col1'];
        }
        if (property === 'sort_context') {
          return [];
        }
        return null;
      });

      render({ model, el });

      // Manually trigger the table_html change handler
      const tableHtmlChangeHandler = model.on.mock.calls.find(
        (call) => call[0] === 'change:table_html',
      )[1];
      tableHtmlChangeHandler();

      const header = el.querySelector('th');
      header.click();

      expect(model.set).toHaveBeenCalledWith('sort_context', [
        { column: 'col1', ascending: true },
      ]);
      expect(model.save_changes).toHaveBeenCalled();
    });

    it('should reverse sort direction when a sorted column is clicked', () => {
      // Mock the initial state
      model.get.mockImplementation((property) => {
        if (property === 'table_html') {
          return '<table><thead><tr><th><div>col1</div></th></tr></thead></table>';
        }
        if (property === 'orderable_columns') {
          return ['col1'];
        }
        if (property === 'sort_context') {
          return [{ column: 'col1', ascending: true }];
        }
        return null;
      });

      render({ model, el });

      // Manually trigger the table_html change handler
      const tableHtmlChangeHandler = model.on.mock.calls.find(
        (call) => call[0] === 'change:table_html',
      )[1];
      tableHtmlChangeHandler();

      const header = el.querySelector('th');
      header.click();

      expect(model.set).toHaveBeenCalledWith('sort_context', [
        { column: 'col1', ascending: false },
      ]);
      expect(model.save_changes).toHaveBeenCalled();
    });

    it('should clear sort when a descending sorted column is clicked', () => {
      // Mock the initial state
      model.get.mockImplementation((property) => {
        if (property === 'table_html') {
          return '<table><thead><tr><th><div>col1</div></th></tr></thead></table>';
        }
        if (property === 'orderable_columns') {
          return ['col1'];
        }
        if (property === 'sort_context') {
          return [{ column: 'col1', ascending: false }];
        }
        return null;
      });

      render({ model, el });

      // Manually trigger the table_html change handler
      const tableHtmlChangeHandler = model.on.mock.calls.find(
        (call) => call[0] === 'change:table_html',
      )[1];
      tableHtmlChangeHandler();

      const header = el.querySelector('th');
      header.click();

      expect(model.set).toHaveBeenCalledWith('sort_context', []);
      expect(model.save_changes).toHaveBeenCalled();
    });

    it('should display the correct sort indicator', () => {
      // Mock the initial state
      model.get.mockImplementation((property) => {
        if (property === 'table_html') {
          return '<table><thead><tr><th><div>col1</div></th><th><div>col2</div></th></tr></thead></table>';
        }
        if (property === 'orderable_columns') {
          return ['col1', 'col2'];
        }
        if (property === 'sort_context') {
          return [{ column: 'col1', ascending: true }];
        }
        return null;
      });

      render({ model, el });

      // Manually trigger the table_html change handler
      const tableHtmlChangeHandler = model.on.mock.calls.find(
        (call) => call[0] === 'change:table_html',
      )[1];
      tableHtmlChangeHandler();

      const headers = el.querySelectorAll('th');
      const indicator1 = headers[0].querySelector('.sort-indicator');
      const indicator2 = headers[1].querySelector('.sort-indicator');

      expect(indicator1.textContent).toBe('▲');
      expect(indicator2.textContent).toBe('●');
    });

    /**
     * Tests that holding the Shift key while clicking a column header
     * adds the new column to the existing sort context for multi-column sorting.
     */
    it('should add a column to sort when Shift+Click is used', () => {
      // Mock the initial state: already sorted by col1 asc
      model.get.mockImplementation((property) => {
        if (property === 'table_html') {
          return '<table><thead><tr><th><div>col1</div></th><th><div>col2</div></th></tr></thead></table>';
        }
        if (property === 'orderable_columns') {
          return ['col1', 'col2'];
        }
        if (property === 'sort_context') {
          return [{ column: 'col1', ascending: true }];
        }
        return null;
      });

      render({ model, el });

      // Manually trigger the table_html change handler
      const tableHtmlChangeHandler = model.on.mock.calls.find(
        (call) => call[0] === 'change:table_html',
      )[1];
      tableHtmlChangeHandler();

      const headers = el.querySelectorAll('th');
      const header2 = headers[1]; // col2

      // Simulate Shift+Click
      const clickEvent = new MouseEvent('click', {
        bubbles: true,
        cancelable: true,
        shiftKey: true,
      });
      header2.dispatchEvent(clickEvent);

      expect(model.set).toHaveBeenCalledWith('sort_context', [
        { column: 'col1', ascending: true },
        { column: 'col2', ascending: true },
      ]);
      expect(model.save_changes).toHaveBeenCalled();
    });
  });

  describe('Theme detection', () => {
    beforeEach(() => {
      jest.useFakeTimers();
      // Mock the initial state for theme detection tests
      model.get.mockImplementation((property) => {
        if (property === 'table_html') {
          return '';
        }
        if (property === 'row_count') {
          return 100;
        }
        if (property === 'error_message') {
          return null;
        }
        if (property === 'page_size') {
          return 10;
        }
        if (property === 'page') {
          return 0;
        }
        return null;
      });
    });

    afterEach(() => {
      jest.useRealTimers();
      document.body.classList.remove('vscode-dark');
    });

    it('should add bigframes-dark-mode class in dark mode', () => {
      document.body.classList.add('vscode-dark');
      render({ model, el });
      jest.runAllTimers();
      expect(el.classList.contains('bigframes-dark-mode')).toBe(true);
    });

    it('should not add bigframes-dark-mode class in light mode', () => {
      render({ model, el });
      jest.runAllTimers();
      expect(el.classList.contains('bigframes-dark-mode')).toBe(false);
    });
  });

  it('should render the series as a table with an index and one value column', () => {
    // Mock the initial state
    model.get.mockImplementation((property) => {
      if (property === 'table_html') {
        return `
      <div class="paginated-table-container">
      <div id="table-c" class="table-container">
        <table class="bigframes-styles">
        <thead>
          <tr>
          <th class="col-header-name"><div></div></th>
          <th class="col-header-name"><div>value</div></th>
          </tr>
        </thead>
        <tbody>
          <tr>
          <td class="cell-align-right">0</td>
          <td class="cell-align-left">a</td>
          </tr>
          <tr>
          <td class="cell-align-right">1</td>
          <td class="cell-align-left">b</td>
          </tr>
        </tbody>
        </table>
      </div>
      </div>`;
      }
      if (property === 'orderable_columns') {
        return [];
      }
      return null;
    });

    render({ model, el });

    // Manually trigger the table_html change handler
    const tableHtmlChangeHandler = model.on.mock.calls.find(
      (call) => call[0] === 'change:table_html',
    )[1];
    tableHtmlChangeHandler();

    // Check that the table has two columns
    const headers = el.querySelectorAll(
      '.paginated-table-container .col-header-name',
    );
    expect(headers).toHaveLength(2);

    // Check that the headers are an empty string (for the index) and "value"
    expect(headers[0].textContent).toBe('');
    expect(headers[1].textContent).toBe('value');
  });

  /**
   * Verifies that hovering over a cell in a group of flattened rows
   * (i.e., rows originating from the same nested data structure)
   * adds a hover class to all cells in that group.
   */
  it('should highlight all rows in a group when hovering over a nested data row', () => {
    // Mock HTML with nested data structure (flattened rows)
    model.get.mockImplementation((property) => {
      if (property === 'table_html') {
        return `<table><tbody>
          <tr data-orig-row="0"><td>Row 1 Part A</td></tr>
          <tr data-orig-row="0"><td>Row 1 Part B</td></tr>
          <tr data-orig-row="1"><td>Row 2</td></tr>
        </tbody></table>`;
      }
      if (property === 'orderable_columns') {
        return [];
      }
      return null;
    });

    render({ model, el });

    // Manually trigger the table_html change handler
    const tableHtmlChangeHandler = model.on.mock.calls.find(
      (call) => call[0] === 'change:table_html',
    )[1];
    tableHtmlChangeHandler();

    const firstRowCell = el.querySelector('tr[data-orig-row="0"] td');
    const rowsInGroup = el.querySelectorAll('tr[data-orig-row="0"] td');

    // Simulate mouseover
    const mouseOverEvent = new MouseEvent('mouseover', {
      bubbles: true,
      cancelable: true,
    });
    firstRowCell.dispatchEvent(mouseOverEvent);

    // Check if row-hover class is added to all cells in the group

    rowsInGroup.forEach((cell) => {
      expect(cell.classList.contains('row-hover')).toBe(true);
    });

    // Simulate mouseout
    const mouseOutEvent = new MouseEvent('mouseout', {
      bubbles: true,
      cancelable: true,
    });
    firstRowCell.dispatchEvent(mouseOutEvent);

    // Check if row-hover class is removed

    rowsInGroup.forEach((cell) => {
      expect(cell.classList.contains('row-hover')).toBe(false);
    });
  });

  it('should not highlight unrelated rows when hovering over a nested data row', () => {
    // Mock HTML with nested data structure
    model.get.mockImplementation((property) => {
      if (property === 'table_html') {
        return `<table><tbody>
          <tr data-orig-row="0"><td>Row 1 Part A</td></tr>
          <tr data-orig-row="0"><td>Row 1 Part B</td></tr>
          <tr data-orig-row="1"><td>Row 2</td></tr>
        </tbody></table>`;
      }
      if (property === 'orderable_columns') {
        return [];
      }
      return null;
    });

    render({ model, el });

    const tableHtmlChangeHandler = model.on.mock.calls.find(
      (call) => call[0] === 'change:table_html',
    )[1];
    tableHtmlChangeHandler();

    const row1Cell = el.querySelector('tr[data-orig-row="0"] td');
    const row2Cell = el.querySelector('tr[data-orig-row="1"] td');

    const mouseOverEvent = new MouseEvent('mouseover', {
      bubbles: true,
      cancelable: true,
    });
    row1Cell.dispatchEvent(mouseOverEvent);

    // Row 2 should NOT have the hover class
    expect(row2Cell.classList.contains('row-hover')).toBe(false);
  });

  it('should not highlight other rows when hovering over a non-nested row', () => {
    // Mock HTML with mixed data structure
    model.get.mockImplementation((property) => {
      if (property === 'table_html') {
        return `<table><tbody>
          <tr><td>Standard Row</td></tr>
          <tr data-orig-row="0"><td>Nested Row</td></tr>
        </tbody></table>`;
      }
      if (property === 'orderable_columns') {
        return [];
      }
      return null;
    });

    render({ model, el });

    const tableHtmlChangeHandler = model.on.mock.calls.find(
      (call) => call[0] === 'change:table_html',
    )[1];
    tableHtmlChangeHandler();

    const standardCell = el.querySelector('tr:not([data-orig-row]) td');
    const nestedCell = el.querySelector('tr[data-orig-row="0"] td');

    const mouseOverEvent = new MouseEvent('mouseover', {
      bubbles: true,
      cancelable: true,
    });
    standardCell.dispatchEvent(mouseOverEvent);

    // The nested row should NOT have the hover class
    expect(nestedCell.classList.contains('row-hover')).toBe(false);
  });
});
