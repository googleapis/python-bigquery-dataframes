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

describe('TableWidget Deferred Mode', () => {
  let model;
  let el;
  let render;

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

  describe('Deferred Mode UI', () => {
    it('should show deferred message and hide table when in deferred mode', () => {
      // Mock deferred mode = true
      model.get.mockImplementation((property) => {
        if (property === 'is_deferred_mode') return true;
        if (property === 'table_html') return '<table></table>';
        return null;
      });

      render({ model, el });

      const deferredContainer = el.querySelector('.deferred-message');
      const tableContainer = el.querySelector('.table-container');
      const footer = el.querySelector('.footer');

      expect(deferredContainer.style.display).toBe('flex');
      expect(tableContainer.style.display).toBe('none');
      expect(footer.style.display).toBe('none');
      expect(deferredContainer.textContent).toContain(
        'This is a preview of the widget',
      );
    });

    it('should show table and hide deferred message when not in deferred mode', () => {
      // Mock deferred mode = false
      model.get.mockImplementation((property) => {
        if (property === 'is_deferred_mode') return false;
        if (property === 'table_html') return '<table></table>';
        return null;
      });

      render({ model, el });

      const deferredContainer = el.querySelector('.deferred-message');
      const tableContainer = el.querySelector('.table-container');
      const footer = el.querySelector('.footer');

      expect(deferredContainer.style.display).toBe('none');
      expect(tableContainer.style.display).toBe('block');
      expect(footer.style.display).toBe('flex');
    });

    it('should trigger start_execution when run button is clicked', () => {
      model.get.mockImplementation((property) => {
        if (property === 'is_deferred_mode') return true;
        return null;
      });

      render({ model, el });

      const runButton = el.querySelector('.run-button');
      runButton.click();

      expect(model.set).toHaveBeenCalledWith('start_execution', true);
      expect(model.save_changes).toHaveBeenCalled();
      expect(runButton.textContent).toBe('Running...');
      expect(runButton.disabled).toBe(true);
    });

    it('should update UI when is_deferred_mode changes', () => {
      // Start in deferred mode
      let isDeferred = true;
      model.get.mockImplementation((property) => {
        if (property === 'is_deferred_mode') return isDeferred;
        if (property === 'table_html') return '<table></table>';
        return null;
      });

      render({ model, el });

      const deferredContainer = el.querySelector('.deferred-message');
      const tableContainer = el.querySelector('.table-container');

      expect(deferredContainer.style.display).toBe('flex');
      expect(tableContainer.style.display).toBe('none');

      // Change to non-deferred mode
      isDeferred = false;
      const changeHandler = model.on.mock.calls.find(
        (call) => call[0] === 'change:is_deferred_mode',
      )[1];
      changeHandler();

      expect(deferredContainer.style.display).toBe('none');
      expect(tableContainer.style.display).toBe('block');
    });
  });
});
