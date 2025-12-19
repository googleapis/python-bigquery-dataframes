/**
 * @jest-environment jsdom
 */

// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { jest } from "@jest/globals";
import "@testing-library/jest-dom";

describe("SeriesWidget", () => {
	let model;
	let el;
	let render;

	beforeEach(async () => {
		jest.resetModules();
		document.body.innerHTML = "<div></div>";
		el = document.body.querySelector("div");

		const tableWidget = (
			await import("../../bigframes/display/table_widget.js")
		).default;
		render = tableWidget.render;

		model = {
			get: jest.fn(),
			set: jest.fn(),
			save_changes: jest.fn(),
			on: jest.fn(),
		};
	});

	it("should render the series as a table with an index and one value column", () => {
		// Mock the initial state
		model.get.mockImplementation((property) => {
			if (property === "table_html") {
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
			if (property === "orderable_columns") {
				return [];
			}
			return null;
		});

		render({ model, el });

		// Manually trigger the table_html change handler
		const tableHtmlChangeHandler = model.on.mock.calls.find(
			(call) => call[0] === "change:table_html",
		)[1];
		tableHtmlChangeHandler();

		// Check that the table has two columns
		const headers = el.querySelectorAll(
			".paginated-table-container .col-header-name",
		);
		expect(headers).toHaveLength(2);

		// Check that the headers are an empty string (for the index) and "value"
		expect(headers[0].textContent).toBe("");
		expect(headers[1].textContent).toBe("value");
	});
});
