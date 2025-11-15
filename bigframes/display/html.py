# Copyright 2024 Google LLC
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

"""HTML rendering for DataFrames and other objects."""

from __future__ import annotations

import html

import pandas as pd
import pandas.api.types

from bigframes._config import options


def _is_dtype_numeric(dtype) -> bool:
    """Check if a dtype is numeric for alignment purposes."""
    return pandas.api.types.is_numeric_dtype(dtype)


def _calculate_rowspans(dataframe: pd.DataFrame) -> list[list[int]]:
    """Calculates the rowspan for each cell in a MultiIndex DataFrame.

    Args:
        dataframe (pd.DataFrame):
            The DataFrame for which to calculate index rowspans.

    Returns:
        list[list[int]]:
            A list of lists, where each inner list corresponds to an index level
            and contains the rowspan for each row at that level. A value of 0
            indicates that the cell should not be rendered (it's covered by a
            previous rowspan).
    """
    if not isinstance(dataframe.index, pd.MultiIndex):
        # If not a MultiIndex, no rowspans are needed for the index itself.
        # Return a structure that indicates each index cell should be rendered once.
        return [[1] * len(dataframe.index)] if dataframe.index.nlevels > 0 else []

    rowspans: list[list[int]] = []
    for level_idx in range(dataframe.index.nlevels):
        current_level_spans: list[int] = []
        current_value = None
        current_span = 0

        for i in range(len(dataframe.index)):
            value = dataframe.index.get_level_values(level_idx)[i]

            if value == current_value:
                current_span += 1
                current_level_spans.append(0)  # Mark as covered by previous rowspan
            else:
                # If new value, finalize previous span and start a new one
                if current_span > 0:
                    # Update the rowspan for the start of the previous span
                    current_level_spans[i - current_span] = current_span
                current_value = value
                current_span = 1
                current_level_spans.append(0)  # Placeholder, will be updated later

        # Finalize the last span
        if current_span > 0:
            current_level_spans[len(dataframe.index) - current_span] = current_span

        rowspans.append(current_level_spans)

    return rowspans


def render_html(
    *,
    dataframe: pd.DataFrame,
    table_id: str,
) -> str:
    """Renders a pandas DataFrame to an HTML table with specific styling.

    This function generates an HTML table representation of a pandas DataFrame,
    including special handling for MultiIndex to create a nested, rowspan-based
    display similar to the BigQuery UI.

    Args:
        dataframe (pd.DataFrame):
            The DataFrame to render.
        table_id (str):
            A unique ID to assign to the HTML table element.

    Returns:
        str:
            An HTML string representing the rendered DataFrame.
    """
    classes = "dataframe table table-striped table-hover"
    table_html = [f'<table border="1" class="{classes}" id="{table_id}">']
    precision = options.display.precision

    # Render table head
    table_html.append("  <thead>")
    table_html.append('    <tr style="text-align: left;">')

    # Add index headers
    for name in dataframe.index.names:
        table_html.append(
            (
                f'      <th style="text-align: left;">'
                f'<div style="resize: horizontal; overflow: auto; box-sizing: border-box; width: 100%; height: 100%; padding: 0.5em;">'
                f"{html.escape(str(name))}</div></th>"
            )
        )

    for col in dataframe.columns:
        table_html.append(
            (
                f'      <th style="text-align: left;">'
                f'<div style="resize: horizontal; overflow: auto; box-sizing: border-box; width: 100%; height: 100%; padding: 0.5em;">'
                f"{html.escape(str(col))}</div></th>"
            )
        )
    table_html.append("    </tr>")
    table_html.append("  </thead>")

    # Render table body
    table_html.append("  <tbody>")

    rowspans = _calculate_rowspans(dataframe)

    for row_idx, row_tuple in enumerate(dataframe.itertuples()):
        table_html.append("    <tr>")
        # First item in itertuples is the index, which can be a tuple for MultiIndex
        index_values = row_tuple[0]
        if not isinstance(index_values, tuple):
            index_values = (index_values,)

        for level_idx, value in enumerate(index_values):
            span = rowspans[level_idx][row_idx]
            if span > 0:
                # Only render the <th> if it's the start of a new span
                rowspan_attr = f' rowspan="{span}"' if span > 1 else ""
                table_html.append(
                    f'      <th{rowspan_attr} style="text-align: left; vertical-align: top; padding: 0.5em;">'
                    f"        {html.escape(str(value))}"
                    f"      </th>"
                )

        # The rest are the column values
        for i, value in enumerate(row_tuple[1:]):
            col_name = dataframe.columns[i]
            dtype = dataframe.dtypes.loc[col_name]  # type: ignore
            align = "right" if _is_dtype_numeric(dtype) else "left"
            table_html.append(
                '      <td style="text-align: {}; padding: 0.5em;">'.format(align)
            )

            if pandas.api.types.is_scalar(value) and pd.isna(value):
                table_html.append('        <em style="color: gray;">&lt;NA&gt;</em>')
            else:
                if isinstance(value, float):
                    formatted_value = f"{value:.{precision}f}"
                    table_html.append(f"        {html.escape(formatted_value)}")
                else:
                    table_html.append(f"        {html.escape(str(value))}")
            table_html.append("      </td>")
        table_html.append("    </tr>")
    table_html.append("  </tbody>")
    table_html.append("</table>")

    return "\n".join(table_html)
