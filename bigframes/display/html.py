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


def render_html(
    *,
    dataframe: pd.DataFrame,
    table_id: str,
) -> str:
    """Render a pandas DataFrame to HTML with specific styling."""
    classes = "dataframe table table-striped table-hover"
    table_html = [f'<table border="1" class="{classes}" id="{table_id}">']
    precision = options.display.precision

    # Render table head
    table_html.append("  <thead>")
    table_html.append('    <tr style="text-align: left;">')

    # Add index headers
    for name in dataframe.index.names:
        table_html.append(
            f'      <th style="text-align: left;"><div style="resize: horizontal; overflow: auto; box-sizing: border-box; width: 100%; height: 100%; padding: 0.5em;">{html.escape(str(name))}</div></th>'
        )

    for col in dataframe.columns:
        table_html.append(
            f'      <th style="text-align: left;"><div style="resize: horizontal; overflow: auto; box-sizing: border-box; width: 100%; height: 100%; padding: 0.5em;">{html.escape(str(col))}</div></th>'
        )
    table_html.append("    </tr>")
    table_html.append("  </thead>")

    # Render table body
    table_html.append("  <tbody>")
    for row_tuple in dataframe.itertuples():
        table_html.append("    <tr>")
        # First item in itertuples is the index, which can be a tuple for MultiIndex
        index_values = row_tuple[0]
        if not isinstance(index_values, tuple):
            index_values = (index_values,)

        for value in index_values:
            table_html.append('      <td style="text-align: left; padding: 0.5em;">')
            table_html.append(f"        {html.escape(str(value))}")
            table_html.append("      </td>")

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
