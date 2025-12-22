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
import traceback
from typing import Any, Union
import warnings

import pandas as pd
import pandas.api.types

import bigframes
from bigframes._config import options
import bigframes.dataframe
from bigframes.display import plaintext
import bigframes.formatting_helpers
import bigframes.series


def _is_dtype_numeric(dtype: Any) -> bool:
    """Check if a dtype is numeric for alignment purposes."""
    return pandas.api.types.is_numeric_dtype(dtype)


def render_html(
    *,
    dataframe: pd.DataFrame,
    table_id: str,
    orderable_columns: list[str] | None = None,
) -> str:
    """Render a pandas DataFrame to HTML with specific styling."""
    classes = "dataframe table table-striped table-hover"
    table_html = [f'<table border="1" class="{classes}" id="{table_id}">']
    precision = options.display.precision
    orderable_columns = orderable_columns or []

    # Render table head
    table_html.append("  <thead>")
    table_html.append('    <tr style="text-align: left;">')
    for col in dataframe.columns:
        th_classes = []
        if col in orderable_columns:
            th_classes.append("sortable")
        class_str = f'class="{" ".join(th_classes)}"' if th_classes else ""
        header_div = (
            '<div style="resize: horizontal; overflow: auto; '
            "box-sizing: border-box; width: 100%; height: 100%; "
            'padding: 0.5em;">'
            f"{html.escape(str(col))}"
            "</div>"
        )
        table_html.append(
            f'      <th style="text-align: left;" {class_str}>{header_div}</th>'
        )
    table_html.append("    </tr>")
    table_html.append("  </thead>")

    # Render table body
    table_html.append("  <tbody>")
    for i in range(len(dataframe)):
        table_html.append("    <tr>")
        row = dataframe.iloc[i]
        for col_name, value in row.items():
            dtype = dataframe.dtypes.loc[col_name]  # type: ignore
            align = "right" if _is_dtype_numeric(dtype) else "left"
            table_html.append(
                '      <td style="text-align: {}; padding: 0.5em;">'.format(align)
            )

            # TODO(b/438181139): Consider semi-exploding ARRAY/STRUCT columns
            # into multiple rows/columns like the BQ UI does.
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


def create_html_representation(
    obj: Union[bigframes.dataframe.DataFrame, bigframes.series.Series],
    pandas_df: pd.DataFrame,
    total_rows: int,
    total_columns: int,
    blob_cols: list[str],
) -> str:
    """Create an HTML representation of the DataFrame or Series."""
    return obj._create_html_representation(
        pandas_df, total_rows, total_columns, blob_cols
    )


def get_anywidget_bundle(
    obj: Union[bigframes.dataframe.DataFrame, bigframes.series.Series],
    include=None,
    exclude=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Helper method to create and return the anywidget mimebundle.
    This function encapsulates the logic for anywidget display.
    """
    from bigframes import display

    if isinstance(obj, bigframes.series.Series):
        df = obj.to_frame()
    else:
        df, blob_cols = obj._get_display_df_and_blob_cols()

    widget = display.TableWidget(df)
    widget_repr_result = widget._repr_mimebundle_(include=include, exclude=exclude)

    if isinstance(widget_repr_result, tuple):
        widget_repr, widget_metadata = widget_repr_result
    else:
        widget_repr = widget_repr_result
        widget_metadata = {}

    widget_repr = dict(widget_repr)

    # Use cached data from widget to render HTML and plain text versions.
    cached_pd = widget._cached_data
    total_rows = widget.row_count
    total_columns = len(df.columns)

    widget_repr["text/html"] = create_html_representation(
        obj,
        cached_pd,
        total_rows,
        total_columns,
        blob_cols if "blob_cols" in locals() else [],
    )
    widget_repr["text/plain"] = plaintext.create_text_representation(
        obj, cached_pd, total_rows
    )

    return widget_repr, widget_metadata


def repr_mimebundle_head(
    obj: Union[bigframes.dataframe.DataFrame, bigframes.series.Series],
    include=None,
    exclude=None,
):
    """Mimebundle display for the start of the data."""
    opts = bigframes.options.display
    if opts.repr_mode == "anywidget":
        try:
            return get_anywidget_bundle(obj, include=include, exclude=exclude)
        except ImportError:
            warnings.warn(
                "Anywidget mode is not available. "
                "Please `pip install anywidget traitlets` or `pip install 'bigframes[anywidget]'` to use interactive tables. "
                f"Falling back to static HTML. Error: {traceback.format_exc()}"
            )

    blob_cols: list[str]
    if isinstance(obj, bigframes.series.Series):
        pandas_df, row_count, query_job = obj._block.retrieve_repr_request_results(
            opts.max_rows
        )
        blob_cols = []
    else:
        df, blob_cols = obj._get_display_df_and_blob_cols()
        pandas_df, row_count, query_job = df._block.retrieve_repr_request_results(
            opts.max_rows
        )

    obj._set_internal_query_job(query_job)
    column_count = len(pandas_df.columns)

    html_string = create_html_representation(
        obj, pandas_df, row_count, column_count, blob_cols
    )

    text_representation = plaintext.create_text_representation(
        obj, pandas_df, row_count
    )

    return {"text/html": html_string, "text/plain": text_representation}


def repr_mimebundle_deferred(
    obj: Union[bigframes.dataframe.DataFrame, bigframes.series.Series],
    include=None,
    exclude=None,
):
    """Mimebundle display for deferred execution mode."""
    # We don't need the mimetype for the deferred case, but we need to match
    # the signature of the other repr_mimebundle_* methods.
    # TODO(swast): Add an HTML representation for deferred queries that simply
    # prints the SQL.
    query_job = obj._compute_dry_run()
    text_representation = bigframes.formatting_helpers.repr_query_job(query_job)
    return {"text/plain": text_representation}


def repr_mimebundle(
    obj: Union[bigframes.dataframe.DataFrame, bigframes.series.Series],
    include=None,
    exclude=None,
):
    """
    Custom display method for IPython/Jupyter environments.
    """
    if options.display.repr_mode == "deferred":
        return repr_mimebundle_deferred(obj, include=include, exclude=exclude)
    else:
        return repr_mimebundle_head(obj, include=include, exclude=exclude)
