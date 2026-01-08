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

"""Utilities for flattening nested data structures for display."""

from __future__ import annotations

import dataclasses
from typing import cast

import pandas as pd
import pyarrow as pa


@dataclasses.dataclass(frozen=True)
class FlattenResult:
    """The result of flattening a DataFrame."""

    dataframe: pd.DataFrame
    """The flattened DataFrame."""

    row_labels: list[str] | None
    """A list of original row labels for each row in the flattened DataFrame."""

    continuation_rows: set[int] | None
    """A set of row indices that are continuation rows."""

    cleared_on_continuation: list[str]
    """A list of column names that should be cleared on continuation rows."""

    nested_columns: set[str]
    """A set of column names that were created from nested data."""


@dataclasses.dataclass(frozen=True)
class ColumnClassification:
    """The result of classifying columns."""

    struct_columns: list[str]
    """Columns that are STRUCTs."""

    array_columns: list[str]
    """Columns that are ARRAYs."""

    array_of_struct_columns: list[str]
    """Columns that are ARRAYs of STRUCTs."""

    clear_on_continuation_cols: list[str]
    """Columns that should be cleared on continuation rows."""

    nested_originated_columns: set[str]
    """Columns that were created from nested data."""


@dataclasses.dataclass(frozen=True)
class ExplodeResult:
    """The result of exploding array columns."""

    dataframe: pd.DataFrame
    """The exploded DataFrame."""

    row_labels: list[str]
    """Labels for the rows."""

    continuation_rows: set[int]
    """Indices of continuation rows."""


def flatten_nested_data(
    dataframe: pd.DataFrame,
) -> FlattenResult:
    """Flatten nested STRUCT and ARRAY columns for display."""
    if dataframe.empty:
        return FlattenResult(
            dataframe=dataframe.copy(),
            row_labels=None,
            continuation_rows=None,
            cleared_on_continuation=[],
            nested_columns=set(),
        )

    result_df = dataframe.copy()

    classification = _classify_columns(result_df)
    # Extract lists to allow modification
    # TODO(b/469966526): The modification of these lists in place by subsequent functions
    # (e.g. _flatten_array_of_struct_columns removing items from array_columns) suggests
    # that the data flow here could be cleaner, but keeping it as is for now.
    struct_columns = classification.struct_columns
    array_columns = classification.array_columns
    array_of_struct_columns = classification.array_of_struct_columns
    clear_on_continuation_cols = classification.clear_on_continuation_cols
    nested_originated_columns = classification.nested_originated_columns

    result_df, array_columns = _flatten_array_of_struct_columns(
        result_df, array_of_struct_columns, array_columns, nested_originated_columns
    )

    result_df, clear_on_continuation_cols = _flatten_struct_columns(
        result_df, struct_columns, clear_on_continuation_cols, nested_originated_columns
    )

    # Now handle ARRAY columns (including the newly created ones from ARRAY of STRUCT)
    if not array_columns:
        return FlattenResult(
            dataframe=result_df,
            row_labels=None,
            continuation_rows=None,
            cleared_on_continuation=clear_on_continuation_cols,
            nested_columns=nested_originated_columns,
        )

    explode_result = _explode_array_columns(result_df, array_columns)
    return FlattenResult(
        dataframe=explode_result.dataframe,
        row_labels=explode_result.row_labels,
        continuation_rows=explode_result.continuation_rows,
        cleared_on_continuation=clear_on_continuation_cols,
        nested_columns=nested_originated_columns,
    )


def _classify_columns(
    dataframe: pd.DataFrame,
) -> ColumnClassification:
    """Identify all STRUCT and ARRAY columns."""
    initial_columns = list(dataframe.columns)
    struct_columns: list[str] = []
    array_columns: list[str] = []
    array_of_struct_columns: list[str] = []
    clear_on_continuation_cols: list[str] = []
    nested_originated_columns: set[str] = set()

    for col_name_raw, col_data in dataframe.items():
        col_name = str(col_name_raw)
        dtype = col_data.dtype
        if isinstance(dtype, pd.ArrowDtype):
            pa_type = dtype.pyarrow_dtype
            if pa.types.is_struct(pa_type):
                struct_columns.append(col_name)
                nested_originated_columns.add(col_name)
            elif pa.types.is_list(pa_type):
                array_columns.append(col_name)
                nested_originated_columns.add(col_name)
                if hasattr(pa_type, "value_type") and (
                    pa.types.is_struct(pa_type.value_type)
                ):
                    array_of_struct_columns.append(col_name)
            else:
                clear_on_continuation_cols.append(col_name)
        elif col_name in initial_columns:
            clear_on_continuation_cols.append(col_name)
    return ColumnClassification(
        struct_columns=struct_columns,
        array_columns=array_columns,
        array_of_struct_columns=array_of_struct_columns,
        clear_on_continuation_cols=clear_on_continuation_cols,
        nested_originated_columns=nested_originated_columns,
    )


def _flatten_array_of_struct_columns(
    dataframe: pd.DataFrame,
    array_of_struct_columns: list[str],
    array_columns: list[str],
    nested_originated_columns: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Flatten ARRAY of STRUCT columns into separate array columns for each field."""
    result_df = dataframe.copy()
    for col_name in array_of_struct_columns:
        col_data = result_df[col_name]
        pa_type = cast(pd.ArrowDtype, col_data.dtype).pyarrow_dtype
        struct_type = pa_type.value_type

        # Use PyArrow to reshape the list<struct> into multiple list<field> arrays
        arrow_array = pa.array(col_data)
        offsets = arrow_array.offsets
        values = arrow_array.values  # StructArray
        flattened_fields = values.flatten()  # List[Array]

        new_cols_to_add = {}
        new_array_col_names = []

        # Create new columns for each struct field
        for field_idx in range(struct_type.num_fields):
            field = struct_type.field(field_idx)
            new_col_name = f"{col_name}.{field.name}"
            nested_originated_columns.add(new_col_name)
            new_array_col_names.append(new_col_name)

            # Reconstruct ListArray for this field. This transforms the
            # array<struct<f1, f2>> into separate array<f1> and array<f2> columns.
            new_list_array = pa.ListArray.from_arrays(
                offsets, flattened_fields[field_idx], mask=arrow_array.is_null()
            )

            new_cols_to_add[new_col_name] = pd.Series(
                new_list_array,
                dtype=pd.ArrowDtype(pa.list_(field.type)),
                index=result_df.index,
            )

        col_idx = result_df.columns.to_list().index(col_name)
        new_cols_df = pd.DataFrame(new_cols_to_add, index=result_df.index)

        result_df = pd.concat(
            [
                result_df.iloc[:, :col_idx],
                new_cols_df,
                result_df.iloc[:, col_idx + 1 :],
            ],
            axis=1,
        )

        # Update array_columns list
        array_columns.remove(col_name)
        # Add the new array columns
        array_columns.extend(new_array_col_names)
    return result_df, array_columns


def _explode_array_columns(
    dataframe: pd.DataFrame, array_columns: list[str]
) -> ExplodeResult:
    """Explode array columns into new rows."""
    if not array_columns:
        return ExplodeResult(dataframe, [], set())

    original_cols = dataframe.columns.tolist()
    work_df = dataframe

    non_array_columns = work_df.columns.drop(array_columns).tolist()
    if not non_array_columns:
        work_df = work_df.copy()  # Avoid modifying input
        # Add a temporary column to allow grouping if all columns are arrays
        non_array_columns = ["_temp_grouping_col"]
        work_df["_temp_grouping_col"] = range(len(work_df))

    # Preserve original index
    if work_df.index.name:
        original_index_name = work_df.index.name
        work_df = work_df.reset_index()
        non_array_columns.append(original_index_name)
    else:
        original_index_name = None
        work_df = work_df.reset_index(names=["_original_index"])
        non_array_columns.append("_original_index")

    exploded_dfs = []
    for col in array_columns:
        # Explode each array column individually
        col_series = work_df[col]
        target_dtype = None
        if isinstance(col_series.dtype, pd.ArrowDtype):
            pa_type = col_series.dtype.pyarrow_dtype
            if pa.types.is_list(pa_type):
                target_dtype = pd.ArrowDtype(pa_type.value_type)
            # Use to_list() to avoid pandas attempting to create a 2D numpy
            # array if the list elements have the same length.
            col_series = pd.Series(
                col_series.to_list(), index=col_series.index, dtype=object
            )

        exploded = work_df[non_array_columns].assign(**{col: col_series}).explode(col)

        if target_dtype is not None:
            # Re-cast to arrow dtype if possible
            exploded[col] = exploded[col].astype(target_dtype)

        exploded["_row_num"] = exploded.groupby(non_array_columns).cumcount()
        exploded_dfs.append(exploded)

    if not exploded_dfs:
        # This should not be reached if array_columns is not empty
        return ExplodeResult(dataframe, [], set())

    # Merge the exploded columns
    merged_df = exploded_dfs[0]
    for i in range(1, len(exploded_dfs)):
        merged_df = pd.merge(
            merged_df,
            exploded_dfs[i],
            on=non_array_columns + ["_row_num"],
            how="outer",
        )

    # Restore original column order and sort
    merged_df = merged_df.sort_values(non_array_columns + ["_row_num"]).reset_index(
        drop=True
    )

    # Generate row labels and continuation mask efficiently
    grouping_col_name = (
        "_original_index" if original_index_name is None else original_index_name
    )
    row_labels = merged_df[grouping_col_name].astype(str).tolist()
    continuation_rows = set(merged_df.index[merged_df["_row_num"] > 0])

    # Restore original columns
    result_df = merged_df[original_cols]

    if original_index_name:
        result_df = result_df.set_index(original_index_name)

    return ExplodeResult(result_df, row_labels, continuation_rows)


def _flatten_struct_columns(
    dataframe: pd.DataFrame,
    struct_columns: list[str],
    clear_on_continuation_cols: list[str],
    nested_originated_columns: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Flatten regular STRUCT columns."""
    result_df = dataframe.copy()
    for col_name in struct_columns:
        col_data = result_df[col_name]
        if isinstance(col_data.dtype, pd.ArrowDtype):
            pa_type = cast(pd.ArrowDtype, col_data.dtype).pyarrow_dtype

        # Use PyArrow to flatten the struct column without row iteration
        # combine_chunks() ensures we have a single array if it was chunked
        arrow_array = pa.array(col_data)
        flattened_fields = arrow_array.flatten()

        new_cols_to_add = {}
        for field_idx in range(pa_type.num_fields):
            field = pa_type.field(field_idx)
            new_col_name = f"{col_name}.{field.name}"
            nested_originated_columns.add(new_col_name)
            clear_on_continuation_cols.append(new_col_name)

            # Create a new Series from the flattened array
            new_cols_to_add[new_col_name] = pd.Series(
                flattened_fields[field_idx],
                dtype=pd.ArrowDtype(field.type),
                index=result_df.index,
            )

        col_idx = result_df.columns.to_list().index(col_name)
        new_cols_df = pd.DataFrame(new_cols_to_add, index=result_df.index)
        result_df = pd.concat(
            [
                result_df.iloc[:, :col_idx],
                new_cols_df,
                result_df.iloc[:, col_idx + 1 :],
            ],
            axis=1,
        )
    return result_df, clear_on_continuation_cols
