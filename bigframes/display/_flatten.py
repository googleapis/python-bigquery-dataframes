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

"""Utilities for flattening nested data structures for display.

This module provides functionality to flatten BigQuery STRUCT and ARRAY columns
in a pandas DataFrame into a format suitable for display in a 2D table widget.
It handles nested structures by:
1.  Expanding STRUCT fields into separate columns (e.g., "struct.field").
2.  Exploding ARRAY elements into multiple rows, replicating other columns.
3.  Generating metadata to grouping rows and handling continuation values.
"""

from __future__ import annotations

import dataclasses
from typing import cast

import pandas as pd
import pyarrow as pa


@dataclasses.dataclass(frozen=True)
class FlattenResult:
    """The result of flattening a DataFrame.

    Attributes:
        dataframe: The flattened DataFrame.
        row_labels: A list of original row labels for each row in the flattened DataFrame.
        continuation_rows: A set of row indices that are continuation rows.
        cleared_on_continuation: A list of column names that should be cleared on continuation rows.
        nested_columns: A set of column names that were created from nested data.
    """

    dataframe: pd.DataFrame
    row_labels: list[str] | None
    continuation_rows: set[int] | None
    cleared_on_continuation: list[str]
    nested_columns: set[str]


@dataclasses.dataclass(frozen=True)
class ColumnClassification:
    """The result of classifying columns.

    Attributes:
        struct_columns: Columns that are STRUCTs.
        array_columns: Columns that are ARRAYs.
        array_of_struct_columns: Columns that are ARRAYs of STRUCTs.
        clear_on_continuation_cols: Columns that should be cleared on continuation rows.
        nested_originated_columns: Columns that were created from nested data.
    """

    struct_columns: tuple[str, ...]
    array_columns: tuple[str, ...]
    array_of_struct_columns: tuple[str, ...]
    clear_on_continuation_cols: tuple[str, ...]
    nested_originated_columns: frozenset[str]


@dataclasses.dataclass(frozen=True)
class FlattenArrayOfStructsResult:
    """The result of flattening array-of-struct columns.

    Attributes:
        dataframe: The flattened DataFrame.
        array_columns: The updated list of array columns.
        nested_originated_columns: The updated set of columns created from nested data.
    """

    dataframe: pd.DataFrame
    array_columns: tuple[str, ...]
    nested_originated_columns: frozenset[str]


@dataclasses.dataclass(frozen=True)
class FlattenStructsResult:
    """The result of flattening struct columns.

    Attributes:
        dataframe: The flattened DataFrame.
        clear_on_continuation_cols: The updated list of columns to clear on continuation.
        nested_originated_columns: The updated set of columns created from nested data.
    """

    dataframe: pd.DataFrame
    clear_on_continuation_cols: tuple[str, ...]
    nested_originated_columns: frozenset[str]


def flatten_nested_data(
    dataframe: pd.DataFrame,
) -> FlattenResult:
    """Flatten nested STRUCT and ARRAY columns for display.

    Args:
        dataframe: The input DataFrame containing potential nested structures.

    Returns:
        A FlattenResult containing the flattened DataFrame and metadata for display.
    """
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

    # Process ARRAY-of-STRUCT columns into multiple ARRAY columns (one per struct field).
    flatten_array_structs_result = _flatten_array_of_struct_columns(
        result_df,
        classification.array_of_struct_columns,
        classification.array_columns,
        classification.nested_originated_columns,
    )
    result_df = flatten_array_structs_result.dataframe
    classification = dataclasses.replace(
        classification,
        array_columns=flatten_array_structs_result.array_columns,
        nested_originated_columns=flatten_array_structs_result.nested_originated_columns,
    )

    # Flatten top-level STRUCT columns into separate columns.
    flatten_structs_result = _flatten_struct_columns(
        result_df,
        classification.struct_columns,
        classification.clear_on_continuation_cols,
        classification.nested_originated_columns,
    )
    result_df = flatten_structs_result.dataframe
    classification = dataclasses.replace(
        classification,
        clear_on_continuation_cols=flatten_structs_result.clear_on_continuation_cols,
        nested_originated_columns=flatten_structs_result.nested_originated_columns,
    )

    # Now handle ARRAY columns (including the newly created ones from ARRAY of STRUCT)
    if not classification.array_columns:
        return FlattenResult(
            dataframe=result_df,
            row_labels=None,
            continuation_rows=None,
            cleared_on_continuation=list(classification.clear_on_continuation_cols),
            nested_columns=set(classification.nested_originated_columns),
        )

    explode_result = _explode_array_columns(
        result_df, list(classification.array_columns)
    )
    return FlattenResult(
        dataframe=explode_result.dataframe,
        row_labels=explode_result.row_labels,
        continuation_rows=explode_result.continuation_rows,
        cleared_on_continuation=list(classification.clear_on_continuation_cols),
        nested_columns=set(classification.nested_originated_columns),
    )


def _classify_columns(
    dataframe: pd.DataFrame,
) -> ColumnClassification:
    """Identify all STRUCT and ARRAY columns in the DataFrame.

    Args:
        dataframe: The DataFrame to inspect.

    Returns:
        A ColumnClassification object containing lists of column names for each category.
    """

    def get_category(dtype: pd.api.extensions.ExtensionDtype) -> str:
        pa_type = getattr(dtype, "pyarrow_dtype", None)
        if pa_type:
            if pa.types.is_struct(pa_type):
                return "struct"
            if pa.types.is_list(pa_type):
                return (
                    "array_of_struct"
                    if pa.types.is_struct(pa_type.value_type)
                    else "array"
                )
        return "clear"

    # Maps column names to their structural category to simplify list building.
    categories = {
        str(col): get_category(dtype) for col, dtype in dataframe.dtypes.items()
    }

    return ColumnClassification(
        struct_columns=tuple(c for c, cat in categories.items() if cat == "struct"),
        array_columns=tuple(
            c for c, cat in categories.items() if cat in ("array", "array_of_struct")
        ),
        array_of_struct_columns=tuple(
            c for c, cat in categories.items() if cat == "array_of_struct"
        ),
        clear_on_continuation_cols=tuple(
            c for c, cat in categories.items() if cat == "clear"
        ),
        nested_originated_columns=frozenset(
            c for c, cat in categories.items() if cat != "clear"
        ),
    )


def _flatten_array_of_struct_columns(
    dataframe: pd.DataFrame,
    array_of_struct_columns: tuple[str, ...],
    array_columns: tuple[str, ...],
    nested_originated_columns: frozenset[str],
) -> FlattenArrayOfStructsResult:
    """Flatten ARRAY of STRUCT columns into separate ARRAY columns for each field.

    Args:
        dataframe: The DataFrame to process.
        array_of_struct_columns: Column names that are ARRAYs of STRUCTs.
        array_columns: The main sequence of ARRAY columns to be updated.
        nested_originated_columns: Columns tracked as originating from nested data.

    Returns:
        A FlattenArrayOfStructsResult containing the updated DataFrame and columns.
    """
    result_df = dataframe.copy()
    current_array_columns = list(array_columns)
    current_nested_columns = set(nested_originated_columns)

    for col_name in array_of_struct_columns:
        col_data = result_df[col_name]
        # Ensure we have a PyArrow array (pa.array handles pandas Series conversion)
        arrow_array = pa.array(col_data)

        # Transpose List<Struct<...>> to {field: List<field_type>}
        new_arrays = _transpose_list_of_structs(arrow_array)

        new_cols_df = pd.DataFrame(
            {
                f"{col_name}.{field_name}": pd.Series(
                    arr, dtype=pd.ArrowDtype(arr.type), index=result_df.index
                )
                for field_name, arr in new_arrays.items()
            }
        )

        current_nested_columns.update(new_cols_df.columns)
        result_df = _replace_column_in_df(result_df, col_name, new_cols_df)

        current_array_columns.remove(col_name)
        current_array_columns.extend(new_cols_df.columns.tolist())

    return FlattenArrayOfStructsResult(
        dataframe=result_df,
        array_columns=tuple(current_array_columns),
        nested_originated_columns=frozenset(current_nested_columns),
    )


def _transpose_list_of_structs(arrow_array: pa.ListArray) -> dict[str, pa.ListArray]:
    """Transposes a ListArray of Structs into multiple ListArrays of fields.

    Args:
        arrow_array: A PyArrow ListArray where the value type is a Struct.

    Returns:
        A dictionary mapping field names to new ListArrays (one for each field in the struct).
    """
    struct_type = arrow_array.type.value_type
    offsets = arrow_array.offsets
    # arrow_array.values is the underlying StructArray.
    # Flattening it gives us the arrays for each field, effectively "removing" the struct layer.
    flattened_fields = arrow_array.values.flatten()
    validity = arrow_array.is_null()

    transposed = {}
    for i in range(struct_type.num_fields):
        field = struct_type.field(i)
        # Reconstruct ListArray for each field using original offsets and validity.
        # This transforms List<Struct<A, B>> into List<A> and List<B>.
        transposed[field.name] = pa.ListArray.from_arrays(
            offsets, flattened_fields[i], mask=validity
        )
    return transposed


def _replace_column_in_df(
    dataframe: pd.DataFrame, col_name: str, new_cols: pd.DataFrame
) -> pd.DataFrame:
    """Replaces a column in a DataFrame with a set of new columns at the same position.

    Args:
        dataframe: The original DataFrame.
        col_name: The name of the column to replace.
        new_cols: A DataFrame containing the new columns to insert.

    Returns:
        A new DataFrame with the substitution made.
    """
    col_idx = dataframe.columns.to_list().index(col_name)
    return pd.concat(
        [
            dataframe.iloc[:, :col_idx],
            new_cols,
            dataframe.iloc[:, col_idx + 1 :],
        ],
        axis=1,
    )


@dataclasses.dataclass(frozen=True)
class ExplodeResult:
    """The result of exploding array columns.

    Attributes:
        dataframe: The exploded DataFrame.
        row_labels: Labels for the rows.
        continuation_rows: Indices of continuation rows.
    """

    dataframe: pd.DataFrame
    row_labels: list[str]
    continuation_rows: set[int]


def _explode_array_columns(
    dataframe: pd.DataFrame, array_columns: list[str]
) -> ExplodeResult:
    """Explode array columns into new rows.

    This function performs the "flattening" of 1D arrays by exploding them.
    It handles multiple array columns by ensuring they are exploded in sync
    relative to the other columns.

    Args:
        dataframe: The DataFrame to explode.
        array_columns: List of array columns to explode.

    Returns:
        An ExplodeResult containing the new DataFrame and row metadata.
    """
    if not array_columns:
        return ExplodeResult(dataframe, [], set())

    # Group by all non-array columns to maintain context.
    # _row_num tracks the index within the exploded array to synchronize multiple
    # arrays. Continuation rows (index > 0) are tracked for display clearing.
    original_cols = dataframe.columns.tolist()
    work_df = dataframe

    non_array_columns = work_df.columns.drop(array_columns).tolist()
    if not non_array_columns:
        work_df = work_df.copy()  # Avoid modifying input
        # Add a temporary column to allow grouping if all columns are arrays.
        # This ensures we can still group by "original row" even if there are no scalar columns.
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

        # Track position in the array for alignment
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
    struct_columns: tuple[str, ...],
    clear_on_continuation_cols: tuple[str, ...],
    nested_originated_columns: frozenset[str],
) -> FlattenStructsResult:
    """Flatten regular STRUCT columns into separate columns.

    Args:
        dataframe: The DataFrame to process.
        struct_columns: STRUCT columns to flatten.
        clear_on_continuation_cols: Columns to clear on continuation.
        nested_originated_columns: Columns tracked as originating from nested data.

    Returns:
        A FlattenStructsResult containing the updated DataFrame and columns.
    """
    result_df = dataframe.copy()
    current_clear_cols = list(clear_on_continuation_cols)
    current_nested_cols = set(nested_originated_columns)

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
            current_nested_cols.add(new_col_name)
            current_clear_cols.append(new_col_name)

            # Create a new Series from the flattened array
            new_cols_to_add[new_col_name] = pd.Series(
                flattened_fields[field_idx],
                dtype=pd.ArrowDtype(field.type),
                index=result_df.index,
            )

        new_cols_df = pd.DataFrame(new_cols_to_add, index=result_df.index)
        result_df = _replace_column_in_df(result_df, col_name, new_cols_df)

    return FlattenStructsResult(
        dataframe=result_df,
        clear_on_continuation_cols=tuple(current_clear_cols),
        nested_originated_columns=frozenset(current_nested_cols),
    )
