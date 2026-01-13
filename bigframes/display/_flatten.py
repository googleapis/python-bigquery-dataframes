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
import enum

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc  # type: ignore


@dataclasses.dataclass(frozen=True)
class FlattenResult:
    """The result of flattening a DataFrame.

    Attributes:
        dataframe: The flattened DataFrame. If the original DataFrame had an index
            (including MultiIndex), it is preserved in this flattened DataFrame,
            duplicated across exploded rows as needed.
        row_labels: A list of original row labels for each row in the flattened DataFrame.
            This corresponds to the original index values (stringified) and serves to
            visually group the exploded rows that belong to the same original row.
        continuation_rows: A set of row indices in the flattened DataFrame that are
            "continuation rows". These are additional rows created to display the
            2nd to Nth elements of an array. The first row (index i-1) contains
            the 1st element, while these rows contain subsequent elements.
        cleared_on_continuation: A list of column names that should be "cleared"
            (displayed as empty) on continuation rows. Typically, these are
            scalar columns (non-array) that were replicated during the explosion
            process but should only be visually displayed once per original row group.
        nested_columns: A set of column names that were created from nested data
            (flattened structs or arrays).
    """

    dataframe: pd.DataFrame
    row_labels: list[str] | None
    continuation_rows: set[int] | None
    cleared_on_continuation: list[str]
    nested_columns: set[str]


class _ColumnCategory(enum.Enum):
    STRUCT = "struct"
    ARRAY = "array"
    ARRAY_OF_STRUCT = "array_of_struct"
    CLEAR = "clear"


@dataclasses.dataclass(frozen=True)
class _ColumnClassification:
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
) -> _ColumnClassification:
    """Identify all STRUCT and ARRAY columns in the DataFrame.

    Args:
        dataframe: The DataFrame to inspect.

    Returns:
        A _ColumnClassification object containing lists of column names for each category.
    """

    def get_category(dtype: pd.api.extensions.ExtensionDtype) -> _ColumnCategory:
        pa_type = getattr(dtype, "pyarrow_dtype", None)
        if pa_type:
            if pa.types.is_struct(pa_type):
                return _ColumnCategory.STRUCT
            if pa.types.is_list(pa_type):
                return (
                    _ColumnCategory.ARRAY_OF_STRUCT
                    if pa.types.is_struct(pa_type.value_type)
                    else _ColumnCategory.ARRAY
                )
        return _ColumnCategory.CLEAR

    # Maps column names to their structural category to simplify list building.
    categories = {
        str(col): get_category(dtype) for col, dtype in dataframe.dtypes.items()
    }

    return _ColumnClassification(
        struct_columns=tuple(
            c for c, cat in categories.items() if cat == _ColumnCategory.STRUCT
        ),
        array_columns=tuple(
            c
            for c, cat in categories.items()
            if cat in (_ColumnCategory.ARRAY, _ColumnCategory.ARRAY_OF_STRUCT)
        ),
        array_of_struct_columns=tuple(
            c for c, cat in categories.items() if cat == _ColumnCategory.ARRAY_OF_STRUCT
        ),
        clear_on_continuation_cols=tuple(
            c for c, cat in categories.items() if cat == _ColumnCategory.CLEAR
        ),
        nested_originated_columns=frozenset(
            c for c, cat in categories.items() if cat != _ColumnCategory.CLEAR
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

    work_df, non_array_columns, index_names = _prepare_explosion_dataframe(
        dataframe, array_columns
    )

    if work_df.empty:
        return ExplodeResult(dataframe, [], set())

    table = pa.Table.from_pandas(work_df)
    arrays = [table.column(col).combine_chunks() for col in array_columns]
    lengths = []
    for arr in arrays:
        row_lengths = pc.list_value_length(arr)
        # Treat null lists as length 1 to match pandas explode behavior for scalars.
        row_lengths = pc.if_else(
            pc.is_null(row_lengths, nan_is_null=True), 1, row_lengths
        )
        lengths.append(row_lengths)

    if not lengths:
        return ExplodeResult(dataframe, [], set())

    max_lens = lengths[0] if len(lengths) == 1 else pc.max_element_wise(*lengths)
    max_lens = max_lens.cast(pa.int64())
    current_offsets = pc.cumulative_sum(max_lens)
    target_offsets = pa.concat_arrays([pa.array([0], type=pa.int64()), current_offsets])

    total_rows = target_offsets[-1].as_py()
    if total_rows == 0:
        empty_df = pd.DataFrame(columns=dataframe.columns)
        if index_names:
            empty_df = empty_df.set_index(index_names)
        return ExplodeResult(empty_df, [], set())

    # parent_indices maps each result row to its original row index.
    dummy_values = pa.nulls(total_rows, type=pa.null())
    dummy_list_array = pa.ListArray.from_arrays(target_offsets, dummy_values)
    parent_indices = pc.list_parent_indices(dummy_list_array)

    range_k = pa.array(range(total_rows))
    starts = target_offsets.take(parent_indices)
    row_nums = pc.subtract(range_k, starts)

    new_columns = {}
    for col_name in non_array_columns:
        new_columns[col_name] = table.column(col_name).take(parent_indices)

    for col_name, arr in zip(array_columns, arrays):
        actual_lens_scattered = pc.list_value_length(arr).take(parent_indices)
        valid_mask = pc.less(row_nums, actual_lens_scattered)
        starts_scattered = arr.offsets.take(parent_indices)

        # safe_mask ensures we don't access out of bounds even if masked out.
        safe_mask = pc.fill_null(valid_mask, False)
        candidate_indices = pc.add(starts_scattered, row_nums)
        safe_indices = pc.if_else(safe_mask, candidate_indices, 0)

        if len(arr.values) == 0:
            final_values = pa.nulls(total_rows, type=arr.type.value_type)
        else:
            taken_values = arr.values.take(safe_indices)
            final_values = pc.if_else(safe_mask, taken_values, None)

        new_columns[col_name] = final_values

    # Convert back to pandas; this is efficient since we have pyarrow arrays.
    result_table = pa.Table.from_pydict(new_columns)
    result_df = result_table.to_pandas(types_mapper=pd.ArrowDtype)

    if index_names:
        if len(index_names) == 1:
            row_labels = result_df[index_names[0]].astype(str).tolist()
        else:
            # For MultiIndex, create a tuple string representation
            row_labels = (
                result_df[index_names].apply(tuple, axis=1).astype(str).tolist()
            )
    else:
        row_labels = result_df["_original_index"].astype(str).tolist()

    continuation_mask = pc.greater(row_nums, 0).to_numpy(zero_copy_only=False)
    continuation_rows = set(np.flatnonzero(continuation_mask).tolist())

    # Select columns: original columns + restored index columns (temporarily)
    cols_to_keep = dataframe.columns.tolist()
    if index_names:
        cols_to_keep.extend(index_names)

    # Filter columns, but allow index columns to pass through if they are not in original columns
    # (which they won't be if they were indices)
    result_df = result_df[cols_to_keep]

    if index_names:
        result_df = result_df.set_index(index_names)

    return ExplodeResult(result_df, row_labels, continuation_rows)


def _prepare_explosion_dataframe(
    dataframe: pd.DataFrame, array_columns: list[str]
) -> tuple[pd.DataFrame, list[str], list[str] | None]:
    """Prepares the DataFrame for explosion by ensuring grouping columns exist."""
    work_df = dataframe.copy()
    non_array_columns = work_df.columns.drop(array_columns).tolist()

    if not non_array_columns:
        # Add a temporary column to allow grouping if all columns are arrays.
        non_array_columns = ["_temp_grouping_col"]
        work_df["_temp_grouping_col"] = range(len(work_df))

    index_names = None
    if work_df.index.nlevels > 1:
        # Handle MultiIndex
        names = list(work_df.index.names)
        # Assign default names if None to ensure reset_index works and we can track them
        names = [n if n is not None else f"level_{i}" for i, n in enumerate(names)]
        work_df.index.names = names
        index_names = names
        work_df = work_df.reset_index()
        non_array_columns.extend(index_names)
    elif work_df.index.name is not None:
        # Handle named Index
        index_names = [work_df.index.name]
        work_df = work_df.reset_index()
        non_array_columns.extend(index_names)
    else:
        # Handle default/unnamed Index
        # We use _original_index for tracking but don't return it as an index to restore
        work_df = work_df.reset_index(names=["_original_index"])
        non_array_columns.append("_original_index")

    return work_df, non_array_columns, index_names


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
    if not struct_columns:
        return FlattenStructsResult(
            dataframe=dataframe.copy(),
            clear_on_continuation_cols=clear_on_continuation_cols,
            nested_originated_columns=nested_originated_columns,
        )

    # Convert to PyArrow table for efficient flattening
    table = pa.Table.from_pandas(dataframe, preserve_index=False)

    current_clear_cols = list(clear_on_continuation_cols)
    current_nested_cols = set(nested_originated_columns)

    # Identify new columns that will be created to update metadata
    for col_name in struct_columns:
        idx = table.schema.get_field_index(col_name)
        if idx == -1:
            continue

        field = table.schema.field(idx)
        if pa.types.is_struct(field.type):
            for i in range(field.type.num_fields):
                child_field = field.type.field(i)
                new_col_name = f"{col_name}.{child_field.name}"
                current_nested_cols.add(new_col_name)
                current_clear_cols.append(new_col_name)

    # Expand all struct columns into "parent.child" columns.
    flattened_table = table.flatten()

    # Convert back to pandas, using ArrowDtype to preserve types and ignoring metadata
    # to avoid issues with stale struct type info.
    result_df = flattened_table.to_pandas(
        types_mapper=pd.ArrowDtype, ignore_metadata=True
    )

    result_df.index = dataframe.index

    return FlattenStructsResult(
        dataframe=result_df,
        clear_on_continuation_cols=tuple(current_clear_cols),
        nested_originated_columns=frozenset(current_nested_cols),
    )
