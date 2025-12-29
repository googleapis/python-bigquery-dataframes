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

from typing import cast

import pandas as pd
import pyarrow as pa


def flatten_nested_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[int]], list[str], set[str]]:
    """Flatten nested STRUCT and ARRAY columns for display."""
    if dataframe.empty:
        return dataframe.copy(), {}, [], set()

    result_df = dataframe.copy()

    (
        struct_columns,
        array_columns,
        array_of_struct_columns,
        clear_on_continuation_cols,
        nested_originated_columns,
    ) = _classify_columns(result_df)

    result_df, array_columns = _flatten_array_of_struct_columns(
        result_df, array_of_struct_columns, array_columns, nested_originated_columns
    )

    result_df, clear_on_continuation_cols = _flatten_struct_columns(
        result_df, struct_columns, clear_on_continuation_cols, nested_originated_columns
    )

    # Now handle ARRAY columns (including the newly created ones from ARRAY of STRUCT)
    if not array_columns:
        return (
            result_df,
            {},
            clear_on_continuation_cols,
            nested_originated_columns,
        )

    result_df, array_row_groups = _explode_array_columns(result_df, array_columns)
    return (
        result_df,
        array_row_groups,
        clear_on_continuation_cols,
        nested_originated_columns,
    )


def _classify_columns(
    dataframe: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str], set[str]]:
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
    return (
        struct_columns,
        array_columns,
        array_of_struct_columns,
        clear_on_continuation_cols,
        nested_originated_columns,
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

            # Reconstruct ListArray for this field
            # Use mask=arrow_array.is_null() to preserve nulls from the original list
            new_list_array = pa.ListArray.from_arrays(
                offsets, flattened_fields[field_idx], mask=arrow_array.is_null()
            )

            new_cols_to_add[new_col_name] = pd.Series(
                new_list_array.to_pylist(),
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
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Explode array columns into new rows."""
    exploded_rows = []
    array_row_groups: dict[str, list[int]] = {}
    non_array_columns = dataframe.columns.drop(array_columns).tolist()
    non_array_df = dataframe[non_array_columns]

    for orig_idx in dataframe.index:
        non_array_data = non_array_df.loc[orig_idx].to_dict()
        array_values = {}
        max_len_in_row = 0
        non_na_array_found = False

        for col_name in array_columns:
            val = dataframe.loc[orig_idx, col_name]
            if val is not None and not (
                isinstance(val, list) and len(val) == 1 and pd.isna(val[0])
            ):
                array_values[col_name] = list(val)
                max_len_in_row = max(max_len_in_row, len(val))
                non_na_array_found = True
            else:
                array_values[col_name] = []

        if not non_na_array_found:
            new_row = non_array_data.copy()
            for col_name in array_columns:
                new_row[f"{col_name}"] = pd.NA
            exploded_rows.append(new_row)
            orig_key = str(orig_idx)
            if orig_key not in array_row_groups:
                array_row_groups[orig_key] = []
            array_row_groups[orig_key].append(len(exploded_rows) - 1)
            continue

        # Create one row per array element, up to max_len_in_row
        for array_idx in range(max_len_in_row):
            new_row = non_array_data.copy()

            # Add the specific array element for this index
            for col_name in array_columns:
                if array_idx < len(array_values.get(col_name, [])):
                    new_row[f"{col_name}"] = array_values[col_name][array_idx]
                else:
                    new_row[f"{col_name}"] = pd.NA

            exploded_rows.append(new_row)

            # Track which rows belong to which original row
            orig_key = str(orig_idx)
            if orig_key not in array_row_groups:
                array_row_groups[orig_key] = []
            array_row_groups[orig_key].append(len(exploded_rows) - 1)

    if exploded_rows:
        # Reconstruct the DataFrame to maintain original column order
        exploded_df = pd.DataFrame(exploded_rows)[dataframe.columns]
        for col in exploded_df.columns:
            # After explosion, object columns that are all-numeric (except for NAs)
            # should be converted to a numeric dtype for proper alignment.
            if exploded_df[col].dtype == "object":
                try:
                    # Use nullable integer type to preserve integers
                    exploded_df[col] = exploded_df[col].astype(pd.Int64Dtype())
                except (ValueError, TypeError):
                    # Fallback for non-integer numerics
                    try:
                        exploded_df[col] = pd.to_numeric(exploded_df[col])
                    except (ValueError, TypeError):
                        # Keep as object if not numeric
                        pass
        return exploded_df, array_row_groups
    else:
        return dataframe, array_row_groups


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
                flattened_fields[field_idx].to_pylist(),
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
