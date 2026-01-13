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

import pandas as pd
import pyarrow as pa

from bigframes.display._flatten import flatten_nested_data


def test_flatten_nested_data_flattens_structs():
    """Verify that flatten_nested_data correctly flattens STRUCT columns."""
    struct_type = pa.struct([("name", pa.string()), ("age", pa.int64())])
    struct_arr = pa.array(
        [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}], type=struct_type
    )
    struct_data = pd.DataFrame(
        {
            "id": [1, 2],
            "struct_col": pd.Series(
                struct_arr,
                dtype=pd.ArrowDtype(struct_type),
            ),
        }
    )

    result = flatten_nested_data(struct_data)
    flattened = result.dataframe
    nested_originated_columns = result.nested_columns

    assert "struct_col.name" in flattened.columns
    assert "struct_col.age" in flattened.columns
    assert flattened["struct_col.name"].tolist() == ["Alice", "Bob"]
    assert "struct_col" in nested_originated_columns
    assert "struct_col.name" in nested_originated_columns
    assert "struct_col.age" in nested_originated_columns


def test_flatten_nested_data_explodes_arrays():
    """Verify that flatten_nested_data correctly explodes ARRAY columns."""
    array_type = pa.list_(pa.int64())
    array_arr = pa.array([[10, 20, 30], [40, 50]], type=array_type)
    array_data = pd.DataFrame(
        {
            "id": [1, 2],
            "array_col": pd.Series(array_arr, dtype=pd.ArrowDtype(array_type)),
        }
    )

    result = flatten_nested_data(array_data)
    flattened = result.dataframe
    row_labels = result.row_labels
    continuation_rows = result.continuation_rows
    nested_originated_columns = result.nested_columns

    assert len(flattened) == 5  # 3 + 2 array elements
    assert row_labels == ["0", "0", "0", "1", "1"]
    assert continuation_rows == {1, 2, 4}
    assert "array_col" in nested_originated_columns


def test_flatten_preserves_original_index():
    """Verify that original index is preserved (and duplicated) during flattening."""
    array_type = pa.list_(pa.int64())
    array_arr = pa.array([[10, 20], [30, 40]], type=array_type)
    array_data = pd.DataFrame(
        {
            "array_col": pd.Series(
                array_arr, dtype=pd.ArrowDtype(array_type), index=["row_a", "row_b"]
            ),
        }
    )
    array_data.index.name = "my_index"

    result = flatten_nested_data(array_data)
    flattened = result.dataframe
    row_labels = result.row_labels

    assert flattened.index.name == "my_index"
    assert flattened.index.tolist() == ["row_a", "row_a", "row_b", "row_b"]
    assert row_labels == ["row_a", "row_a", "row_b", "row_b"]


def test_flatten_preserves_multiindex():
    """Verify that MultiIndex is preserved (and duplicated) during flattening."""
    index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2)], names=["idx1", "idx2"])
    array_type = pa.list_(pa.int64())
    array_arr = pa.array([[10, 20], [30, 40]], type=array_type)
    array_data = pd.DataFrame(
        {
            "array_col": pd.Series(
                array_arr, dtype=pd.ArrowDtype(array_type), index=index
            ),
        }
    )

    result = flatten_nested_data(array_data)
    flattened = result.dataframe

    assert flattened.index.names == ["idx1", "idx2"]
    assert len(flattened) == 4
    assert flattened.index.tolist() == [("A", 1), ("A", 1), ("B", 2), ("B", 2)]


def test_flatten_empty_dataframe():
    """Verify behavior with an empty DataFrame."""
    empty_df = pd.DataFrame({"col": []})
    result = flatten_nested_data(empty_df)

    assert result.dataframe.empty
    assert result.dataframe.columns.tolist() == ["col"]
    assert result.row_labels is None
    assert result.continuation_rows is None


def test_flatten_mixed_struct_array():
    """Verify flattening of a DataFrame with both STRUCT and ARRAY columns."""
    struct_type = pa.struct([("a", pa.int64())])
    struct_arr = pa.array([{"a": 1}, {"a": 2}], type=struct_type)

    array_type = pa.list_(pa.int64())
    array_arr = pa.array([[10, 20], [30]], type=array_type)

    df = pd.DataFrame(
        {
            "struct_col": pd.Series(struct_arr, dtype=pd.ArrowDtype(struct_type)),
            "array_col": pd.Series(array_arr, dtype=pd.ArrowDtype(array_type)),
            "scalar_col": [100, 200],
        },
        index=[0, 1],
    )

    result = flatten_nested_data(df)
    flattened = result.dataframe
    continuation_rows = result.continuation_rows
    cleared_on_continuation = result.cleared_on_continuation

    # Row 0 explodes to 2 rows (array len 2). Row 1 stays 1 row (array len 1).
    # Total rows = 3.
    assert len(flattened) == 3

    # struct_col should be flattened to struct_col.a
    assert "struct_col.a" in flattened.columns
    assert flattened["struct_col.a"].tolist() == [1, 1, 2]

    # array_col should be exploded
    assert flattened["array_col"].tolist() == [10, 20, 30]

    # scalar_col should be duplicated
    assert flattened["scalar_col"].tolist() == [100, 100, 200]

    # Check metadata
    # continuation_rows should only contain index 1 (the second element of the first row's array)
    assert continuation_rows == {1}

    # struct_col.a and scalar_col should be in cleared_on_continuation
    assert "struct_col.a" in cleared_on_continuation
    assert "scalar_col" in cleared_on_continuation
