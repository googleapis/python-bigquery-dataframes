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

import datetime

import pandas as pd
import pyarrow as pa
import pytest

import bigframes.pandas as bpd


@pytest.fixture(scope="module")
def session():
    # Using a module-scoped session to avoid repeated setup/teardown for each test
    # This assumes tests are not modifying global session state in a conflicting way
    return bpd.get_global_session()


def test_read_arrow_basic(session):
    data = [
        pa.array([1, 2, 3], type=pa.int64()),
        pa.array([0.1, 0.2, 0.3], type=pa.float64()),
        pa.array(["foo", "bar", "baz"], type=pa.string()),
    ]
    arrow_table = pa.Table.from_arrays(
        data, names=["ints", "floats", "strings"]
    )

    bf_df = bpd.read_arrow(arrow_table)

    assert bf_df.shape == (3, 3)
    # Expected dtypes (BigQuery/BigFrames dtypes)
    assert str(bf_df.dtypes["ints"]) == "Int64"
    assert str(bf_df.dtypes["floats"]) == "Float64"
    assert str(bf_df.dtypes["strings"]) == "string[pyarrow]"

    # For deferred loading, the comparison should be against a pandas DataFrame
    # created with ArrowDtype for consistency.
    expected_pd_df = arrow_table.to_pandas(types_mapper=pd.ArrowDtype)

    bf_pd_df = bf_df.to_pandas()

    # Ensure dtypes are consistent for comparison, especially for string which might differ
    bf_pd_df["strings"] = bf_pd_df["strings"].astype(pd.ArrowDtype(pa.string()))

    pd.testing.assert_frame_equal(
        bf_pd_df, expected_pd_df, check_dtype=True
    )


def test_read_arrow_all_types(session):
    data = [
        pa.array([1, None, 3], type=pa.int64()),
        pa.array([0.1, None, 0.3], type=pa.float64()),
        pa.array(["foo", "bar", None], type=pa.string()),
        pa.array([True, False, True], type=pa.bool_()),
        pa.array(
            [
                datetime.datetime(2023, 1, 1, 12, 30, 0, tzinfo=datetime.timezone.utc),
                None,
                datetime.datetime(2023, 1, 2, 10, 0, 0, tzinfo=datetime.timezone.utc),
            ],
            type=pa.timestamp("us", tz="UTC"),
        ),
        pa.array(
            [datetime.date(2023, 1, 1), None, datetime.date(2023, 1, 3)],
            type=pa.date32(),
        ),
    ]
    names = [
        "int_col",
        "float_col",
        "str_col",
        "bool_col",
        "ts_col",
        "date_col",
    ]
    arrow_table = pa.Table.from_arrays(data, names=names)

    bf_df = bpd.read_arrow(arrow_table)

    assert bf_df.shape == (3, len(names))
    assert str(bf_df.dtypes["int_col"]) == "Int64"  # Uses pandas nullable Int64
    assert str(bf_df.dtypes["float_col"]) == "Float64" # Uses pandas nullable Float64
    assert str(bf_df.dtypes["str_col"]) == "string[pyarrow]"
    assert str(bf_df.dtypes["bool_col"]) == "boolean[pyarrow]"
    assert str(bf_df.dtypes["ts_col"]) == "timestamp[us, tz=UTC]"
    assert str(bf_df.dtypes["date_col"]) == "date" # Translates to dbdate in BigQuery pandas

    expected_pd_df = arrow_table.to_pandas(types_mapper=pd.ArrowDtype)
    bf_pd_df = bf_df.to_pandas() # This will also use ArrowDtypes where applicable

    # Date column from BQ might be dbdate, convert expected to match for direct comparison if necessary
    # However, if bf_df.to_pandas() also yields ArrowDtype for dates, direct comparison is fine.
    # Let's assume bf_pd_df["date_col"] is already ArrowDtype(pa.date32())
    # or compatible for direct comparison after `to_pandas(types_mapper=pd.ArrowDtype)`

    pd.testing.assert_frame_equal(
        bf_pd_df, expected_pd_df, check_dtype=True, rtol=1e-5
    )


def test_read_arrow_empty_table(session):
    data = [
        pa.array([], type=pa.int64()),
        pa.array([], type=pa.string()),
    ]
    arrow_table = pa.Table.from_arrays(data, names=["empty_int", "empty_str"])

    bf_df = bpd.read_arrow(arrow_table)

    assert bf_df.shape == (0, 2)
    assert str(bf_df.dtypes["empty_int"]) == "Int64"
    assert str(bf_df.dtypes["empty_str"]) == "string[pyarrow]"
    assert bf_df.empty


def test_read_arrow_list_types(session):
    data = [
        pa.array([[1, 2], None, [3, 4, 5], []], type=pa.list_(pa.int64())),
        pa.array([["a", "b"], ["c"], None, []], type=pa.list_(pa.string())),
    ]
    names = ["list_int_col", "list_str_col"]
    arrow_table = pa.Table.from_arrays(data, names=names)

    bf_df = bpd.read_arrow(arrow_table)

    assert bf_df.shape == (4, 2)
    assert isinstance(bf_df.dtypes["list_int_col"], pd.ArrowDtype)
    assert bf_df.dtypes["list_int_col"].pyarrow_dtype == pa.list_(pa.int64())
    assert isinstance(bf_df.dtypes["list_str_col"], pd.ArrowDtype)
    assert bf_df.dtypes["list_str_col"].pyarrow_dtype == pa.list_(pa.string())

    expected_pd_df = arrow_table.to_pandas(types_mapper=pd.ArrowDtype)
    bf_pd_df = bf_df.to_pandas() # Should also use ArrowDtypes

    pd.testing.assert_frame_equal(bf_pd_df, expected_pd_df, check_dtype=True)


def test_read_arrow_no_columns_empty_rows(session):
    arrow_table = pa.Table.from_arrays([], names=[])
    bf_df = bpd.read_arrow(arrow_table)
    assert bf_df.shape == (0, 0)
    assert bf_df.empty


def test_read_arrow_special_column_names(session):
    # Using names that are valid in Arrow but might be sanitized by BigQuery or BigFrames
    # BigFrames should handle mapping these to valid BigQuery column names,
    # and then map them back to original names when converting to pandas.
    col_names = ["col with space", "col/slash", "col.dot", "col:colon", "col(paren)", "col[bracket]"]

    arrow_data = [pa.array([1, 2], type=pa.int64())] * len(col_names)
    arrow_table = pa.Table.from_arrays(arrow_data, names=col_names)

    bf_df = bpd.read_arrow(arrow_table)

    assert bf_df.shape[1] == len(col_names)

    # The column names in bf_df should match the original Arrow table column names
    # as BigFrames aims to preserve original column labels where possible.
    pd.testing.assert_index_equal(bf_df.columns, pd.Index(col_names))

    expected_pd_df = arrow_table.to_pandas(types_mapper=pd.ArrowDtype)
    bf_pd_df = bf_df.to_pandas() # This should also have original column names

    pd.testing.assert_frame_equal(bf_pd_df, expected_pd_df, check_dtype=True)


# TODO(b/340350610): Add tests for edge cases:
# - Table with all None values in a column
# - Table with very long strings or large binary data
# - Table with duplicate column names (Arrow allows this, BigFrames should handle, possibly by raising error or renaming)
# - Test interaction with session-specific configurations if any affect read_arrow
#   (e.g., default index type, though read_arrow primarily creates from data columns)
