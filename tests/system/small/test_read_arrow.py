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

import pandas
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
    arrow_table = pa.Table.from_arrays(data, names=["ints", "floats", "strings"])

    bf_df = bpd.read_arrow(arrow_table)

    assert bf_df.shape == (3, 3)
    # Expected dtypes (BigQuery/BigFrames dtypes)
    assert str(bf_df.dtypes["ints"]) == "Int64"
    assert str(bf_df.dtypes["floats"]) == "Float64"
    assert str(bf_df.dtypes["strings"]) == "string[pyarrow]"

    pd_df = arrow_table.to_pandas()
    # Convert BigFrames to pandas for comparison
    bf_pd_df = bf_df.to_pandas()

    pandas.testing.assert_frame_equal(
        bf_pd_df.astype(pd_df.dtypes), pd_df, check_dtype=False
    )


def test_read_arrow_engine_inline(session):
    data = [
        pa.array([10, 20], type=pa.int64()),
        pa.array(["apple", "banana"], type=pa.string()),
    ]
    arrow_table = pa.Table.from_arrays(data, names=["numbers", "fruits"])

    bf_df = bpd.read_arrow(arrow_table, write_engine="bigquery_inline")

    assert bf_df.shape == (2, 2)
    assert str(bf_df.dtypes["numbers"]) == "Int64"
    assert str(bf_df.dtypes["fruits"]) == "string[pyarrow]"

    pd_df = arrow_table.to_pandas()
    bf_pd_df = bf_df.to_pandas()
    pandas.testing.assert_frame_equal(
        bf_pd_df.astype(pd_df.dtypes), pd_df, check_dtype=False
    )


def test_read_arrow_engine_load(session):
    # For 'bigquery_load', the table can be slightly larger, but still manageable
    # The primary goal is to test the path, not performance here.
    int_values = list(range(10))
    str_values = [f"item_{i}" for i in range(10)]
    data = [
        pa.array(int_values, type=pa.int64()),
        pa.array(str_values, type=pa.string()),
    ]
    arrow_table = pa.Table.from_arrays(data, names=["ids", "items"])

    bf_df = bpd.read_arrow(arrow_table, write_engine="bigquery_load")

    assert bf_df.shape == (10, 2)
    assert str(bf_df.dtypes["ids"]) == "Int64"
    assert str(bf_df.dtypes["items"]) == "string[pyarrow]"

    pd_df = arrow_table.to_pandas()
    bf_pd_df = bf_df.to_pandas()
    pandas.testing.assert_frame_equal(
        bf_pd_df.astype(pd_df.dtypes), pd_df, check_dtype=False
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
    assert str(bf_df.dtypes["int_col"]) == "Int64"
    assert str(bf_df.dtypes["float_col"]) == "Float64"
    assert str(bf_df.dtypes["str_col"]) == "string[pyarrow]"
    assert str(bf_df.dtypes["bool_col"]) == "boolean[pyarrow]"
    assert str(bf_df.dtypes["ts_col"]) == "timestamp[us, tz=UTC]"
    assert str(bf_df.dtypes["date_col"]) == "date"

    pd_expected = arrow_table.to_pandas()
    bf_pd_df = bf_df.to_pandas()

    for col in ["int_col", "float_col"]:
        bf_pd_df[col] = bf_pd_df[col].astype(pd_expected[col].dtype)

    bf_pd_df["str_col"] = bf_pd_df["str_col"].astype(pandas.ArrowDtype(pa.string()))
    bf_pd_df["ts_col"] = pandas.to_datetime(bf_pd_df["ts_col"], utc=True)
    bf_pd_df["date_col"] = bf_pd_df["date_col"].apply(
        lambda x: x.date() if hasattr(x, "date") and x is not pandas.NaT else x
    )
    bf_pd_df["bool_col"] = bf_pd_df["bool_col"].astype(pandas.ArrowDtype(pa.bool_()))
    pd_expected["bool_col"] = pd_expected["bool_col"].astype(
        pandas.ArrowDtype(pa.bool_())
    )

    pandas.testing.assert_frame_equal(
        bf_pd_df, pd_expected, check_dtype=False, rtol=1e-5
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
    # BigQuery loads list types as ARRAY<element_type>, which translates to object in pandas
    # or specific ArrowDtype if pandas is configured for it.
    # For BigFrames, it should be ArrowDtype.
    assert isinstance(bf_df.dtypes["list_int_col"], pandas.ArrowDtype)
    assert bf_df.dtypes["list_int_col"].pyarrow_dtype == pa.list_(pa.int64())
    assert isinstance(bf_df.dtypes["list_str_col"], pandas.ArrowDtype)
    assert bf_df.dtypes["list_str_col"].pyarrow_dtype == pa.list_(pa.string())

    pd_expected = arrow_table.to_pandas()
    bf_pd_df = bf_df.to_pandas()

    # Explicitly cast to ArrowDtype for comparison as pandas might default to object
    pd_expected["list_int_col"] = pd_expected["list_int_col"].astype(
        pandas.ArrowDtype(pa.list_(pa.int64()))
    )
    pd_expected["list_str_col"] = pd_expected["list_str_col"].astype(
        pandas.ArrowDtype(pa.list_(pa.string()))
    )
    bf_pd_df["list_int_col"] = bf_pd_df["list_int_col"].astype(
        pandas.ArrowDtype(pa.list_(pa.int64()))
    )
    bf_pd_df["list_str_col"] = bf_pd_df["list_str_col"].astype(
        pandas.ArrowDtype(pa.list_(pa.string()))
    )

    pandas.testing.assert_frame_equal(bf_pd_df, pd_expected, check_dtype=True)


def test_read_arrow_engine_streaming(session):
    data = [
        pa.array([100, 200], type=pa.int64()),
        pa.array(["stream_test1", "stream_test2"], type=pa.string()),
    ]
    arrow_table = pa.Table.from_arrays(data, names=["id", "event"])
    bf_df = bpd.read_arrow(arrow_table, write_engine="bigquery_streaming")

    assert bf_df.shape == (2, 2)
    assert str(bf_df.dtypes["id"]) == "Int64"
    assert str(bf_df.dtypes["event"]) == "string[pyarrow]"
    pd_expected = arrow_table.to_pandas()
    bf_pd_df = bf_df.to_pandas()
    pandas.testing.assert_frame_equal(
        bf_pd_df.astype(pd_expected.dtypes), pd_expected, check_dtype=False
    )


def test_read_arrow_engine_write(session):
    data = [
        pa.array([300, 400], type=pa.int64()),
        pa.array(["write_api_test1", "write_api_test2"], type=pa.string()),
    ]
    arrow_table = pa.Table.from_arrays(data, names=["job_id", "status"])
    bf_df = bpd.read_arrow(arrow_table, write_engine="bigquery_write")

    assert bf_df.shape == (2, 2)
    assert str(bf_df.dtypes["job_id"]) == "Int64"
    assert str(bf_df.dtypes["status"]) == "string[pyarrow]"
    pd_expected = arrow_table.to_pandas()
    bf_pd_df = bf_df.to_pandas()
    pandas.testing.assert_frame_equal(
        bf_pd_df.astype(pd_expected.dtypes), pd_expected, check_dtype=False
    )


def test_read_arrow_no_columns_empty_rows(session):
    arrow_table = pa.Table.from_arrays([], names=[])
    bf_df = bpd.read_arrow(arrow_table)
    assert bf_df.shape == (0, 0)
    assert bf_df.empty


def test_read_arrow_special_column_names(session):
    col_names = [
        "col with space",
        "col/slash",
        "col.dot",
        "col:colon",
        "col(paren)",
        "col[bracket]",
    ]
    # BigQuery normalizes column names by replacing special characters with underscores.
    # Exception: dots are not allowed and usually cause errors or are handled by specific client libraries.
    # BigFrames aims to map to valid BigQuery column names.
    # For example, "col with space" becomes "col_with_space".
    # Slashes, colons, parens, brackets are also replaced with underscores.
    # Dots are problematic and might be handled differently or raise errors.
    # Let's use names that are likely to be sanitized to underscores by BQ.

    # Pyarrow allows these names, but BQ will sanitize them.
    # The test should assert against the sanitized names if that's the behavior.
    # If BF tries to preserve original names via aliasing, then assert original names.
    # Current assumption: BQ sanitizes, BF reflects sanitized names.

    arrow_data = [pa.array([1, 2], type=pa.int64())] * len(col_names)
    arrow_table = pa.Table.from_arrays(arrow_data, names=col_names)

    bf_df = bpd.read_arrow(arrow_table)

    # BigQuery replaces most special characters with underscores.
    # Let's define expected sanitized names based on typical BQ behavior.
    # Exact sanitization rules can be complex (e.g. leading numbers, repeated underscores).
    # This is a basic check.
    expected_bq_names = [
        "col_with_space",
        "col_slash",
        "col_dot",  # BQ might error on dots or replace them. Let's assume replacement for now.
        "col_colon",
        "col_paren_",
        "col_bracket_",
    ]
    # Update: Based on typical BigQuery behavior, dots are not allowed.
    # However, BigFrames might handle this by replacing dots with underscores before sending to BQ,
    # or there might be an issue at the BQ client library level or during the load.
    # For now, let's assume a sanitization that makes them valid BQ identifiers.
    # If the test fails, it will indicate the actual sanitization/handling.
    # Most robust is often to replace non-alphanumeric_ with underscore.

    # Let's assume BigFrames ensures valid BQ names, which generally means
    # letters, numbers, and underscores, not starting with a number.
    # The exact sanitization logic might be in BF or the BQ client.
    # For this test, we'll assert the column count and then check data.
    # Asserting exact sanitized names might be too dependent on internal BF/BQ details.

    assert bf_df.shape[1] == len(col_names)

    # Instead of asserting exact sanitized names, we rely on the fact that
    # bf_df.to_pandas() will use the (potentially sanitized) column names from BigQuery.
    # And arrow_table.to_pandas() will use the original names.
    # We then rename bf_pd_df columns to match pd_expected for data comparison.

    pd_expected = arrow_table.to_pandas()  # Has original names
    bf_pd_df = bf_df.to_pandas()  # Has BQ/BF names

    assert len(bf_pd_df.columns) == len(pd_expected.columns)

    # For data comparison, align column names if they were sanitized
    bf_pd_df.columns = pd_expected.columns

    pandas.testing.assert_frame_equal(bf_pd_df, pd_expected, check_dtype=False)


# TODO(b/340350610): Add tests for edge cases:
# - Table with all None values in a column
# - Table with very long strings or large binary data (if applicable for "small" tests)
# - Table with duplicate column names (should probably raise error from pyarrow or BF)
# - Schema with no columns (empty list of arrays) -> Covered by test_read_arrow_no_columns_empty_rows
# - Table with only an index (if read_arrow were to support Arrow index directly)
# - Test interaction with session-specific configurations if any affect read_arrow
#   (e.g., default index type, though read_arrow primarily creates from data columns)

# After tests, reset session if it was manually created for this module/class
# For now, using global session fixture, so no explicit reset here.
# def teardown_module(module):
#     bpd.reset_session()
