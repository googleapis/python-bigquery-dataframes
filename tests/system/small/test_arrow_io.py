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

import datetime

import pyarrow as pa
import pytest

import bigframes.pandas as bpd


@pytest.fixture(scope="session")
def arrow_all_types_table():
    # Using a dictionary to create the PyArrow table
    data_dict = {
        "int64_col": pa.array([1, 2, None, 4], type=pa.int64()),
        "float64_col": pa.array([0.1, 0.2, None, 0.4], type=pa.float64()),
        "bool_col": pa.array([True, False, None, True], type=pa.bool_()),
        "string_col": pa.array(["apple", "banana", None, "cherry"], type=pa.string()),
        "bytes_col": pa.array([b"one", b"two", None, b"four"], type=pa.large_binary()), # Using large_binary for BQ compatibility
        "date_col": pa.array(
            [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), None, datetime.date(2023, 1, 4)],
            type=pa.date32(),
        ),
        "timestamp_s_col": pa.array(
            [
                datetime.datetime.fromisoformat(ts) if ts else None
                for ts in ["2023-01-01T00:00:00", "2023-01-02T12:34:56", None, "2023-01-04T23:59:59"]
            ],
            type=pa.timestamp("s", tz="UTC"),
        ),
        "timestamp_ms_col": pa.array(
            [
                datetime.datetime.fromisoformat(ts) if ts else None
                for ts in ["2023-01-01T00:00:00.123", "2023-01-02T12:34:56.456", None, "2023-01-04T23:59:59.789"]
            ],
            type=pa.timestamp("ms", tz="UTC"),
        ),
        "timestamp_us_col": pa.array(
            [
                datetime.datetime.fromisoformat(ts) if ts else None
                for ts in ["2023-01-01T00:00:00.123456", "2023-01-02T12:34:56.789012", None, "2023-01-04T23:59:59.345678"]
            ],
            type=pa.timestamp("us", tz="UTC"),
        ),
        # BigQuery doesn't directly support nanosecond precision timestamps, they are typically truncated or rounded to microsecond.
        # "timestamp_ns_col": pa.array(
        #     [
        #         datetime.datetime.fromisoformat(ts) if ts else None
        #         for ts in ["2023-01-01T00:00:00.123456789", "2023-01-02T12:34:56.890123456", None, "2023-01-04T23:59:59.456789012"]
        #     ],
        #     type=pa.timestamp("ns", tz="UTC"),
        # ),
        # TODO: Add more complex types if supported by the conversion logic
        # "list_col": pa.array([[1, 2], None, [3, 4]], type=pa.list_(pa.int64())),
        # "struct_col": pa.array([{"a": 1, "b": "x"}, None, {"a": 2, "b": "y"}], type=pa.struct([("a", pa.int64()), ("b", pa.string())])),
    }
    return pa.Table.from_pydict(data_dict)


def test_read_arrow_all_types_global_session(arrow_all_types_table):
    # System test using the global session (or default session if set)
    # bpd.options.bigquery.project = "your-gcp-project"  # Ensure project is set if not using global default
    # bpd.options.bigquery.location = "your-location"    # Ensure location is set

    df = bpd.read_arrow(arrow_all_types_table)

    assert isinstance(df, bpd.DataFrame)
    # Number of columns might change if more types are added
    assert df.shape == (4, 9)
    assert list(df.columns) == [
        "int64_col",
        "float64_col",
        "bool_col",
        "string_col",
        "bytes_col",
        "date_col",
        "timestamp_s_col",
        "timestamp_ms_col",
        "timestamp_us_col",
        # "timestamp_ns_col",
    ]

    # Fetching the data to pandas to verify
    pd_df = df.to_pandas()
    assert pd_df.shape == (4, 9)

    # Basic check of data integrity (first non-null value)
    assert pd_df["int64_col"][0] == 1
    assert pd_df["float64_col"][0] == 0.1
    assert pd_df["bool_col"][0] is True
    assert pd_df["string_col"][0] == "apple"
    assert pd_df["bytes_col"][0] == b"one"
    assert str(pd_df["date_col"][0]) == "2023-01-01"  # Pandas converts to its own date/datetime objects
    assert str(pd_df["timestamp_s_col"][0]) == "2023-01-01 00:00:00+00:00"
    assert str(pd_df["timestamp_ms_col"][0]) == "2023-01-01 00:00:00.123000+00:00"
    assert str(pd_df["timestamp_us_col"][0]) == "2023-01-01 00:00:00.123456+00:00"
    # assert str(pd_df["timestamp_ns_col"][0]) == "2023-01-01 00:00:00.123456789+00:00" # BQ truncates to us

    # Check for None/NaT where PyArrow table had None
    assert pd_df["int64_col"][2] is pandas.NA
    assert pd_df["float64_col"][2] is pandas.NA
    assert pd_df["bool_col"][2] is pandas.NA
    assert pd_df["string_col"][2] is None  # Pandas uses None for object types (like string)
    assert pd_df["bytes_col"][2] is None
    assert pd_df["date_col"][2] is pandas.NaT
    assert pd_df["timestamp_s_col"][2] is pandas.NaT
    assert pd_df["timestamp_ms_col"][2] is pandas.NaT
    assert pd_df["timestamp_us_col"][2] is pandas.NaT
    # assert pd_df["timestamp_ns_col"][2] is pandas.NaT


def test_read_arrow_empty_table_global_session(session):
    empty_table = pa.Table.from_pydict({
        "col_a": pa.array([], type=pa.int64()),
        "col_b": pa.array([], type=pa.string())
    })
    df = bpd.read_arrow(empty_table)
    assert isinstance(df, bpd.DataFrame)
    assert df.shape == (0, 2)
    assert list(df.columns) == ["col_a", "col_b"]
    pd_df = df.to_pandas()
    assert pd_df.empty
    assert list(pd_df.columns) == ["col_a", "col_b"]
    assert pd_df["col_a"].dtype == "int64" # Or Int64 if there were NAs
    assert pd_df["col_b"].dtype == "object"


def test_read_arrow_specific_session(session, arrow_all_types_table):
    # Create a new session to ensure it's used
    # In a real test setup, you might create a session with specific configurations
    specific_session = bpd.get_global_session() # or bpd.Session(...)
    df = specific_session.read_arrow(arrow_all_types_table)

    assert isinstance(df, bpd.DataFrame)
    assert df.shape == (4, 9)
    pd_df = df.to_pandas() # Forces execution
    assert pd_df.shape == (4, 9)
    assert pd_df["int64_col"][0] == 1 # Basic data check

    # Ensure the dataframe is associated with the correct session if possible to check
    # This might involve checking some internal property or behavior linked to the session
    # For example, if temporary tables are created, they should use this session's context.
    # For local data, this is harder to verify directly without inspecting internals.
    assert df.session == specific_session

@pytest.mark.parametrize(
    "data,arrow_type,expected_bq_type_kind",
    [
        ([1, 2], pa.int8(), "INT64"),
        ([1, 2], pa.int16(), "INT64"),
        ([1, 2], pa.int32(), "INT64"),
        ([1, 2], pa.int64(), "INT64"),
        ([1.0, 2.0], pa.float16(), "FLOAT64"), # BQ promotes half to float
        ([1.0, 2.0], pa.float32(), "FLOAT64"),
        ([1.0, 2.0], pa.float64(), "FLOAT64"),
        ([True, False], pa.bool_(), "BOOL"),
        (["a", "b"], pa.string(), "STRING"),
        (["a", "b"], pa.large_string(), "STRING"),
        ([b"a", b"b"], pa.binary(), "BYTES"),
        ([b"a", b"b"], pa.large_binary(), "BYTES"),
        # Duration types are tricky, BQ doesn't have a direct equivalent, often converted to INT64 or STRING
        # ([pa.scalar(1000, type=pa.duration('s')), pa.scalar(2000, type=pa.duration('s'))], pa.duration('s'), "INT64"), # Or error
        ([datetime.date(2023,1,1)], pa.date32(), "DATE"),
        ([datetime.date(2023,1,1)], pa.date64(), "DATE"), # BQ promotes date64 to date
        ([datetime.datetime(2023,1,1,12,0,0, tzinfo=datetime.timezone.utc)], pa.timestamp("s", tz="UTC"), "TIMESTAMP"),
        ([datetime.datetime(2023,1,1,12,0,0, tzinfo=datetime.timezone.utc)], pa.timestamp("ms", tz="UTC"), "TIMESTAMP"),
        ([datetime.datetime(2023,1,1,12,0,0, tzinfo=datetime.timezone.utc)], pa.timestamp("us", tz="UTC"), "TIMESTAMP"),
        # Time types also tricky, BQ has TIME type
        # ([datetime.time(12,34,56)], pa.time32("s"), "TIME"),
        # ([datetime.time(12,34,56,789000)], pa.time64("us"), "TIME"),
    ],
)
def test_read_arrow_type_mappings(session, data, arrow_type, expected_bq_type_kind):
    """
    Tests that various arrow types are mapped to the expected BigQuery types.
    This is an indirect check via the resulting DataFrame's schema.
    """
    pa_table = pa.Table.from_arrays([pa.array(data, type=arrow_type)], names=["col"])
    df = session.read_arrow(pa_table)

    # Access the underlying ibis schema to check BQ types
    # This is an internal detail, but useful for this kind of test.
    # Adjust if the way to get ibis schema changes.
    ibis_schema = df._block.expr.schema
    assert ibis_schema["col"].name.upper() == expected_bq_type_kind

    # Also check pandas dtype after conversion for good measure
    pd_df = df.to_pandas()
    # The pandas dtype can be more complex (e.g., ArrowDtype, nullable dtypes)
    # but this ensures it's processed.
    assert pd_df["col"].shape == (len(data),)

# TODO(developer): Add tests for more complex arrow types like lists, structs, maps
# if and when they are supported by create_dataframe_from_arrow_table.
# For example:
# def test_read_arrow_list_type(session):
#     pa_table = pa.Table.from_arrays([pa.array([[1,2], [3,4,5]], type=pa.list_(pa.int64()))], names=['list_col'])
#     df = session.read_arrow(pa_table)
#     ibis_schema = df._block.expr.schema
#     assert ibis_schema["list_col"].is_array()
#     assert ibis_schema["list_col"].value_type.is_integer()
#     # Further checks on data

# def test_read_arrow_struct_type(session):
#     struct_type = pa.struct([("a", pa.int64()), ("b", pa.string())])
#     pa_table = pa.Table.from_arrays([pa.array([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], type=struct_type)], names=['struct_col'])
#     df = session.read_arrow(pa_table)
#     ibis_schema = df._block.expr.schema
#     assert ibis_schema["struct_col"].is_struct()
#     assert ibis_schema["struct_col"].fields["a"].is_integer()
#     assert ibis_schema["struct_col"].fields["b"].is_string()
#     # Further checks on dataTool output for `create_file_with_block`:
