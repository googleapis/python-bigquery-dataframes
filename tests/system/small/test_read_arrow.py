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


class TestReadArrow:
    def test_read_arrow_basic(self, session):
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

        pd_df = arrow_table.to_pandas()
        # Convert BigFrames to pandas for comparison
        bf_pd_df = bf_df.to_pandas()

        pandas.testing.assert_frame_equal(
            bf_pd_df.astype(pd_df.dtypes), pd_df, check_dtype=False
        )

    def test_read_arrow_engine_inline(self, session):
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

    def test_read_arrow_engine_load(self, session):
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

    def test_read_arrow_all_types(self, session):
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
            # TODO(b/340350610): Enable list type once supported by all engines or add engine-specific tests
            # pa.array([[1, 2], None, [3, 4, 5]], type=pa.list_(pa.int64())),
        ]
        names = [
            "int_col",
            "float_col",
            "str_col",
            "bool_col",
            "ts_col",
            "date_col",
            # "list_col",
        ]
        arrow_table = pa.Table.from_arrays(data, names=names)

        bf_df = bpd.read_arrow(arrow_table)

        assert bf_df.shape == (3, len(names))
        assert str(bf_df.dtypes["int_col"]) == "Int64"
        assert str(bf_df.dtypes["float_col"]) == "Float64"
        assert str(bf_df.dtypes["str_col"]) == "string[pyarrow]"
        assert str(bf_df.dtypes["bool_col"]) == "boolean[pyarrow]" # TODO(b/340350610): should be boolean not boolean[pyarrow]
        assert str(bf_df.dtypes["ts_col"]) == "timestamp[us, tz=UTC]"
        assert str(bf_df.dtypes["date_col"]) == "date"
        # assert str(bf_df.dtypes["list_col"]) == "TODO" # Define expected BQ/BF dtype

        # Using to_pandas for data comparison, ensure dtypes are compatible.
        # BigQuery DataFrames might use ArrowDtype for some types by default.
        pd_expected = arrow_table.to_pandas()

        # Convert to pandas with specific dtype handling for comparison
        bf_pd_df = bf_df.to_pandas()

        # Pandas to_datetime might be needed for proper comparison of timestamp/date
        # Forcing types to be consistent for comparison
        for col in ["int_col", "float_col"]: # "bool_col"
             bf_pd_df[col] = bf_pd_df[col].astype(pd_expected[col].dtype)

        # String columns are compared as objects by default in pandas if there are NaNs
        # We expect string[pyarrow] from BigQuery DataFrames
        bf_pd_df["str_col"] = bf_pd_df["str_col"].astype(pandas.ArrowDtype(pa.string()))

        # Timestamps and dates need careful handling for comparison
        bf_pd_df["ts_col"] = pandas.to_datetime(bf_pd_df["ts_col"], utc=True)
        # pd_expected["ts_col"] is already correct due to pa.timestamp("us", tz="UTC")

        # Date comparison
        # bf_pd_df["date_col"] comes as dbdate, convert to datetime.date
        bf_pd_df["date_col"] = bf_pd_df["date_col"].apply(lambda x: x.date() if hasattr(x, 'date') else x)
        # pd_expected["date_col"] is already datetime.date objects

        # Bool comparison (pyarrow bools can be different from pandas bools with NAs)
        bf_pd_df["bool_col"] = bf_pd_df["bool_col"].astype(pandas.ArrowDtype(pa.bool_()))
        pd_expected["bool_col"] = pd_expected["bool_col"].astype(pandas.ArrowDtype(pa.bool_()))


        pandas.testing.assert_frame_equal(
            bf_pd_df, pd_expected, check_dtype=False, # check_dtype often problematic with Arrow mixed
            rtol=1e-5 # for float comparisons
        )


    def test_read_arrow_empty_table(self, session):
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

    # TODO(b/340350610): Add tests for write_engine="bigquery_streaming" and "bigquery_write"
    # once they are fully implemented and stable for pyarrow.Table inputs.
    # These might require specific setups or larger data to be meaningful.

    # TODO(b/340350610): Add tests for edge cases:
    # - Table with all None values in a column
    # - Table with very long strings or large binary data (if applicable for "small" tests)
    # - Table with duplicate column names (should probably raise error from pyarrow or BF)
    # - Table with unusual but valid column names (e.g., spaces, special chars)
    # - Schema with no columns (empty list of arrays)
    # - Table with only an index (if read_arrow were to support Arrow index directly)
    # - Test interaction with session-specific configurations if any affect read_arrow
    #   (e.g., default index type, though read_arrow primarily creates from data columns)

    # After tests, reset session if it was manually created for this module/class
    # For now, using global session fixture, so no explicit reset here.
    # def teardown_module(module):
    #     bpd.reset_session()
