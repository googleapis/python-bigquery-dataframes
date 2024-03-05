# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load test for query (SQL) inputs with large results sizes."""

import pytest

import bigframes.pandas as bpd

KB_BYTES = 1000
MB_BYTES = 1000 * KB_BYTES
GB_BYTES = 1000 * MB_BYTES
TB_BYTES = 1000 * GB_BYTES


@pytest.mark.parametrize(
    ("sql", "expected_bytes"),
    (
        pytest.param(
            "SELECT * FROM load_testing.scalars_1gb",
            GB_BYTES,
            id="1gb",
        ),
        pytest.param(
            "SELECT * FROM load_testing.scalars_10gb",
            10 * GB_BYTES,
            id="10gb",
        ),
        pytest.param(
            "SELECT * FROM load_testing.scalars_100gb",
            100 * GB_BYTES,
            id="100gb",
        ),
        pytest.param(
            "SELECT * FROM load_testing.scalars_1tb",
            TB_BYTES,
            id="1tb",
        ),
    ),
)
def test_read_gbq_sql_large_results(sql, expected_bytes):
    df = bpd.read_gbq(sql)
    assert df.memory_usage().sum() >= expected_bytes


def test_df_repr_large_table():
    df = bpd.read_gbq("load_testing.scalars_100gb")
    row_count, column_count = df.shape
    expected = f"[{row_count} rows x {column_count} columns]"
    actual = repr(df)
    assert expected in actual


def test_series_repr_large_table():
    df = bpd.read_gbq("load_testing.scalars_1tb")
    actual = repr(df["string_col"])
    assert actual is not None


def test_index_repr_large_table():
    df = bpd.read_gbq("load_testing.scalars_1tb")
    actual = repr(df.index)
    assert actual is not None


# FAILED
# tests/system/load/test_large_tables.py::test_to_pandas_batches_large_table
# google.api_core.exceptions.Forbidden: 403 Response too large to return.
# Consider specifying a destination table in your job...
@pytest.mark.xfail
def test_to_pandas_batches_large_table():
    df = bpd.read_gbq("load_testing.scalars_100gb")
    expected_row_count, expected_column_count = df.shape

    row_count = 0
    for df in df.to_pandas_batches():
        batch_row_count, batch_column_count = df.shape
        assert batch_column_count == expected_column_count
        row_count += batch_row_count

        # Attempt to save on memory by manually removing the batch df
        # from local memory after finishing with processing.
        del df

    assert row_count == expected_row_count
