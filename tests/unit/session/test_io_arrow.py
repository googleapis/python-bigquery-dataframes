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
from bigframes.session._io.arrow import create_dataframe_from_arrow_table
from bigframes.testing import mocks


@pytest.fixture(scope="module")
def session():
    # Use the mock session from bigframes.testing
    return mocks.create_bigquery_session()


def test_create_dataframe_from_arrow_table(session):
    pa_table = pa.Table.from_pydict(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        }
    )
    # The mock session might need specific setup for to_pandas if it hits BQ client
    # For now, let's assume the LocalNode execution path for to_pandas is self-contained enough
    # or that the default mock bqclient.query_and_wait is sufficient.
    df = create_dataframe_from_arrow_table(pa_table, session=session)
    assert len(df.columns) == 2
    # The default mock session's query_and_wait returns an empty table by default for non-special queries.
    # We need to mock the specific table result for to_pandas to work as expected for LocalDataNode.
    # This is a bit of a deeper mock interaction.
    # A simpler unit test might avoid to_pandas() and check Block properties.
    # However, if create_bigquery_session's mocks are sufficient, this might pass.
    # For LocalDataNode, to_pandas eventually calls executor.execute(block.expr)
    # which for the default mock session might not return the actual data.
    # Let's adjust how the mock session's query_and_wait works for this specific table.

    # For local data, to_pandas will use the local data directly if possible,
    # avoiding BQ calls. Let's see if the current mocks are enough.
    pd_df = df.to_pandas()
    assert pd_df.shape == (3,2)
    assert list(df.columns) == ["col1", "col2"]
    assert pd_df["col1"].dtype == "int64"
    assert pd_df["col2"].dtype == "object"


def test_create_dataframe_from_arrow_table_empty(session):
    pa_table = pa.Table.from_pydict(
        {
            "col1": pa.array([], type=pa.int64()),
            "col2": pa.array([], type=pa.string()),
        }
    )
    df = create_dataframe_from_arrow_table(pa_table, session=session)
    assert len(df.columns) == 2
    pd_df = df.to_pandas()
    assert pd_df.shape == (0,2)
    assert list(df.columns) == ["col1", "col2"]
    assert pd_df["col1"].dtype == "int64"
    assert pd_df["col2"].dtype == "object"


def test_create_dataframe_from_arrow_table_all_types(session):
    pa_table = pa.Table.from_pydict(
        {
            "int_col": [1, None, 3],
            "float_col": [1.0, None, 3.0],
            "bool_col": [True, None, False],
            "string_col": ["a", None, "c"],
            "bytes_col": [b"a", None, b"c"],
            "date_col": pa.array([datetime.date(2023, 1, 1), None, datetime.date(2023, 1, 3)], type=pa.date32()),
            "timestamp_col": pa.array([datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc), None, datetime.datetime(2023, 1, 3, 12, 0, 0, tzinfo=datetime.timezone.utc)], type=pa.timestamp("us", tz="UTC")),
        }
    )
    df = create_dataframe_from_arrow_table(pa_table, session=session)
    assert len(df.columns) == 7

    # For LocalDataNode, head() should work locally.
    df_head = df.head(5)
    assert len(df_head) == 3

    pd_df = df.to_pandas()
    assert pd_df["int_col"].dtype == "Int64"
    assert pd_df["float_col"].dtype == "float64"
    assert pd_df["bool_col"].dtype == "boolean"
    assert pd_df["string_col"].dtype == "object"
    assert pd_df["bytes_col"].dtype == "object"
    assert pd_df["date_col"].dtype.name.startswith("date32")
    assert pd_df["timestamp_col"].dtype.name.startswith("timestamp[us, tz=UTC]")

@pytest.fixture(scope="module") # Changed to module as it's shared and doesn't change
def arrow_sample_data():
    return pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["foo", "bar", "baz", "qux", "quux"],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

def test_session_read_arrow(session, arrow_sample_data):
    session._set_arrow_table_for_shape(arrow_sample_data)
    df = session.read_arrow(arrow_sample_data)
    assert isinstance(df, bpd.DataFrame)
    # assert df.shape == (5, 3)
    assert list(df.columns) == ["id", "name", "value"]
    pd_df = df.to_pandas()
    assert pd_df["id"].tolist() == [1,2,3,4,5]
    assert pd_df["name"].tolist() == ["foo", "bar", "baz", "qux", "quux"]

def test_pandas_read_arrow(arrow_sample_data, mocker):
    # This test uses the global session, so we mock what get_global_session returns
    mock_session_instance = session() # Get an instance of our MockSession
    mock_session_instance._set_arrow_table_for_shape(arrow_sample_data)
    mocker.patch("bigframes.pandas.global_session.get_global_session", return_value=mock_session_instance)

    df = bpd.read_arrow(arrow_sample_data)
    assert isinstance(df, bpd.DataFrame)
    # assert df.shape == (5, 3)
    assert list(df.columns) == ["id", "name", "value"]
    pd_df = df.to_pandas()


def test_create_dataframe_from_arrow_table(session):
    pa_table = pa.Table.from_pydict(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        }
    )
    df = create_dataframe_from_arrow_table(pa_table, session=session)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["col1", "col2"]
    # Verify dtypes by converting to pandas - more direct checks might be needed depending on internal structure
    pd_df = df.to_pandas()
    assert pd_df["col1"].dtype == "int64"
    assert pd_df["col2"].dtype == "object"  # StringDtype might be more accurate depending on session config


def test_create_dataframe_from_arrow_table_empty(session):
    pa_table = pa.Table.from_pydict(
        {
            "col1": pa.array([], type=pa.int64()),
            "col2": pa.array([], type=pa.string()),
        }
    )
    df = create_dataframe_from_arrow_table(pa_table, session=session)
    assert df.shape == (0, 2)
    assert list(df.columns) == ["col1", "col2"]
    pd_df = df.to_pandas()
    assert pd_df["col1"].dtype == "int64"
    assert pd_df["col2"].dtype == "object"


def test_create_dataframe_from_arrow_table_all_types(session):
    pa_table = pa.Table.from_pydict(
        {
            "int_col": [1, None, 3],
            "float_col": [1.0, None, 3.0],
            "bool_col": [True, None, False],
            "string_col": ["a", None, "c"],
            "bytes_col": [b"a", None, b"c"],
            "date_col": pa.array([datetime.date(2023, 1, 1), None, datetime.date(2023, 1, 3)], type=pa.date32()),
            "timestamp_col": pa.array([datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc), None, datetime.datetime(2023, 1, 3, 12, 0, 0, tzinfo=datetime.timezone.utc)], type=pa.timestamp("us", tz="UTC")),
        }
    )
    df = create_dataframe_from_arrow_table(pa_table, session=session)
    assert df.shape == (3, 7)
    # This will execute the query and fetch results, good for basic validation
    df_head = df.head(5)
    assert len(df_head) == 3

    # More detailed dtype checks might require looking at the bq schema or ibis schema
    # For now, we rely on to_pandas() for a basic check
    pd_df = df.to_pandas()
    assert pd_df["int_col"].dtype == "Int64" # Pandas uses nullable Int64Dtype
    assert pd_df["float_col"].dtype == "float64"
    assert pd_df["bool_col"].dtype == "boolean" # Pandas uses nullable BooleanDtype
    assert pd_df["string_col"].dtype == "object" # Or StringDtype
    assert pd_df["bytes_col"].dtype == "object" # Or ArrowDtype(pa.binary())
    assert pd_df["date_col"].dtype.name.startswith("date32") # ArrowDtype(pa.date32())
    assert pd_df["timestamp_col"].dtype.name.startswith("timestamp[us, tz=UTC]") # ArrowDtype(pa.timestamp('us', tz='UTC'))

@pytest.fixture(scope="session")
def arrow_sample_data():
    return pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["foo", "bar", "baz", "qux", "quux"],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

def test_session_read_arrow(session, arrow_sample_data):
    df = session.read_arrow(arrow_sample_data)
    assert isinstance(df, bpd.DataFrame)
    assert df.shape == (5, 3)
    assert list(df.columns) == ["id", "name", "value"]
    pd_df = df.to_pandas()
    assert pd_df["id"].tolist() == [1,2,3,4,5]
    assert pd_df["name"].tolist() == ["foo", "bar", "baz", "qux", "quux"]

    def test_pandas_read_arrow(arrow_sample_data, mocker):
        # This test uses the global session, so we mock what get_global_session returns
        mock_session_instance = session() # Get an instance of our MockSession
        mocker.patch("bigframes.pandas.global_session.get_global_session", return_value=mock_session_instance)

    df = bpd.read_arrow(arrow_sample_data)
    assert isinstance(df, bpd.DataFrame)
    assert df.shape == (5, 3)
    assert list(df.columns) == ["id", "name", "value"]
    pd_df = df.to_pandas()
    assert pd_df["id"].tolist() == [1,2,3,4,5]
    assert pd_df["name"].tolist() == ["foo", "bar", "baz", "qux", "quux"]

def test_read_arrow_with_index_col_error(session, arrow_sample_data):
    # read_arrow doesn't support index_col, ensure it's not accidentally passed or used
    # For now, there's no index_col parameter. If added, this test would need adjustment.
    # This test is more of a placeholder to remember this aspect.
    # If create_dataframe_from_arrow_table were to accept index_col, this would be relevant.
    with pytest.raises(TypeError) as excinfo:
        session.read_arrow(arrow_sample_data, index_col="id") # type: ignore
    assert "got an unexpected keyword argument 'index_col'" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        bpd.read_arrow(arrow_sample_data, index_col="id") # type: ignore
    assert "got an unexpected keyword argument 'index_col'" in str(excinfo.value)
