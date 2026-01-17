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

import pytest
from unittest.mock import MagicMock

import google.cloud.bigquery as bigquery
import bigframes.bigquery as bbq
import bigframes.session

@pytest.fixture
def mock_session():
    session = MagicMock(spec=bigframes.session.Session)
    session._storage_manager = MagicMock()
    return session

def test_load_data_minimal(mock_session):
    # Setup
    uris = ["gs://my-bucket/file.csv"]
    format = "CSV"
    destination_table = "my_project.my_dataset.my_table"

    # Execution
    bbq.load_data(uris, format, destination_table, session=mock_session)

    # Verification
    mock_session.read_gbq_query.assert_called_once()
    sql = mock_session.read_gbq_query.call_args[0][0]
    assert "LOAD DATA INTO `my_project.my_dataset.my_table`" in sql
    assert "FROM FILES" in sql
    assert "format='CSV'" in sql
    assert "uris=['gs://my-bucket/file.csv']" in sql

    mock_session.read_gbq.assert_called_once_with(destination_table)

def test_load_data_single_uri(mock_session):
    # Setup
    uris = "gs://my-bucket/file.csv"
    format = "CSV"
    destination_table = "t"

    # Execution
    bbq.load_data(uris, format, destination_table, session=mock_session)

    # Verification
    sql = mock_session.read_gbq_query.call_args[0][0]
    assert "uris=['gs://my-bucket/file.csv']" in sql

def test_load_data_temp_table(mock_session):
    # Setup
    uris = "gs://my-bucket/file.csv"
    format = "CSV"

    # Mock return of create_temp_table
    mock_session._storage_manager.create_temp_table.return_value = bigquery.TableReference.from_string("p.d.t")

    # Execution
    bbq.load_data(uris, format, session=mock_session)

    # Verification
    mock_session._storage_manager.create_temp_table.assert_called_once()

    mock_session.read_gbq_query.assert_called_once()
    sql = mock_session.read_gbq_query.call_args[0][0]
    # Should use OVERWRITE for temp table we just created
    assert "LOAD DATA OVERWRITE `p.d.t`" in sql

    mock_session.read_gbq.assert_called_once_with("p.d.t")

def test_load_data_all_options(mock_session):
    # Setup
    uris = ["gs://file.parquet"]
    format = "PARQUET"
    destination_table = "dest"
    schema = [
        bigquery.SchemaField("col1", "INT64", mode="REQUIRED", description="my col"),
        bigquery.SchemaField("col2", "STRING")
    ]
    cluster_by = ["col1"]
    partition_by = "col1"
    options = {"description": "desc"}
    load_options = {"ignore_unknown_values": True}
    connection = "my_conn"
    hive_partition_columns = [bigquery.SchemaField("pcol", "STRING")]
    overwrite = True

    # Execution
    bbq.load_data(
        uris, format, destination_table,
        schema=schema,
        cluster_by=cluster_by,
        partition_by=partition_by,
        options=options,
        load_options=load_options,
        connection=connection,
        hive_partition_columns=hive_partition_columns,
        overwrite=overwrite,
        session=mock_session
    )

    # Verification
    sql = mock_session.read_gbq_query.call_args[0][0]
    # Normalize newlines for easier assertion or check parts
    assert "LOAD DATA OVERWRITE `dest`" in sql
    assert "col1 INT64 NOT NULL OPTIONS(description='my col')" in sql
    assert "col2 STRING" in sql
    assert "PARTITION BY col1" in sql
    assert "CLUSTER BY col1" in sql
    assert "OPTIONS(description='desc')" in sql
    assert "FROM FILES" in sql
    assert "ignore_unknown_values=True" in sql
    assert "WITH PARTITION COLUMNS" in sql
    assert "pcol STRING" in sql
    assert "WITH CONNECTION my_conn" in sql

def test_load_data_hive_partition_inference(mock_session):
    # Setup
    uris = ["gs://file.parquet"]
    format = "PARQUET"
    destination_table = "dest"

    # Execution
    bbq.load_data(
        uris, format, destination_table,
        hive_partition_columns=[], # Empty list -> Inference
        session=mock_session
    )

    # Verification
    sql = mock_session.read_gbq_query.call_args[0][0]
    assert "WITH PARTITION COLUMNS" in sql
    assert "WITH PARTITION COLUMNS (" not in sql

def test_nested_schema_generation(mock_session):
    # Setup
    uris = "gs://file.json"
    format = "JSON"
    destination_table = "dest"
    schema = [
        bigquery.SchemaField("nested", "STRUCT", fields=[
            bigquery.SchemaField("sub", "INT64")
        ]),
        bigquery.SchemaField("arr", "INT64", mode="REPEATED")
    ]

    # Execution
    bbq.load_data(uris, format, destination_table, schema=schema, session=mock_session)

    # Verification
    sql = mock_session.read_gbq_query.call_args[0][0]
    assert "nested STRUCT<sub INT64>" in sql
    assert "arr ARRAY<INT64>" in sql
