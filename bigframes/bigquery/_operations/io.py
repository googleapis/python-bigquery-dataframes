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

from __future__ import annotations

import typing
from typing import Any, List, Optional

import google.cloud.bigquery as bigquery

import bigframes.core.sql

if typing.TYPE_CHECKING:
    import bigframes.dataframe as dataframe
    import bigframes.session

_PLACEHOLDER_SCHEMA = [
    bigquery.SchemaField("bf_load_placeholder", "INT64"),
]


def load_data(
    uris: str | List[str],
    format: str,
    destination_table: Optional[str] = None,
    *,
    schema: Optional[List[bigquery.SchemaField]] = None,
    cluster_by: Optional[List[str]] = None,
    partition_by: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
    load_options: Optional[dict[str, Any]] = None,
    connection: Optional[str] = None,
    hive_partition_columns: Optional[List[bigquery.SchemaField]] = None,
    overwrite: bool = False,
    session: Optional[bigframes.session.Session] = None,
) -> dataframe.DataFrame:
    """
    Loads data from external files into a BigQuery table using the `LOAD DATA` statement.

    Args:
        uris (str | List[str]):
            The fully qualified URIs for the external data locations (e.g., 'gs://bucket/path/file.csv').
        format (str):
            The format of the external data (e.g., 'CSV', 'PARQUET', 'AVRO', 'JSON').
        destination_table (str, optional):
            The name of the destination table. If not specified, a temporary table will be created.
        schema (List[google.cloud.bigquery.SchemaField], optional):
            The schema of the destination table. If not provided, schema auto-detection will be used.
        cluster_by (List[str], optional):
            A list of columns to cluster the table by.
        partition_by (str, optional):
            The partition expression for the table.
        options (dict[str, Any], optional):
            Table options (e.g., {'description': 'my table'}).
        load_options (dict[str, Any], optional):
            Options for loading data (e.g., {'skip_leading_rows': 1}).
        connection (str, optional):
            The connection name to use for reading external data.
        hive_partition_columns (List[google.cloud.bigquery.SchemaField], optional):
            The external partitioning columns. If set to an empty list, partitioning is inferred.
        overwrite (bool, default False):
            If True, overwrites the destination table. If False, appends to it.
        session (bigframes.session.Session, optional):
            The session to use. If not provided, the default session is used.

    Returns:
        bigframes.dataframe.DataFrame: A DataFrame representing the loaded table.
    """
    import bigframes.pandas as bpd

    if session is None:
        session = bpd.get_global_session()

    if isinstance(uris, str):
        uris = [uris]

    if destination_table is None:
        # Create a temporary table name
        # We need to access the storage manager from the session
        # This is internal API usage, but requested by the user
        table_ref = session._storage_manager.create_temp_table(_PLACEHOLDER_SCHEMA)
        destination_table = f"{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}"
        # Since we created a placeholder table, we must overwrite it
        overwrite = True

    sql = bigframes.core.sql.load_data_ddl(
        destination_table=destination_table,
        uris=uris,
        format=format,
        schema_fields=schema,
        cluster_by=cluster_by,
        partition_by=partition_by,
        table_options=options,
        load_options=load_options,
        connection=connection,
        hive_partition_columns=hive_partition_columns,
        overwrite=overwrite,
    )

    # Execute the LOAD DATA statement
    session.read_gbq_query(sql)

    # Return a DataFrame pointing to the destination table
    # We use session.read_gbq to ensure it uses the same session
    return session.read_gbq(destination_table)
