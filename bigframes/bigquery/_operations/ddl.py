# Copyright 2026 Google LLC
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

from typing import Any, Mapping, Optional, Sequence, Union

import bigframes_vendored.constants
import google.cloud.bigquery
import pandas as pd

import bigframes.core.logging.log_adapter as log_adapter
import bigframes.core.sql.ddl
import bigframes.session


def _get_table_metadata(
    *,
    bqclient: google.cloud.bigquery.Client,
    table_name: str,
) -> pd.Series:
    table_metadata = bqclient.get_table(table_name)
    table_dict = table_metadata.to_api_repr()
    return pd.Series(table_dict)


@log_adapter.method_logger(custom_base_name="bigquery_table")
def create_external_table(
    table_name: str,
    *,
    replace: bool = False,
    if_not_exists: bool = False,
    columns: Optional[Mapping[str, str]] = None,
    partition_columns: Optional[Mapping[str, str]] = None,
    connection_name: Optional[str] = None,
    options: Mapping[str, Union[str, int, float, bool, list]],
    session: Optional[bigframes.session.Session] = None,
) -> pd.Series:
    """
    Creates a BigQuery external table.

    See the `BigQuery CREATE EXTERNAL TABLE DDL syntax
    <https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_external_table_statement>`_
    for additional reference.

    Args:
        table_name (str):
            The name of the table in BigQuery.
        replace (bool, default False):
            Whether to replace the table if it already exists.
        if_not_exists (bool, default False):
            Whether to ignore the error if the table already exists.
        columns (Mapping[str, str], optional):
            The table's schema.
        partition_columns (Mapping[str, str], optional):
            The table's partition columns.
        connection_name (str, optional):
            The connection to use for the table.
        options (Mapping[str, Union[str, int, float, bool, list]]):
            The OPTIONS clause, which specifies the table options.
        session (bigframes.session.Session, optional):
            The session to use. If not provided, the default session is used.

    Returns:
        pandas.Series:
            A Series with object dtype containing the table metadata. Reference
            the `BigQuery Table REST API reference
            <https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table>`_
            for available fields.
    """
    import bigframes.pandas as bpd

    sql = bigframes.core.sql.ddl.create_external_table_ddl(
        table_name=table_name,
        replace=replace,
        if_not_exists=if_not_exists,
        columns=columns,
        partition_columns=partition_columns,
        connection_name=connection_name,
        options=options,
    )

    if session is None:
        bpd.read_gbq_query(sql)
        session = bpd.get_global_session()
        assert (
            session is not None
        ), f"Missing connection to BigQuery. Please report how you encountered this error at {bigframes_vendored.constants.FEEDBACK_LINK}."
    else:
        session.read_gbq_query(sql)

    return _get_table_metadata(bqclient=session.bqclient, table_name=table_name)


@log_adapter.method_logger(custom_base_name="bigquery_table")
def load_data(
    uris: str | Sequence[str],
    format: str,
    destination_table: str,
    *,
    schema: Optional[Mapping[str, str]] = None,
    cluster_by: Optional[Sequence[str]] = None,
    partition_by: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
    load_options: Optional[dict[str, Any]] = None,
    connection: Optional[str] = None,
    hive_partition_columns: Optional[Mapping[str, str]] = None,
    overwrite: bool = False,
    session: Optional[bigframes.session.Session] = None,
) -> pd.Series:
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
        pandas.Series:
            A Series with object dtype containing the table metadata. Reference
            the `BigQuery Table REST API reference
            <https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#Table>`_
            for available fields.
    """
    import bigframes.pandas as bpd

    if session is None:
        session = bpd.get_global_session()

    if isinstance(uris, str):
        uris = [uris]

    sql = bigframes.core.sql.ddl.load_data_ddl(
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
