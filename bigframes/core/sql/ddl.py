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

from typing import Any, Mapping, Optional, Union

from google.cloud import bigquery

import bigframes.core.compile.googlesql as googlesql
import bigframes.core.sql


def create_external_table_ddl(
    table_name: str,
    *,
    replace: bool = False,
    if_not_exists: bool = False,
    columns: Optional[Mapping[str, str]] = None,
    partition_columns: Optional[Mapping[str, str]] = None,
    connection_name: Optional[str] = None,
    options: Mapping[str, Union[str, int, float, bool, list]],
) -> str:
    """Generates the CREATE EXTERNAL TABLE DDL statement."""
    statement = ["CREATE"]
    if replace:
        statement.append("OR REPLACE")
    statement.append("EXTERNAL TABLE")
    if if_not_exists:
        statement.append("IF NOT EXISTS")
    statement.append(table_name)

    if columns:
        column_defs = ", ".join([f"{name} {typ}" for name, typ in columns.items()])
        statement.append(f"({column_defs})")

    if connection_name:
        statement.append(f"WITH CONNECTION `{connection_name}`")

    if partition_columns:
        part_defs = ", ".join(
            [f"{name} {typ}" for name, typ in partition_columns.items()]
        )
        statement.append(f"WITH PARTITION COLUMNS ({part_defs})")

    if options:
        opts = []
        for key, value in options.items():
            if isinstance(value, str):
                value_sql = repr(value)
                opts.append(f"{key} = {value_sql}")
            elif isinstance(value, bool):
                opts.append(f"{key} = {str(value).upper()}")
            elif isinstance(value, list):
                list_str = ", ".join([repr(v) for v in value])
                opts.append(f"{key} = [{list_str}]")
            else:
                opts.append(f"{key} = {value}")
        options_str = ", ".join(opts)
        statement.append(f"OPTIONS ({options_str})")

    return " ".join(statement)


def load_data_ddl(
    destination_table: str,
    uris: list[str],
    format: str,
    *,
    schema_fields: list[bigquery.SchemaField] | None = None,
    cluster_by: list[str] | None = None,
    partition_by: str | None = None,
    table_options: dict[str, Any] | None = None,
    load_options: dict[str, Any] | None = None,
    connection: str | None = None,
    hive_partition_columns: list[bigquery.SchemaField] | None = None,
    overwrite: bool = False,
) -> str:
    """Construct a LOAD DATA DDL statement."""
    action = "OVERWRITE" if overwrite else "INTO"

    query = f"LOAD DATA {action} {googlesql.identifier(destination_table)}\n"

    if schema_fields:
        columns_sql = ",\n".join(
            bigframes.core.sql.schema_field_to_sql(field) for field in schema_fields
        )
        query += f"(\n{columns_sql}\n)\n"

    if partition_by:
        query += f"PARTITION BY {partition_by}\n"

    if cluster_by:
        query += f"CLUSTER BY {', '.join(cluster_by)}\n"

    if table_options:
        opts_list = []
        for k, v in table_options.items():
            opts_list.append(f"{k}={bigframes.core.sql.simple_literal(v)}")
        query += f"OPTIONS({', '.join(opts_list)})\n"

    files_opts = {}
    if load_options:
        files_opts.update(load_options)

    files_opts["uris"] = uris
    files_opts["format"] = format

    files_opts_list = []
    for k, v in files_opts.items():
        files_opts_list.append(f"{k}={bigframes.core.sql.simple_literal(v)}")

    query += f"FROM FILES({', '.join(files_opts_list)})\n"

    if hive_partition_columns:
        cols_sql = ",\n".join(
            bigframes.core.sql.schema_field_to_sql(field)
            for field in hive_partition_columns
        )
        query += f"WITH PARTITION COLUMNS (\n{cols_sql}\n)\n"
    elif hive_partition_columns is not None:
        query += "WITH PARTITION COLUMNS\n"

    if connection:
        query += f"WITH CONNECTION {connection}\n"

    return query
