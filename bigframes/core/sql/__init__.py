# Copyright 2023 Google LLC
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

"""
Utility functions for SQL construction.
"""

import datetime
import decimal
import json
import math
from typing import Any, cast, Collection, Iterable, Mapping, Optional, TYPE_CHECKING, Union

import shapely.geometry.base  # type: ignore

import bigframes.core.compile.googlesql as googlesql

if TYPE_CHECKING:
    import google.cloud.bigquery as bigquery

    import bigframes.core.ordering


# shapely.wkt.dumps was moved to shapely.io.to_wkt in 2.0.
try:
    from shapely.io import to_wkt  # type: ignore
except ImportError:
    from shapely.wkt import dumps  # type: ignore

    to_wkt = dumps


SIMPLE_LITERAL_TYPES = Union[
    bytes,
    str,
    int,
    bool,
    float,
    datetime.datetime,
    datetime.date,
    datetime.time,
    decimal.Decimal,
    list,
]


### Writing SQL Values (literals, column references, table references, etc.)
def simple_literal(value: Union[SIMPLE_LITERAL_TYPES, None]) -> str:
    """Return quoted input string."""

    # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#literals
    if value is None:
        return "NULL"
    elif isinstance(value, str):
        # Single quoting seems to work nicer with ibis than double quoting
        return f"'{googlesql._escape_chars(value)}'"
    elif isinstance(value, bytes):
        return repr(value)
    elif isinstance(value, (bool, int)):
        return str(value)
    elif isinstance(value, float):
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#floating_point_literals
        if math.isnan(value):
            return 'CAST("nan" as FLOAT)'
        if value == math.inf:
            return 'CAST("+inf" as FLOAT)'
        if value == -math.inf:
            return 'CAST("-inf" as FLOAT)'
        return str(value)
    # Check datetime first as it is a subclass of date
    elif isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            return f"DATETIME('{value.isoformat()}')"
        else:
            return f"TIMESTAMP('{value.isoformat()}')"
    elif isinstance(value, datetime.date):
        return f"DATE('{value.isoformat()}')"
    elif isinstance(value, datetime.time):
        return f"TIME(DATETIME('1970-01-01 {value.isoformat()}'))"
    elif isinstance(value, shapely.geometry.base.BaseGeometry):
        return f"ST_GEOGFROMTEXT({simple_literal(to_wkt(value))})"
    elif isinstance(value, decimal.Decimal):
        # TODO: disambiguate BIGNUMERIC based on scale and/or precision
        return f"CAST('{str(value)}' AS NUMERIC)"
    elif isinstance(value, list):
        simple_literals = [simple_literal(i) for i in value]
        return f"[{', '.join(simple_literals)}]"

    else:
        raise ValueError(f"Cannot produce literal for {value}")


def multi_literal(*values: str):
    literal_strings = [simple_literal(i) for i in values]
    return "(" + ", ".join(literal_strings) + ")"


def cast_as_string(column_name: str) -> str:
    """Return a string representing string casting of a column."""

    return googlesql.Cast(
        googlesql.ColumnExpression(column_name), googlesql.DataType.STRING
    ).sql()


def to_json_string(column_name: str) -> str:
    """Return a string representing JSON version of a column."""

    return f"TO_JSON_STRING({googlesql.identifier(column_name)})"


def csv(values: Iterable[str]) -> str:
    """Return a string of comma separated values."""
    return ", ".join(values)


def infix_op(opname: str, left_arg: str, right_arg: str):
    # Maybe should add parentheses??
    return f"{left_arg} {opname} {right_arg}"


def is_distinct_sql(columns: Iterable[str], table_ref: bigquery.TableReference) -> str:
    is_unique_sql = f"""WITH full_table AS (
        {googlesql.Select().from_(table_ref).select(columns).sql()}
    ),
    distinct_table AS (
        {googlesql.Select().from_(table_ref).select(columns, distinct=True).sql()}
    )

    SELECT (SELECT COUNT(*) FROM full_table) AS `total_count`,
    (SELECT COUNT(*) FROM distinct_table) AS `distinct_count`
    """
    return is_unique_sql


def ordering_clause(
    ordering: Iterable[bigframes.core.ordering.OrderingExpression],
) -> str:
    import bigframes.core.expression

    parts = []
    for col_ref in ordering:
        asc_desc = "ASC" if col_ref.direction.is_ascending else "DESC"
        null_clause = "NULLS LAST" if col_ref.na_last else "NULLS FIRST"
        ordering_expr = col_ref.scalar_expression
        # We don't know how to compile scalar expressions in isolation
        if ordering_expr.is_const:
            # Probably shouldn't have constants in ordering definition, but best to ignore if somehow they end up here.
            continue
        assert isinstance(ordering_expr, bigframes.core.expression.DerefOp)
        part = f"`{ordering_expr.id.sql}` {asc_desc} {null_clause}"
        parts.append(part)
    return f"ORDER BY {' ,'.join(parts)}"


def create_vector_index_ddl(
    *,
    replace: bool,
    index_name: str,
    table_name: str,
    column_name: str,
    stored_column_names: Collection[str],
    options: Mapping[str, Union[str | int | bool | float]] = {},
) -> str:
    """Encode the VECTOR INDEX statement for BigQuery Vector Search."""

    if replace:
        create = "CREATE OR REPLACE VECTOR INDEX "
    else:
        create = "CREATE VECTOR INDEX IF NOT EXISTS "

    if len(stored_column_names) > 0:
        escaped_stored = [
            f"{googlesql.identifier(name)}" for name in stored_column_names
        ]
        storing = f"STORING({', '.join(escaped_stored)}) "
    else:
        storing = ""

    rendered_options = ", ".join(
        [
            f"{option_name} = {simple_literal(option_value)}"
            for option_name, option_value in options.items()
        ]
    )

    return f"""
    {create} {googlesql.identifier(index_name)}
    ON {googlesql.identifier(table_name)}({googlesql.identifier(column_name)})
    {storing}
    OPTIONS({rendered_options});
    """


def create_vector_search_sql(
    sql_string: str,
    *,
    base_table: str,
    column_to_search: str,
    query_column_to_search: Optional[str] = None,
    top_k: Optional[int] = None,
    distance_type: Optional[str] = None,
    options: Optional[Mapping[str, Union[str | int | bool | float]]] = None,
) -> str:
    """Encode the VECTOR SEARCH statement for BigQuery Vector Search."""

    vector_search_args = [
        f"TABLE {googlesql.identifier(cast(str, base_table))}",
        f"{simple_literal(column_to_search)}",
        f"({sql_string})",
    ]

    if query_column_to_search is not None:
        vector_search_args.append(
            f"query_column_to_search => {simple_literal(query_column_to_search)}"
        )

    if top_k is not None:
        vector_search_args.append(f"top_k=> {simple_literal(top_k)}")

    if distance_type is not None:
        vector_search_args.append(f"distance_type => {simple_literal(distance_type)}")

    if options is not None:
        vector_search_args.append(
            f"options => {simple_literal(json.dumps(options, indent=None))}"
        )

    args_str = ",\n".join(vector_search_args)
    return f"""
    SELECT
        query.*,
        base.*,
        distance,
    FROM VECTOR_SEARCH({args_str})
    """


def _field_type_to_sql(field: bigquery.SchemaField) -> str:
    if field.field_type in ("RECORD", "STRUCT"):
        sub_defs = []
        for sub in field.fields:
            sub_type = _field_type_to_sql(sub)
            sub_def = f"{sub.name} {sub_type}"
            if sub.mode == "REQUIRED":
                sub_def += " NOT NULL"
            sub_defs.append(sub_def)
        type_str = f"STRUCT<{', '.join(sub_defs)}>"
    else:
        type_str = field.field_type

    if field.mode == "REPEATED":
        return f"ARRAY<{type_str}>"
    return type_str


def schema_field_to_sql(field: bigquery.SchemaField) -> str:
    """Convert a BigQuery SchemaField to a SQL DDL column definition."""
    type_sql = _field_type_to_sql(field)
    sql = f"{field.name} {type_sql}"
    if field.mode == "REQUIRED":
        sql += " NOT NULL"
    if field.description:
        sql += f" OPTIONS(description={simple_literal(field.description)})"
    return sql


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
        columns_sql = ",\n".join(schema_field_to_sql(field) for field in schema_fields)
        query += f"(\n{columns_sql}\n)\n"

    if partition_by:
        query += f"PARTITION BY {partition_by}\n"

    if cluster_by:
        query += f"CLUSTER BY {', '.join(cluster_by)}\n"

    if table_options:
        opts_list = []
        for k, v in table_options.items():
            opts_list.append(f"{k}={simple_literal(v)}")
        query += f"OPTIONS({', '.join(opts_list)})\n"

    files_opts = {}
    if load_options:
        files_opts.update(load_options)

    files_opts["uris"] = uris
    files_opts["format"] = format

    files_opts_list = []
    for k, v in files_opts.items():
        files_opts_list.append(f"{k}={simple_literal(v)}")

    query += f"FROM FILES({', '.join(files_opts_list)})\n"

    if hive_partition_columns:
        cols_sql = ",\n".join(
            schema_field_to_sql(field) for field in hive_partition_columns
        )
        query += f"WITH PARTITION COLUMNS (\n{cols_sql}\n)\n"
    elif hive_partition_columns is not None:
        query += "WITH PARTITION COLUMNS\n"

    if connection:
        query += f"WITH CONNECTION {connection}\n"

    return query
