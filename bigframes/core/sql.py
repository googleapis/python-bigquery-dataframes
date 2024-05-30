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
import math
import textwrap
import typing

import bigframes.core.transpiler.googlesql as sql

# Literals and identifiers matching this pattern can be unquoted
unquoted = r"^[A-Za-z_][A-Za-z_0-9]*$"


if typing.TYPE_CHECKING:
    import google.cloud.bigquery as bigquery

    import bigframes.core.ordering


### Writing SQL Values (literals, column references, table references, etc.)
def simple_literal(value: str | int | bool | float | datetime.datetime):
    """Return quoted input string."""
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#literals
    if isinstance(value, str):
        # Single quoting seems to work nicer with ibis than double quoting
        return f"'{escape_special_characters(value)}'"
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
    if isinstance(value, datetime.datetime):
        return f"TIMESTAMP('{value.isoformat()}')"
    else:
        raise ValueError(f"Cannot produce literal for {value}")


def multi_literal(*values: str):
    literal_strings = [simple_literal(i) for i in values]
    return "(" + ", ".join(literal_strings) + ")"


def identifier(id: str) -> str:
    """Return a string representing column reference in a SQL."""
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#identifiers
    # Just always escape, otherwise need to check against every reserved sql keyword
    return f"`{escape_special_characters(id)}`"


def escape_special_characters(value: str):
    """Escapes all special charactesrs"""
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#string_and_bytes_literals
    trans_table = str.maketrans(
        {
            "\a": r"\a",
            "\b": r"\b",
            "\f": r"\f",
            "\n": r"\n",
            "\r": r"\r",
            "\t": r"\t",
            "\v": r"\v",
            "\\": r"\\",
            "?": r"\?",
            '"': r"\"",
            "'": r"\'",
            "`": r"\`",
        }
    )
    return value.translate(trans_table)


def cast_as_string(column_name: str) -> str:
    """Return a string representing string casting of a column."""

    return f"CAST({identifier(column_name)} AS STRING)"


def csv(values: typing.Iterable[str]) -> str:
    """Return a string of comma separated values."""
    return ", ".join(values)


def table_reference(table_ref: bigquery.TableReference) -> str:
    return f"`{escape_special_characters(table_ref.project)}`.`{escape_special_characters(table_ref.dataset_id)}`.`{escape_special_characters(table_ref.table_id)}`"


def infix_op(opname: str, left_arg: str, right_arg: str):
    # Maybe should add parentheses??
    return f"{left_arg} {opname} {right_arg}"


### Writing SELECT expressions
def select_from_subquery(
    columns: typing.Iterable[str], subquery: str, distinct: bool = False
):
    select_list = [sql.SelectExpression(sql.Expression(col)) for col in columns]
    from_clause_list = [sql.FromClause(sql.FromItem(query_expr=subquery))]
    return sql.Select(
        select_list=select_list,
        distinct=distinct,
        from_clause_list=from_clause_list,
    ).sql()


# def select_from_table_ref(
#     columns: typing.Iterable[str],
#     table_ref: bigquery.TableReference,
#     distinct: bool = False,
# ):
#     select_list = [sql.SelectExpression(sql.Expression(col)) for col in columns]
#     from_clause_list = [
#         sql.FromClause(sql.FromItem(table_name=sql.TableRef(table_ref)))
#     ]
#     return sql.Select(
#         select_list=select_list,
#         distinct=distinct,
#         from_clause_list=from_clause_list,
#     ).sql()


# def select_table(table_ref: bigquery.TableReference):
#     from_clause_list = [
#         sql.FromClause(sql.FromItem(table_name=sql.TableRef(table_ref)))
#     ]
#     return sql.Select(
#         select_list=[sql.SelectExpression(sql.StarExpression())],
#         from_clause_list=from_clause_list,
#     ).sql()


def is_distinct_sql(
    columns: typing.Iterable[str], table_ref: bigquery.TableReference
) -> str:
    select_list = [sql.SelectExpression(sql.Expression(col)) for col in columns]
    from_clause_list = [
        sql.FromClause(sql.FromItem(table_name=sql.TableRef(table_ref=table_ref)))
    ]

    with_cte_list = [
        sql.NonRecursiveCTE(
            cte_name="full_table",
            query_expr=sql.QueryExpr(
                select=sql.Select(
                    select_list=select_list, from_clause_list=from_clause_list
                )
            ),
        ),
        sql.NonRecursiveCTE(
            cte_name="distinct_table",
            query_expr=sql.QueryExpr(
                select=sql.Select(
                    select_list=select_list,
                    from_clause_list=from_clause_list,
                    distinct=True,
                )
            ),
        ),
    ]

    select_count = [sql.SelectExpression(expression=sql.Count(expression=sql.StarExpression()))]
    full_table_count = sql.QueryExpr(
        sql.Select(
            select_list=select_count,
            from_clause_list=[sql.FromClause(sql.FromItem(table_name=sql.Expression("full_table")))],
        )
    )
    distinct_table_count = sql.QueryExpr(
        sql.Select(
            select_list=select_count,
            from_clause_list=[
                sql.FromClause(sql.FromItem(table_name=sql.Expression("distinct_table")))
            ],
        )
    )

    # TODO: Init Expression from sql.QueryExpr
    # SELECT expression
    # Items in a SELECT list can be expressions. These expressions evaluate to a
    # single value and produce one output column, with an optional explicit alias.
    return sql.QueryExpr(
        sql.Select(
            select_list=[
                sql.SelectExpression(sql.Expression(full_table_count.sql())),
                sql.SelectExpression(sql.Expression(distinct_table_count.sql())),
            ],
            from_clause_list=[
                sql.FromClause(sql.FromItem(table_name=sql.Expression("distinct_table")))
            ],
        ),
        with_cte_list=with_cte_list,
    ).sql()


def ordering_clause(
    ordering: typing.Iterable[bigframes.core.ordering.OrderingExpression],
) -> str:
    import bigframes.core.expression as expr

    order_by_sql_list: list[str]
    for col_ref in ordering:
        ordering_expr = col_ref.scalar_expression
        # We don't know how to compile scalar expressions in isolation.
        # Probably shouldn't have constants in ordering definition, but best to ignore
        # if somehow they end up here.
        if ordering_expr.is_const:
            continue

        assert isinstance(ordering_expr, expr.UnboundVariableExpression)

        order_by_sql_list.append(
            sql.OrderByExpr(
                sql.Expression(ordering_expr.id),
                asc=col_ref.direction.is_ascending,
                nulls_last=col_ref.na_last,
            ).sql()
        )
    return f"ORDER BY {' ,'.join(order_by_sql_list)}"
