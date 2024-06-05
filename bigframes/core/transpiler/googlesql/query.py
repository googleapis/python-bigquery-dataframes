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

import dataclasses
import typing

import bigframes.core.transpiler.googlesql.abc as abc
import bigframes.core.transpiler.googlesql.expression as expr

"""Python classes to defind GoogleSQL syntax nodes, adhering to the official syntax rules:
https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax"""


@dataclasses.dataclass
class QueryExpr(abc.SQLSyntax):
    """GoogleSQL query_expr syntax."""

    select: Select
    with_cte_list: typing.Sequence[NonRecursiveCTE] = ()

    def sql(self) -> str:
        text = []
        if len(self.with_cte_list) > 0:
            with_cte_text = ",\n".join(
                [with_cte.sql() for with_cte in self.with_cte_list]
            )
            text.append(f"WITH {with_cte_text}")

        text.append(self.select.sql())
        return "\n".join(text)


@dataclasses.dataclass
class SelectExpression(abc.SQLSyntax):
    """GoogleSQL select_expression and select_all syntax."""

    expression: expr.Expression
    alias: typing.Optional[expr.AliasExpression] = None

    def __post_init__(self):
        if isinstance(self.expression, expr.StarExpression) and self.alias is not None:
            raise ValueError("Cannot alias when select star.")

    def sql(self) -> str:
        if self.alias is None:
            return self.expression.sql()
        else:
            return f"{self.expression.sql()} {self.alias.sql()}"


@dataclasses.dataclass
class Select(abc.SQLSyntax):
    """GoogleSQL select syntax."""

    select_list: typing.Sequence[SelectExpression]
    from_clause_list: typing.Sequence[FromClause] = ()

    def sql(self) -> str:
        text = ["SELECT"]

        select_list_sql = ",\n".join([select.sql() for select in self.select_list])
        text.append(select_list_sql)

        if self.from_clause_list is not None:
            from_clauses_sql = ",\n".join(
                [clause.sql() for clause in self.from_clause_list]
            )
            text.append(f"FROM\n{from_clauses_sql}")
        return "\n".join(text)


@dataclasses.dataclass
class FromItem(abc.SQLSyntax):
    """GoogleSQL from_item syntax."""

    table_name: typing.Optional[expr.TableExpression] = None
    query_expr: typing.Optional[QueryExpr | str] = None
    cte_name: typing.Optional[expr.Expression] = None
    alias: typing.Optional[expr.AliasExpression] = None

    def __post_init__(self):
        non_none = sum(
            expr is not None
            for expr in [
                self.table_name,
                self.query_expr,
                self.cte_name,
            ]
        )
        if non_none != 1:
            raise ValueError("Exactly one of expressions must be provided.")

    def sql(self) -> str:
        if self.table_name is not None:
            text = self.table_name.sql()
        elif self.query_expr is not None:
            text = (
                self.query_expr
                if isinstance(self.query_expr, str)
                else self.query_expr.sql()
            )
            text = f"({text})"
        elif self.cte_name is not None:
            text = self.cte_name.sql()
        else:
            raise ValueError("One of from item must be provided.")

        if self.alias is not None:
            text = f"{text} {self.alias.sql()}"
        return text


@dataclasses.dataclass
class FromClause(abc.SQLSyntax):
    """GoogleSQL from_clause syntax."""

    from_item: FromItem

    def sql(self) -> str:
        return self.from_item.sql()


@dataclasses.dataclass
class NonRecursiveCTE(abc.SQLSyntax):
    """GoogleSQL non_recursive_cte syntax."""

    cte_name: expr.CTEExpression
    query_expr: QueryExpr

    def sql(self) -> str:
        return f"{self.cte_name.sql()} AS (\n{self.query_expr.sql()}\n)"
