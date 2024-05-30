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

import dataclasses
import typing

import bigframes.core.transpiler.googlesql.expression as expr
import bigframes.core.transpiler.googlesql.from_ as from_
import bigframes.core.transpiler.googlesql.sql as sql

"""Python classes for GoogleSQL SELECT statement, adhering to the official syntax rules: 
https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#select_list"""


# TODO: SelectAll is similar to StarExpression
# SelectList, SelectExpression, SelectAll are too nagging at the constructor.
# TODO: rename `expression` as `name` for simpler


@dataclasses.dataclass
class SelectExpression(sql.SQLSyntax):
    """GoogleSQL select_expression and select_all syntax."""

    expression: expr.ABCExpression
    alias: typing.Optional[expr.AsAlias] = None

    def __post_init__(self):
        if isinstance(self.expression, expr.StarExpression) and self.alias is not None:
            raise ValueError("Cannot alias when select star.")

    def sql(self) -> str:
        if self.alias is None:
            return self.expression.sql()
        else:
            return f"{self.expression.sql()} {self.alias}"


@dataclasses.dataclass
class Select(sql.SQLSyntax):
    """GoogleSQL select syntax."""

    select_list: typing.Sequence[SelectExpression]
    distinct: typing.Optional[bool] = None,
    from_clause_list: typing.Sequence[from_.FromClause] = ()
    where_expr: typing.Optional[expr.ABCExpression] = None
    # TODO: add group_by, having_expr, window_clause

    def __post_init__(self):
        if len(self.select_list) < 1:
            raise ValueError("At least select one expression.")

    def sql(self) -> str:
        text = ["SELECT"]

        if self.distinct is not None:
            text.append("DISTINCT")

        select_list_sql = "\n".join([select.sql() for select in self.select_list])
        text.append(select_list_sql)

        if self.from_clause_list is not None:
            from_clauses_sql = " ".join(
                [clause.sql() for clause in self.from_clause_list]
            )
            text.append(f"FROM {from_clauses_sql}")
        if self.where_expr is not None:
            text.append(f"WHERE {self.where_expr.sql()}")
        return "\n".join(text)
