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
import bigframes.core.transpiler.googlesql.join as join
import bigframes.core.transpiler.googlesql.query as query
import bigframes.core.transpiler.googlesql.sql as sql
import bigframes.core.transpiler.googlesql.unnest as unnest

"""Python classes for GoogleSQL FROM clause, adhering to the official syntax rules: 
https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#from_clause"""


@dataclasses.dataclass
class FromItem(sql.SQLSyntax):
    """GoogleSQL from_item syntax."""

    table_name: typing.Optional[expr.ABCExpression] = None
    query_expr: typing.Optional[query.QueryExpr | str] = None
    join_operation: typing.Optional[join.JoinOperation] = None
    unnest_operator: typing.Optional[unnest.UnnestOperator] = None
    cte_name: typing.Optional[expr.ABCExpression] = None
    as_alias: typing.Optional[expr.AsAlias] = None

    def __post_init__(self):
        non_none = sum(
            expr is not None
            for expr in [
                self.table_name,
                self.query_expr,
                self.join_operation,
                self.unnest_operator,
                self.cte_name,
            ]
        )
        if non_none != 1:
            raise ValueError("Exactly one of expressions must be provided.")
        if self.as_alias is not None and self.join_operation is not None:
            raise ValueError("The alias given to the join operation is not valid.")
        if self.as_alias is not None and self.unnest_operator is not None:
            raise ValueError("The alias given to the unnest operator is not valid.")

    def sql(self) -> str:
        if self.table_name is not None:
            text = self.table_name.sql()
        elif self.query_expr is not None:
            text = self.query_expr if isinstance(self.query_expr, str) else self.query_expr.sql()
            text = f"({text})"
        elif self.join_operation is not None:
            text = self.join_operation.sql()
        elif self.unnest_operator is not None:
            text = self.unnest_operator.sql()
        elif self.cte_name is not None:
            text = self.cte_name.sql()
        else:
            raise ValueError("One of from item must be provided.")

        if self.as_alias is not None:
            text = f"{text} {self.as_alias.sql()}"
        return text


@dataclasses.dataclass
class FromClause(sql.SQLSyntax):
    """GoogleSQL from_clause syntax."""

    from_item: FromItem

    def sql(self) -> str:
        return self.from_item.sql()
