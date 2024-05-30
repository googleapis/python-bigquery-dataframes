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

import bigframes.core.transpiler.googlesql.cte as cte
import bigframes.core.transpiler.googlesql.expression as expr
import bigframes.core.transpiler.googlesql.select as select
import bigframes.core.transpiler.googlesql.sql as sql

"""Python classes for GoogleSQL query statement, adhering to the official syntax rules: 
https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#sql_syntax"""


@dataclasses.dataclass
class OrderByExpr(sql.SQLSyntax):
    """GoogleSQL order_by_expr syntax."""

    expression: expr.ABCExpression
    asc: typing.Optional[bool] = None
    nulls_last: typing.Optional[bool] = None

    def sql(self) -> str:
        text = f"{self.expression.sql()}"
        if self.asc is not None:
            text = f"{text} {'ASC' if self.asc else 'DESC'}"
        if self.nulls_last is not None:
            text = f"{text} NULL {'LAST' if self.nulls_last else 'FIRST'}"
        return text


@dataclasses.dataclass
class QueryExpr(sql.SQLSyntax):
    """GoogleSQL query_expr syntax."""

    select: select.Select
    with_cte_list: typing.Sequence[cte.NonRecursiveCTE] = ()
    order_by_expr_list: typing.Sequence[OrderByExpr] = ()
    limit_count: typing.Optional[int] = None

    def sql(self) -> str:
        text = []
        if len(self.with_cte_list) > 0:
            with_cte_text = ",\n".join(
                [with_cte.sql() for with_cte in self.with_cte_list]
            )
            text.append(f"WITH {with_cte_text}")

        text.append(self.select.sql())

        if len(self.order_by_expr_list) > 0:
            order_by_text = ", ".join([expr.sql() for expr in self.order_by_expr_list])
            text.append(f"ORDER BY {order_by_text}")

        if self.limit_count is not None:
            text.append(f"LIMIT {self.limit_count}")
        return "\n".join(text)
