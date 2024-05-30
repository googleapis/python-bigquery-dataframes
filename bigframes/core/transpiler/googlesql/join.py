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

"""Python classes for GoogleSQL JOIN operation, adhering to the official syntax rules: 
https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#join_types"""


@dataclasses.dataclass
class JoinOperator(sql.SQLSyntax):
    """GoogleSQL join_operator syntax."""

    how: typing.Literal["inner", "full", "left", "right", "cross"]

    def sql(self) -> str:
        return f"{self.how.upper()} JOIN"


@dataclasses.dataclass
class JoinCondition(sql.SQLSyntax):
    """GoogleSQL join_condition syntax."""

    on_clause: expr.ABCExpression

    def sql(self) -> str:
        return f"ON {self.on_clause.sql()}"


@dataclasses.dataclass
class JoinOperation(sql.SQLSyntax):
    """GoogleSQL join_operation syntax."""

    left: from_.FromItem
    right: from_.FromItem
    join_operator: JoinOperator
    join_condition: typing.Optional[JoinCondition] = None

    def __post_init__(self):
        if self.join_operator.how == "cross" and self.join_condition is not None:
            raise ValueError(
                "The condition given to the CROSS join operation is not valid."
            )

    def sql(self) -> str:
        text = [self.left.sql(), self.join_operator.sql(), self.right.sql()]
        if self.join_condition is not None:
            text.append(self.join_condition.sql())
        return "\n".join(text)
