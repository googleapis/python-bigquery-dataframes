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
import bigframes.core.transpiler.googlesql.types as types

# Aggregate functions:
# https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions


@dataclasses.dataclass
class Count(expr.ABCExpression):
    """GoogleSQL count aggregate functions."""

    name: str = "count"
    expression: expr.ABCExpression
    distinct: typing.Optional[bool] = None,

    def __post_init__(self):
        if (
            isinstance(self.expression, expr.StarExpression)
            and self.distinct is not None
        ):
            raise ValueError("Cannot count distinct over all.")

    def sql(self) -> str:
        return f"COUNT ({'DISTINCT ' if self.distinct else ''}{self.expression.sql()})"


# Conversion functions:
# https://cloud.google.com/bigquery/docs/reference/standard-sql/conversion_functions


@dataclasses.dataclass
class Cast(expr.ABCExpression):
    """GoogleSQL cast syntax."""

    expression: expr.ABCExpression
    type: types.DataType

    def __post_init__(self):
        # TODO: check if the type is converable.
        pass

    def sql(self) -> str:
        return f"CAST ({self.expression.sql()} AS {self.type.sql()})"
