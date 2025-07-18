# Copyright 2025 Google LLC
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
from typing import Union

from bigframes import operations as ops
from bigframes.core import expression
from bigframes.sql import column
import bigframes.sql.utils


def col(col: str) -> column.Column:
    return column.Column(expression.deref(col)).alias(col)


def lit(value) -> column.Column:
    return column.Column(expression.const(value)).alias(str(col))


def abs(value: Union[str, column.Column]) -> column.Column:
    expr = bigframes.sql.utils.resolve_column_or_name(value)
    return column.Column(ops.abs_op.as_expr(expr))


def expr(sql: str) -> column.Column:
    return column.Column(expression.RawSqlExpression(sql))
