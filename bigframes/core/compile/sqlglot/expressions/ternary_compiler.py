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

import functools

from sqlglot import expressions as sge

from bigframes import operations as ops
from bigframes.core.compile.sqlglot.expressions import typed_expr


@functools.singledispatch
def compile(
    op: ops.TernaryOp,
    expr1: typed_expr.TypedExpr,
    expr2: typed_expr.TypedExpr,
    expr3: typed_expr.TypedExpr,
) -> sge.Expression:
    raise TypeError(f"Unrecognized ternary operator: {op.name}")
