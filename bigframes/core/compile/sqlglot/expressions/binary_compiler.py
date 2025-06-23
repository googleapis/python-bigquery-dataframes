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

from bigframes import dtypes
from bigframes import operations as ops
from bigframes.core.compile.sqlglot.expressions import typed_expr


# TODO: add parenthesize for operators
@functools.singledispatch
def compile(
    op: ops.BinaryOp, left: typed_expr.TypedExpr, right: typed_expr.TypedExpr
) -> sge.Expression:
    raise TypeError(f"Unrecognized binary operator: {op.name}")


@compile.register
def _(
    op: ops.AddOp, left: typed_expr.TypedExpr, right: typed_expr.TypedExpr
) -> sge.Expression:
    if left.dtype == dtypes.STRING_DTYPE and right.dtype == dtypes.STRING_DTYPE:
        # String addition
        return sge.Concat(expressions=[left.sge_expr, right.sge_expr])

    # Numerical addition
    return sge.Add(this=left.sge_expr, expression=right.sge_expr)
