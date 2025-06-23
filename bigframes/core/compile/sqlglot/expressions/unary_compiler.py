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
import typing

import sqlglot
from sqlglot import expressions as sge

from bigframes import operations as ops
from bigframes.core.compile.sqlglot.expressions import typed_expr


@functools.singledispatch
def compile(op: ops.UnaryOp, expr: typed_expr.TypedExpr) -> sge.Expression:
    raise TypeError(f"Unrecognized unary operator: {op.name}")


@compile.register
def _(op: ops.ArrayToStringOp, expr: typed_expr.TypedExpr) -> sge.Expression:
    return sge.ArrayToString(this=expr.sge_expr, expression=f"'{op.delimiter}'")


@compile.register
def _(op: ops.ArrayIndexOp, expr: typed_expr.TypedExpr) -> sge.Expression:
    offset = sge.Anonymous(
        this="safe_offset", expressions=[sge.Literal.number(op.index)]
    )
    return expr.sge_expr[offset]


@compile.register
def _(op: ops.ArraySliceOp, expr: typed_expr.TypedExpr) -> sge.Expression:
    slice_idx = sqlglot.to_identifier("slice_idx")

    conditions: typing.List[sge.Predicate] = [slice_idx >= op.start]

    if op.stop is not None:
        conditions.append(slice_idx < op.stop)

    # local name for each element in the array
    el = sqlglot.to_identifier("el")

    selected_elements = (
        sge.select(el)
        .from_(sge.Unnest(expressions=[expr.sge_expr], as_=el, offset=slice_idx))
        .where(*conditions)
    )

    return sge.array(selected_elements)
