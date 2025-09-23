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

from __future__ import annotations

import sqlglot.expressions as sge

from bigframes import dtypes
from bigframes import operations as ops
from bigframes.core.compile.sqlglot.expressions.typed_expr import TypedExpr
import bigframes.core.compile.sqlglot.scalar_compiler as scalar_compiler
from bigframes.core.compile.sqlglot.sqlglot_types import SQLGlotType

register_unary_op = scalar_compiler.scalar_op_compiler.register_unary_op


@register_unary_op(ops.AsTypeOp, pass_op=True)
def _(expr: TypedExpr, op: ops.AsTypeOp) -> sge.Expression:
    from_type = expr.dtype
    to_type = op.to_type
    sg_to_type = SQLGlotType.from_bigframes_dtype(to_type)
    sg_expr = expr.expr

    if to_type == dtypes.JSON_DTYPE:
        return _cast_to_json(expr, op)

    if from_type == dtypes.JSON_DTYPE:
        return _cast_from_json(expr, op)

    if to_type == dtypes.INT_DTYPE:
        result = _cast_to_int(expr, op)
        if result is not None:
            return result

    if to_type == dtypes.FLOAT_DTYPE and from_type == dtypes.BOOL_DTYPE:
        sg_expr = _cast(sg_expr, "INT64", op.safe)
        return _cast(sg_expr, sg_to_type, op.safe)

    if to_type == dtypes.BOOL_DTYPE:
        if from_type == dtypes.BOOL_DTYPE:
            return sg_expr
        else:
            return sge.NEQ(this=sg_expr, expression=sge.convert(0))

    if to_type == dtypes.STRING_DTYPE:
        sg_expr = _cast(sg_expr, sg_to_type, op.safe)
        if from_type == dtypes.BOOL_DTYPE:
            sg_expr = sge.func("INITCAP", sg_expr)
        return sg_expr

    if dtypes.is_time_like(to_type) and from_type == dtypes.INT_DTYPE:
        sg_expr = sge.func("TIMESTAMP_MICROS", sg_expr)
        return _cast(sg_expr, sg_to_type, op.safe)

    return _cast(sg_expr, sg_to_type, op.safe)


@register_unary_op(ops.hash_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.func("FARM_FINGERPRINT", expr.expr)


@register_unary_op(ops.isnull_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.Is(this=expr.expr, expression=sge.Null())


@register_unary_op(ops.MapOp, pass_op=True)
def _(expr: TypedExpr, op: ops.MapOp) -> sge.Expression:
    return sge.Case(
        this=expr.expr,
        ifs=[
            sge.If(this=sge.convert(key), true=sge.convert(value))
            for key, value in op.mappings
        ],
    )


@register_unary_op(ops.notnull_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.Not(this=sge.Is(this=expr.expr, expression=sge.Null()))


# Helper functions
def _cast_to_json(expr: TypedExpr, op: ops.AsTypeOp) -> sge.Expression:
    from_type = expr.dtype
    sg_expr = expr.expr

    if from_type == dtypes.STRING_DTYPE:
        func_name = "PARSE_JSON_IN_SAFE" if op.safe else "PARSE_JSON"
        return sge.func(func_name, sg_expr)
    if from_type in (dtypes.INT_DTYPE, dtypes.BOOL_DTYPE, dtypes.FLOAT_DTYPE):
        sg_expr = sge.Cast(this=sg_expr, to="STRING")
        return sge.func("PARSE_JSON", sg_expr)
    raise TypeError(f"Cannot cast from {from_type} to {dtypes.JSON_DTYPE}")


def _cast_from_json(expr: TypedExpr, op: ops.AsTypeOp) -> sge.Expression:
    to_type = op.to_type
    sg_expr = expr.expr
    func_name = ""
    if to_type == dtypes.INT_DTYPE:
        func_name = "INT64"
    elif to_type == dtypes.FLOAT_DTYPE:
        func_name = "FLOAT64"
    elif to_type == dtypes.BOOL_DTYPE:
        func_name = "BOOL"
    elif to_type == dtypes.STRING_DTYPE:
        func_name = "STRING"
    if func_name:
        func_name = "SAFE." + func_name if op.safe else func_name
        return sge.func(func_name, sg_expr)
    raise TypeError(f"Cannot cast from {dtypes.JSON_DTYPE} to {to_type}")


def _cast_to_int(expr: TypedExpr, op: ops.AsTypeOp) -> sge.Expression | None:
    from_type = expr.dtype
    sg_expr = expr.expr
    # Cannot cast DATETIME to INT directly so need to convert to TIMESTAMP first.
    if from_type == dtypes.DATETIME_DTYPE:
        sg_expr = _cast(sg_expr, "TIMESTAMP", op.safe)
        return sge.func("UNIX_MICROS", sg_expr)
    if from_type == dtypes.TIMESTAMP_DTYPE:
        return sge.func("UNIX_MICROS", sg_expr)
    if from_type == dtypes.TIME_DTYPE:
        return sge.func(
            "TIME_DIFF",
            _cast(sg_expr, "TIME", op.safe),
            sge.convert("00:00:00"),
            "MICROSECOND",
        )
    if from_type == dtypes.NUMERIC_DTYPE or from_type == dtypes.FLOAT_DTYPE:
        sg_expr = sge.func("TRUNC", sg_expr)
        return _cast(sg_expr, "INT64", op.safe)
    return None


def _cast(expr: sge.Expression, to: str, safe: bool):
    if safe:
        return sge.TryCast(this=expr, to=to)
    else:
        return sge.Cast(this=expr, to=to)
