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

import functools

import sqlglot.expressions as sge

from bigframes import operations as ops
from bigframes.core.compile.sqlglot.expressions.typed_expr import TypedExpr
import bigframes.core.compile.sqlglot.scalar_compiler as scalar_compiler

register_unary_op = scalar_compiler.scalar_op_compiler.register_unary_op


@register_unary_op(ops.capitalize_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.Initcap(this=expr.expr)


@register_unary_op(ops.StrContainsOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrContainsOp) -> sge.Expression:
    return sge.Like(this=expr.expr, expression=sge.convert(f"%{op.pat}%"))


@register_unary_op(ops.StrContainsRegexOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrContainsRegexOp) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(op.pat))


@register_unary_op(ops.StrExtractOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrExtractOp) -> sge.Expression:
    return sge.RegexpExtract(
        this=expr.expr, expression=sge.convert(op.pat), group=sge.convert(op.n)
    )


@register_unary_op(ops.StrFindOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrFindOp) -> sge.Expression:
    # INSTR is 1-based, so we need to adjust the start position.
    start = sge.convert(op.start + 1) if op.start is not None else sge.convert(1)
    if op.end is not None:
        # BigQuery's INSTR doesn't support `end`, so we need to use SUBSTR.
        return sge.func(
            "INSTR",
            sge.Substring(
                this=expr.expr,
                start=start,
                length=sge.convert(op.end - (op.start or 0)),
            ),
            sge.convert(op.substr),
        ) - sge.convert(1)
    else:
        return sge.func(
            "INSTR",
            expr.expr,
            sge.convert(op.substr),
            start,
        ) - sge.convert(1)


@register_unary_op(ops.StrLstripOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrLstripOp) -> sge.Expression:
    return sge.Trim(this=expr.expr, expression=sge.convert(op.to_strip), side="LEFT")


@register_unary_op(ops.StrPadOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrPadOp) -> sge.Expression:
    pad_length = sge.func(
        "GREATEST", sge.Length(this=expr.expr), sge.convert(op.length)
    )
    if op.side == "left":
        return sge.func(
            "LPAD",
            expr.expr,
            pad_length,
            sge.convert(op.fillchar),
        )
    elif op.side == "right":
        return sge.func(
            "RPAD",
            expr.expr,
            pad_length,
            sge.convert(op.fillchar),
        )
    else:  # side == both
        lpad_amount = sge.Cast(
            this=sge.func(
                "SAFE_DIVIDE",
                sge.Sub(this=pad_length, expression=sge.Length(this=expr.expr)),
                sge.convert(2),
            ),
            to="INT64",
        ) + sge.Length(this=expr.expr)
        return sge.func(
            "RPAD",
            sge.func(
                "LPAD",
                expr.expr,
                lpad_amount,
                sge.convert(op.fillchar),
            ),
            pad_length,
            sge.convert(op.fillchar),
        )


@register_unary_op(ops.StrRepeatOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrRepeatOp) -> sge.Expression:
    return sge.Repeat(this=expr.expr, times=sge.convert(op.repeats))


@register_unary_op(ops.EndsWithOp, pass_op=True)
def _(expr: TypedExpr, op: ops.EndsWithOp) -> sge.Expression:
    if not op.pat:
        return sge.false()

    def to_endswith(pat: str) -> sge.Expression:
        return sge.func("ENDS_WITH", expr.expr, sge.convert(pat))

    conditions = [to_endswith(pat) for pat in op.pat]
    return functools.reduce(lambda x, y: sge.Or(this=x, expression=y), conditions)


@register_unary_op(ops.isalnum_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^(\p{N}|\p{L})+$"))


@register_unary_op(ops.isalpha_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\p{L}+$"))


@register_unary_op(ops.isdecimal_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\d+$"))


@register_unary_op(ops.isdigit_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\p{Nd}+$"))


@register_unary_op(ops.islower_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.And(
        this=sge.EQ(
            this=sge.Lower(this=expr.expr),
            expression=expr.expr,
        ),
        expression=sge.NEQ(
            this=sge.Upper(this=expr.expr),
            expression=expr.expr,
        ),
    )


@register_unary_op(ops.isnumeric_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\pN+$"))


@register_unary_op(ops.isspace_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\s+$"))


@register_unary_op(ops.isupper_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.And(
        this=sge.EQ(
            this=sge.Upper(this=expr.expr),
            expression=expr.expr,
        ),
        expression=sge.NEQ(
            this=sge.Lower(this=expr.expr),
            expression=expr.expr,
        ),
    )


@register_unary_op(ops.len_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.Length(this=expr.expr)


@register_unary_op(ops.lower_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.Lower(this=expr.expr)


@register_unary_op(ops.ReplaceStrOp, pass_op=True)
def _(expr: TypedExpr, op: ops.ReplaceStrOp) -> sge.Expression:
    return sge.func("REPLACE", expr.expr, sge.convert(op.pat), sge.convert(op.repl))


@register_unary_op(ops.RegexReplaceStrOp, pass_op=True)
def _(expr: TypedExpr, op: ops.RegexReplaceStrOp) -> sge.Expression:
    return sge.func(
        "REGEXP_REPLACE", expr.expr, sge.convert(op.pat), sge.convert(op.repl)
    )


@register_unary_op(ops.reverse_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.func("REVERSE", expr.expr)


@register_unary_op(ops.StrRstripOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrRstripOp) -> sge.Expression:
    return sge.Trim(this=expr.expr, expression=sge.convert(op.to_strip), side="RIGHT")


@register_unary_op(ops.StartsWithOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StartsWithOp) -> sge.Expression:
    if not op.pat:
        return sge.false()

    def to_startswith(pat: str) -> sge.Expression:
        return sge.func("STARTS_WITH", expr.expr, sge.convert(pat))

    conditions = [to_startswith(pat) for pat in op.pat]
    return functools.reduce(lambda x, y: sge.Or(this=x, expression=y), conditions)


@register_unary_op(ops.StrStripOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrStripOp) -> sge.Expression:
    return sge.Trim(this=sge.convert(op.to_strip), expression=expr.expr)


@register_unary_op(ops.StringSplitOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StringSplitOp) -> sge.Expression:
    return sge.Split(this=expr.expr, expression=sge.convert(op.pat))


@register_unary_op(ops.StrGetOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrGetOp) -> sge.Expression:
    return sge.Substring(
        this=expr.expr,
        start=sge.convert(op.i + 1),
        length=sge.convert(1),
    )


@register_unary_op(ops.StrSliceOp, pass_op=True)
def _(expr: TypedExpr, op: ops.StrSliceOp) -> sge.Expression:
    start = op.start + 1 if op.start is not None else None
    if op.end is None:
        length = None
    elif op.start is None:
        length = op.end
    else:
        length = op.end - op.start
    return sge.Substring(
        this=expr.expr,
        start=sge.convert(start) if start is not None else None,
        length=sge.convert(length) if length is not None else None,
    )


@register_unary_op(ops.upper_op)
def _(expr: TypedExpr) -> sge.Expression:
    return sge.Upper(this=expr.expr)


@register_unary_op(ops.ZfillOp, pass_op=True)
def _(expr: TypedExpr, op: ops.ZfillOp) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=sge.EQ(
                    this=sge.Substring(
                        this=expr.expr, start=sge.convert(1), length=sge.convert(1)
                    ),
                    expression=sge.convert("-"),
                ),
                true=sge.Concat(
                    expressions=[
                        sge.convert("-"),
                        sge.func(
                            "LPAD",
                            sge.Substring(this=expr.expr, start=sge.convert(1)),
                            sge.convert(op.width - 1),
                            sge.convert("0"),
                        ),
                    ]
                ),
            )
        ],
        default=sge.func("LPAD", expr.expr, sge.convert(op.width), sge.convert("0")),
    )
