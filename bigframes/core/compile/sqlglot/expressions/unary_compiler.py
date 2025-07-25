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

import typing

import sqlglot
import sqlglot.expressions as sge

from bigframes import operations as ops
from bigframes.core.compile.sqlglot.expressions.op_registration import OpRegistration
from bigframes.core.compile.sqlglot.expressions.typed_expr import TypedExpr

_NAN = sge.Cast(this=sge.convert("NaN"), to="FLOAT64")
_INF = sge.Cast(this=sge.convert("Infinity"), to="FLOAT64")

# Approx Highest number you can pass in to EXP function and get a valid FLOAT64 result
# FLOAT64 has 11 exponent bits, so max values is about 2**(2**10)
# ln(2**(2**10)) == (2**10)*ln(2) ~= 709.78, so EXP(x) for x>709.78 will overflow.
_FLOAT64_EXP_BOUND = sge.convert(709.78)

UNARY_OP_REGISTRATION = OpRegistration()


def compile(op: ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return UNARY_OP_REGISTRATION[op](op, expr)


@UNARY_OP_REGISTRATION.register(ops.abs_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Abs(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.arccosh_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=expr.expr < sge.convert(1),
                true=_NAN,
            )
        ],
        default=sge.func("ACOSH", expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.arccos_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=sge.func("ABS", expr.expr) > sge.convert(1),
                true=_NAN,
            )
        ],
        default=sge.func("ACOS", expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.arcsin_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=sge.func("ABS", expr.expr) > sge.convert(1),
                true=_NAN,
            )
        ],
        default=sge.func("ASIN", expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.arcsinh_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("ASINH", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.arctan_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("ATAN", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.arctanh_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=sge.func("ABS", expr.expr) > sge.convert(1),
                true=_NAN,
            )
        ],
        default=sge.func("ATANH", expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.ArrayToStringOp)
def _(op: ops.ArrayToStringOp, expr: TypedExpr) -> sge.Expression:
    return sge.ArrayToString(this=expr.expr, expression=f"'{op.delimiter}'")


@UNARY_OP_REGISTRATION.register(ops.ArrayIndexOp)
def _(op: ops.ArrayIndexOp, expr: TypedExpr) -> sge.Expression:
    return sge.Bracket(
        this=expr.expr,
        expressions=[sge.Literal.number(op.index)],
        safe=True,
        offset=False,
    )


@UNARY_OP_REGISTRATION.register(ops.ArraySliceOp)
def _(op: ops.ArraySliceOp, expr: TypedExpr) -> sge.Expression:
    slice_idx = sqlglot.to_identifier("slice_idx")

    conditions: typing.List[sge.Predicate] = [slice_idx >= op.start]

    if op.stop is not None:
        conditions.append(slice_idx < op.stop)

    # local name for each element in the array
    el = sqlglot.to_identifier("el")

    selected_elements = (
        sge.select(el)
        .from_(
            sge.Unnest(
                expressions=[expr.expr],
                alias=sge.TableAlias(columns=[el]),
                offset=slice_idx,
            )
        )
        .where(*conditions)
    )

    return sge.array(selected_elements)


@UNARY_OP_REGISTRATION.register(ops.capitalize_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Initcap(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.ceil_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Ceil(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.cos_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("COS", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.cosh_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=sge.func("ABS", expr.expr) > sge.convert(709.78),
                true=_INF,
            )
        ],
        default=sge.func("COSH", expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.StrContainsRegexOp)
def _(op: ops.StrContainsRegexOp, expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(op.pat))


@UNARY_OP_REGISTRATION.register(ops.StrContainsOp)
def _(op: ops.StrContainsOp, expr: TypedExpr) -> sge.Expression:
    return sge.Like(this=expr.expr, expression=sge.convert(f"%{op.pat}%"))


@UNARY_OP_REGISTRATION.register(ops.date_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Date(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.day_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="DAY"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.dayofweek_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    # Adjust the 1-based day-of-week index (from SQL) to a 0-based index.
    return sge.Extract(
        this=sge.Identifier(this="DAYOFWEEK"), expression=expr.expr
    ) - sge.convert(1)


@UNARY_OP_REGISTRATION.register(ops.dayofyear_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="DAYOFYEAR"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.exp_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=expr.expr > _FLOAT64_EXP_BOUND,
                true=_INF,
            )
        ],
        default=sge.func("EXP", expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.expm1_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=expr.expr > _FLOAT64_EXP_BOUND,
                true=_INF,
            )
        ],
        default=sge.func("EXP", expr.expr),
    ) - sge.convert(1)


@UNARY_OP_REGISTRATION.register(ops.floor_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Floor(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.geo_area_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("ST_AREA", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.geo_st_astext_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("ST_ASTEXT", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.geo_x_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("SAFE.ST_X", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.geo_y_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("SAFE.ST_Y", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.hash_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("FARM_FINGERPRINT", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.hour_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="HOUR"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.invert_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.BitwiseNot(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.isalnum_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^(\p{N}|\p{L})+$"))


@UNARY_OP_REGISTRATION.register(ops.isalpha_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\p{L}+$"))


@UNARY_OP_REGISTRATION.register(ops.isdecimal_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\d+$"))


@UNARY_OP_REGISTRATION.register(ops.isdigit_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\p{Nd}+$"))


@UNARY_OP_REGISTRATION.register(ops.islower_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
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


@UNARY_OP_REGISTRATION.register(ops.isnumeric_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\pN+$"))


@UNARY_OP_REGISTRATION.register(ops.isspace_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.RegexpLike(this=expr.expr, expression=sge.convert(r"^\s+$"))


@UNARY_OP_REGISTRATION.register(ops.isupper_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
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


@UNARY_OP_REGISTRATION.register(ops.len_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Length(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.ln_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=expr.expr < sge.convert(0),
                true=_NAN,
            )
        ],
        default=sge.Ln(this=expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.log10_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=expr.expr < sge.convert(0),
                true=_NAN,
            )
        ],
        default=sge.Log(this=expr.expr, expression=sge.convert(10)),
    )


@UNARY_OP_REGISTRATION.register(ops.log1p_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=expr.expr < sge.convert(-1),
                true=_NAN,
            )
        ],
        default=sge.Ln(this=sge.convert(1) + expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.lower_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Lower(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.minute_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="MINUTE"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.month_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="MONTH"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.StrLstripOp)
def _(op: ops.StrLstripOp, expr: TypedExpr) -> sge.Expression:
    return sge.Trim(this=expr.expr, expression=sge.convert(op.to_strip), side="LEFT")


@UNARY_OP_REGISTRATION.register(ops.neg_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Neg(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.normalize_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.TimestampTrunc(this=expr.expr, unit=sge.Identifier(this="DAY"))


@UNARY_OP_REGISTRATION.register(ops.pos_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return expr.expr


@UNARY_OP_REGISTRATION.register(ops.quarter_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="QUARTER"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.reverse_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("REVERSE", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.second_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="SECOND"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.StrRstripOp)
def _(op: ops.StrRstripOp, expr: TypedExpr) -> sge.Expression:
    return sge.Trim(this=expr.expr, expression=sge.convert(op.to_strip), side="RIGHT")


@UNARY_OP_REGISTRATION.register(ops.sqrt_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=expr.expr < sge.convert(0),
                true=_NAN,
            )
        ],
        default=sge.Sqrt(this=expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.StrStripOp)
def _(op: ops.StrStripOp, expr: TypedExpr) -> sge.Expression:
    return sge.Trim(this=sge.convert(op.to_strip), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.iso_day_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="DAYOFWEEK"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.iso_week_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="ISOWEEK"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.iso_year_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="ISOYEAR"), expression=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.isnull_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Is(this=expr.expr, expression=sge.Null())


@UNARY_OP_REGISTRATION.register(ops.notnull_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Not(this=sge.Is(this=expr.expr, expression=sge.Null()))


@UNARY_OP_REGISTRATION.register(ops.sin_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("SIN", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.sinh_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Case(
        ifs=[
            sge.If(
                this=sge.func("ABS", expr.expr) > _FLOAT64_EXP_BOUND,
                true=sge.func("SIGN", expr.expr) * _INF,
            )
        ],
        default=sge.func("SINH", expr.expr),
    )


@UNARY_OP_REGISTRATION.register(ops.StrGetOp)
def _(op: ops.StrGetOp, expr: TypedExpr) -> sge.Expression:
    return sge.Substring(
        this=expr.expr,
        start=sge.convert(op.i + 1),
        length=sge.convert(1),
    )


@UNARY_OP_REGISTRATION.register(ops.StrSliceOp)
def _(op: ops.StrSliceOp, expr: TypedExpr) -> sge.Expression:
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


@UNARY_OP_REGISTRATION.register(ops.tan_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("TAN", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.tanh_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("TANH", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.time_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.func("TIME", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.timedelta_floor_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Floor(this=expr.expr)


# JSON Ops
@UNARY_OP_REGISTRATION.register(ops.JSONExtract)
def _(op: ops.JSONExtract, expr: TypedExpr) -> sge.Expression:
    return sge.func("JSON_EXTRACT", expr.expr, sge.convert(op.json_path))


@UNARY_OP_REGISTRATION.register(ops.JSONExtractArray)
def _(op: ops.JSONExtractArray, expr: TypedExpr) -> sge.Expression:
    return sge.func("JSON_EXTRACT_ARRAY", expr.expr, sge.convert(op.json_path))


@UNARY_OP_REGISTRATION.register(ops.JSONExtractStringArray)
def _(op: ops.JSONExtractStringArray, expr: TypedExpr) -> sge.Expression:
    return sge.func("JSON_EXTRACT_STRING_ARRAY", expr.expr, sge.convert(op.json_path))


@UNARY_OP_REGISTRATION.register(ops.JSONQuery)
def _(op: ops.JSONQuery, expr: TypedExpr) -> sge.Expression:
    return sge.func("JSON_QUERY", expr.expr, sge.convert(op.json_path))


@UNARY_OP_REGISTRATION.register(ops.JSONQueryArray)
def _(op: ops.JSONQueryArray, expr: TypedExpr) -> sge.Expression:
    return sge.func("JSON_QUERY_ARRAY", expr.expr, sge.convert(op.json_path))


@UNARY_OP_REGISTRATION.register(ops.JSONValue)
def _(op: ops.JSONValue, expr: TypedExpr) -> sge.Expression:
    return sge.func("JSON_VALUE", expr.expr, sge.convert(op.json_path))


@UNARY_OP_REGISTRATION.register(ops.JSONValueArray)
def _(op: ops.JSONValueArray, expr: TypedExpr) -> sge.Expression:
    return sge.func("JSON_VALUE_ARRAY", expr.expr, sge.convert(op.json_path))


@UNARY_OP_REGISTRATION.register(ops.ParseJSON)
def _(op: ops.ParseJSON, expr: TypedExpr) -> sge.Expression:
    return sge.func("PARSE_JSON", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.ToJSONString)
def _(op: ops.ToJSONString, expr: TypedExpr) -> sge.Expression:
    return sge.func("TO_JSON_STRING", expr.expr)


@UNARY_OP_REGISTRATION.register(ops.upper_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Upper(this=expr.expr)


@UNARY_OP_REGISTRATION.register(ops.year_op)
def _(op: ops.base_ops.UnaryOp, expr: TypedExpr) -> sge.Expression:
    return sge.Extract(this=sge.Identifier(this="YEAR"), expression=expr.expr)
