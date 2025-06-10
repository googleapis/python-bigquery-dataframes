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

from bigframes.core import expression
import bigframes.core.compile.sqlglot.sqlglot_ir as ir
import bigframes.operations as ops


@functools.singledispatch
def compile_scalar_expression(
    expression: expression.Expression,
) -> sge.Expression:
    """Compiles BigFrames scalar expression into SQLGlot expression."""
    raise ValueError(f"Can't compile unrecognized node: {expression}")


@compile_scalar_expression.register
def compile_deref_expression(expr: expression.DerefOp) -> sge.Expression:
    return sge.ColumnDef(this=sge.to_identifier(expr.id.sql, quoted=True))


@compile_scalar_expression.register
def compile_field_ref_expression(
    expr: expression.SchemaFieldRefExpression,
) -> sge.Expression:
    return sge.ColumnDef(this=sge.to_identifier(expr.field.id.sql, quoted=True))


@compile_scalar_expression.register
def compile_constant_expression(
    expr: expression.ScalarConstantExpression,
) -> sge.Expression:
    return ir._literal(expr.value, expr.dtype)


@compile_scalar_expression.register
def compile_op_expression(expr: expression.OpExpression):
    # Non-recursively compiles the children scalar expressions.
    args = tuple(map(compile_scalar_expression, expr.inputs))

    op = expr.op
    op_name = expr.op.__class__.__name__
    method_name = f"compile_{op_name.lower()}"
    method = globals().get(method_name, None)
    if method is None:
        raise ValueError(
            f"Compilation method '{method_name}' not found for operator '{op_name}'."
        )

    if isinstance(op, ops.UnaryOp):
        return method(op, args[0])
    elif isinstance(op, ops.BinaryOp):
        return method(op, args[0], args[1])
    elif isinstance(op, ops.TernaryOp):
        return method(op, args[0], args[1], args[2])
    elif isinstance(op, ops.NaryOp):
        return method(op, *args)
    else:
        raise TypeError(
            f"Operator '{op_name}' has an unrecognized arity or type "
            "and cannot be compiled."
        )


def compile_rowkey(op: ops.RowKey, *columns: sge.Expression) -> sge.Expression:
    # TODO: add `uid_gen` to the scalar_compiler instance
    # ordering_hash_part = guid.generate_guid("bford_")
    # ordering_hash_part2 = guid.generate_guid("bford_")
    # ordering_rand_part = guid.generate_guid("bford_")

    # cast_to_str = lambda a : ir._cast(a, "STRING")
    # str_columns = list(map(cast_to_str, ))
    return sge.Star()

    # # All inputs into hash must be non-null or resulting hash will be null
    # str_values = list(map(_convert_to_nonnull_string, columns))
    # full_row_str = (
    #     str_values[0].concat(*str_values[1:]) if len(str_values) > 1 else str_values[0]
    # )
    # full_row_hash = (
    #     full_row_str.hash()
    #     .name(ordering_hash_part)
    #     .cast(ibis_dtypes.String(nullable=True))
    # )
    # # By modifying value slightly, we get another hash uncorrelated with the first
    # full_row_hash_p2 = (
    #     (full_row_str + "_")
    #     .hash()
    #     .name(ordering_hash_part2)
    #     .cast(ibis_dtypes.String(nullable=True))
    # )
    # # Used to disambiguate between identical rows (which will have identical hash)
    # random_value = (
    #     bigframes_vendored.ibis.random()
    #     .name(ordering_rand_part)
    #     .cast(ibis_dtypes.String(nullable=True))
    # )

    # return full_row_hash.concat(full_row_hash_p2, random_value)


# TODO: add parenthesize for operators
def compile_addop(
    op: ops.AddOp, left: sge.Expression, right: sge.Expression
) -> sge.Expression:
    # TODO: support addop for string dtype.
    return sge.Add(this=left, expression=right)
