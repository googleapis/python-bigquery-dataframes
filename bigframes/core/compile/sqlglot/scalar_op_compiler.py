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

import dataclasses
import functools
import io
import itertools
import typing
from typing import cast, Sequence, Tuple, TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import sqlglot as sg
import sqlglot.expressions as sge

import bigframes.core
import bigframes.core.compile.sqlglot.sqlglot_types as sgt
import bigframes.core.expression as ex
import bigframes.core.guid as guid
import bigframes.core.nodes as nodes
import bigframes.core.rewrite
import bigframes.dtypes as dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops


@dataclasses.dataclass(frozen=True)
class SQLGlotScalarOpCompiler:
    """Scalar Op Compiler for converting BigFrames scalar op expressions to SQLGlot
    expressions."""

    # Whether to always quote identifiers.
    quoted: bool = True

    @functools.singledispatchmethod
    def compile_expression(self, expr: ex.Expression):
        raise NotImplementedError(f"Cannot compile expression: {expr}")

    @compile_expression.register
    def compile_deref_op(self, expr: ex.DerefOp) -> sge.Expression:
        return sge.ColumnDef(this=sge.to_identifier(expr.id.sql, quoted=self.quoted))

    @compile_expression.register
    def compile_scalar_constant_expression(
        self, expr: ex.ScalarConstantExpression
    ) -> sge.Expression:
        return sge.Literal(this=expr.value, is_string=dtypes.is_string_like(expr.dtype))

    @compile_expression.register
    def compile_op_expression(self, expr: ex.OpExpression) -> sge.Expression:
        op = expr.op
        # TODO: This can block non-recursively compiler.
        args = tuple(map(self.compile_expression, expr.inputs))

        op_name = expr.op.__class__.__name__
        method_name = f"compile_{op_name}"
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(f"Cannot compile operator {method_name}")

        if isinstance(op, ops.UnaryOp):
            return method(op, args[0])
        elif isinstance(op, ops.BinaryOp):
            return method(op, args[0], args[1])
        elif isinstance(op, ops.TernaryOp):
            return method(op, args[0], args[1], args[2])
        elif isinstance(op, ops.NaryOp):
            return method(op, *args)
        else:
            raise NotImplementedError(f"Cannot compile operator {method_name}")

    # TODO: add parenthesize for operators
    def compile_AddOp(
        self, op: ops.AddOp, left: sge.Expression, right: sge.Expression
    ) -> sge.Expression:
        return sge.Add(this=left, expression=right)

    # TODO: check why it's not `GeOp`
    def compile_ge(
        self, op, left: sge.Expression, right: sge.Expression
    ) -> sge.Expression:
        return sge.GTE(this=left, expression=right)
