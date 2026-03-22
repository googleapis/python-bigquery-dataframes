# Copyright 2026 Google LLC
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

import datafusion
import datafusion.functions as f

import bigframes.core.compile.datafusion.compiler as df_compiler
from bigframes.operations import numeric_ops


@df_compiler.register_op(numeric_ops.AddOp)
def add_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.AddOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left + right


@df_compiler.register_op(numeric_ops.SubOp)
def sub_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.SubOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left - right


@df_compiler.register_op(numeric_ops.MulOp)
def mul_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.MulOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left * right


@df_compiler.register_op(numeric_ops.DivOp)
def div_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.DivOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left / right


@df_compiler.register_op(numeric_ops.AbsOp)
def abs_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.AbsOp,  # type: ignore
    input: datafusion.Expr,
) -> datafusion.Expr:
    return f.abs(input)


@df_compiler.register_op(numeric_ops.FloorOp)
def floor_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.FloorOp,  # type: ignore
    input: datafusion.Expr,
) -> datafusion.Expr:
    return f.floor(input)


@df_compiler.register_op(numeric_ops.CeilOp)
def ceil_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.CeilOp,  # type: ignore
    input: datafusion.Expr,
) -> datafusion.Expr:
    return f.ceil(input)


@df_compiler.register_op(numeric_ops.SqrtOp)
def sqrt_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.SqrtOp,  # type: ignore
    input: datafusion.Expr,
) -> datafusion.Expr:
    return f.sqrt(input)


@df_compiler.register_op(numeric_ops.ExpOp)
def exp_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.ExpOp,  # type: ignore
    input: datafusion.Expr,
) -> datafusion.Expr:
    return f.exp(input)


@df_compiler.register_op(numeric_ops.LnOp)
def ln_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.LnOp,  # type: ignore
    input: datafusion.Expr,
) -> datafusion.Expr:
    return f.ln(input)


@df_compiler.register_op(numeric_ops.Log10Op)
def log10_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: numeric_ops.Log10Op,  # type: ignore
    input: datafusion.Expr,
) -> datafusion.Expr:
    return f.log10(input)
