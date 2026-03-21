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
from bigframes.operations import generic_ops


@df_compiler.register_op(generic_ops.IsNullOp)
def isnull_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: generic_ops.IsNullOp,  # type: ignore
    input: datafusion.Expr,
) -> datafusion.Expr:
    return input.is_null()


@df_compiler.register_op(generic_ops.NotNullOp)
def notnull_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: generic_ops.NotNullOp,  # type: ignore
    input: datafusion.Expr,
) -> datafusion.Expr:
    return input.is_not_null()


@df_compiler.register_op(generic_ops.CoalesceOp)
def coalesce_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: generic_ops.CoalesceOp,  # type: ignore
    *args: datafusion.Expr,
) -> datafusion.Expr:
    return f.coalesce(*args)


@df_compiler.register_op(generic_ops.WhereOp)
def where_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: generic_ops.WhereOp,  # type: ignore
    cond: datafusion.Expr,
    value: datafusion.Expr,
    other: datafusion.Expr,
) -> datafusion.Expr:
    return f.when(cond, value).otherwise(other)
