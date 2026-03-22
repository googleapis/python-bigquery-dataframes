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

"""
BigFrames -> DataFusion compilation for the operations in bigframes.operations.comparison_ops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bigframes.core.compile.datafusion.compiler as df_compiler
from bigframes.operations import comparison_ops

if TYPE_CHECKING:
    import datafusion


@df_compiler.register_op(comparison_ops.EqOp)
def eq_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: comparison_ops.EqOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left == right


@df_compiler.register_op(comparison_ops.NeOp)
def ne_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: comparison_ops.NeOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left != right


@df_compiler.register_op(comparison_ops.LtOp)
def lt_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: comparison_ops.LtOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left < right


@df_compiler.register_op(comparison_ops.LeOp)
def le_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: comparison_ops.LeOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left <= right


@df_compiler.register_op(comparison_ops.GtOp)
def gt_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: comparison_ops.GtOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left > right


@df_compiler.register_op(comparison_ops.GeOp)
def ge_op_impl(
    compiler: df_compiler.DataFusionExpressionCompiler,
    op: comparison_ops.GeOp,  # type: ignore
    left: datafusion.Expr,
    right: datafusion.Expr,
) -> datafusion.Expr:
    return left >= right
