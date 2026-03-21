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

import itertools
from typing import Optional

import pyarrow as pa

from bigframes.core import agg_expressions, bigframe_node, expression, nodes
import bigframes.operations
from bigframes.operations import generic_ops, numeric_ops
from bigframes.session import executor, semi_executor

_COMPATIBLE_NODES = (
    nodes.ReadLocalNode,
    nodes.SelectionNode,
    nodes.ProjectionNode,
    nodes.FilterNode,
)

_COMPATIBLE_SCALAR_OPS = (
    generic_ops.IsNullOp,
    generic_ops.NotNullOp,
    generic_ops.CoalesceOp,
    generic_ops.WhereOp,
    numeric_ops.AddOp,
    numeric_ops.SubOp,
    numeric_ops.MulOp,
    numeric_ops.DivOp,
    numeric_ops.AbsOp,
    numeric_ops.FloorOp,
    numeric_ops.CeilOp,
    numeric_ops.SqrtOp,
    numeric_ops.ExpOp,
    numeric_ops.LnOp,
    numeric_ops.Log10Op,
)

_COMPATIBLE_AGG_OPS = ()


def _get_expr_ops(expr: expression.Expression) -> set[bigframes.operations.ScalarOp]:
    if isinstance(expr, expression.OpExpression):
        return set(itertools.chain.from_iterable(map(_get_expr_ops, expr.children)))
    return set()


def _is_node_datafusion_executable(node: nodes.BigFrameNode):
    if not isinstance(node, _COMPATIBLE_NODES):
        return False
    for expr in node._node_expressions:
        if isinstance(expr, agg_expressions.Aggregation):
            if not type(expr.op) in _COMPATIBLE_AGG_OPS:
                return False
        if isinstance(expr, expression.Expression):
            if not set(map(type, _get_expr_ops(expr))).issubset(_COMPATIBLE_SCALAR_OPS):
                return False
    return True


class DataFusionExecutor(semi_executor.SemiExecutor):
    def __init__(self):
        from bigframes.core.compile.datafusion import DataFusionCompiler

        self._compiler = DataFusionCompiler()

    def execute(
        self,
        plan: bigframe_node.BigFrameNode,
        ordered: bool,
        peek: Optional[int] = None,
    ) -> Optional[executor.ExecuteResult]:
        if not self._can_execute(plan):
            return None
        # Note: Ignoring ordered flag for now, similar to Polars executor
        try:
            df = self._compiler.compile(plan)
        except Exception:
            return None

        if peek is not None:
            df = df.limit(peek)

        batches = df.collect()
        pa_table = pa.Table.from_batches(batches)
        return executor.LocalExecuteResult(
            data=pa_table,
            bf_schema=plan.schema,
        )

    def _can_execute(self, plan: bigframe_node.BigFrameNode):
        return all(_is_node_datafusion_executable(node) for node in plan.unique_nodes())
