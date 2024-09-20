# Copyright 2024 Google LLC
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

import pandas as pd
import polars as pl

import bigframes.core
import bigframes.core.expression as ex
import bigframes.core.nodes as nodes
import bigframes.operations as ops

SUPPORTED_NODES = (
    nodes.ReadLocalNode,
    nodes.SelectionNode,
    nodes.ProjectionNode,
    nodes.OrderByNode,
    nodes.ReversedNode,
    nodes.ReprojectOpNode,
    nodes.FilterNode,
    nodes.RowCountNode,
)


@dataclasses.dataclass(frozen=True)
class PolarsExpressionCompiler:
    @functools.singledispatchmethod
    def compile_expression(self, expression: ex.Expression) -> pl.Expr:
        ...

    @compile_expression.register
    def _(
        self,
        expression: ex.ScalarConstantExpression,
    ) -> pl.Expr:
        return pl.lit(expression.value)

    @compile_expression.register
    def _(
        self,
        expression: ex.UnboundVariableExpression,
    ) -> pl.Expr:
        return pl.col(expression.id)

    @compile_expression.register
    def _(
        self,
        expression: ex.OpExpression,
    ) -> pl.Expr:
        # TODO: Complete the implementation
        op = expression.op
        args = tuple(map(self.compile_expression, expression.inputs))
        if isinstance(op, ops.invert_op.__class__):
            return args[0].neg()
        if isinstance(op, ops.add_op.__class__):
            return args[0] + args[1]
        if isinstance(op, ops.ge_op.__class__):
            return args[0] >= args[1]
        raise NotImplementedError(f"Polars compiler hasn't implemented {op}")


@dataclasses.dataclass(frozen=True)
class PolarsLocalExecutor:
    """
    A simple local executor for a subset of node types.
    """

    expr_compiler = PolarsExpressionCompiler()

    # TODO: Support more node types
    # TODO: Use lazy frame instead?
    def can_execute(self, node: nodes.BigFrameNode) -> bool:
        if not isinstance(node, SUPPORTED_NODES):
            return False
        return all(map(self.can_execute, node.child_nodes))

    def execute_local(self, array_value: bigframes.core.ArrayValue) -> pd.DataFrame:
        return self.execute_node(array_value.node).collect().to_pandas()

    def execute_node(self, node: nodes.BigFrameNode) -> pl.LazyFrame:
        """Compile node into CompileArrayValue. Caches result."""
        return self._execute_node(node)

    @functools.singledispatchmethod
    def _execute_node(self, node: nodes.BigFrameNode) -> pl.DataFrame:
        """Defines transformation but isn't cached, always use compile_node instead"""
        raise ValueError(f"Can't compile unrecognized node: {node}")

    @_execute_node.register
    def compile_readlocal(self, node: nodes.ReadLocalNode):
        return pl.read_ipc(node.feather_bytes).lazy()

    @_execute_node.register
    def compile_filter(self, node: nodes.FilterNode):
        return self.execute_node(node.child).filter(
            self.expr_compiler.compile_expression(node.predicate)
        )

    @_execute_node.register
    def compile_orderby(self, node: nodes.OrderByNode):
        frame = self.execute_node(node.child)
        for by in node.by:
            frame = frame.sort(
                self.expr_compiler.compile_expression(by.scalar_expression),
                descending=not by.direction.is_ascending,
                nulls_last=by.na_last,
                maintain_order=True,
            )
        return frame

    @_execute_node.register
    def compile_reversed(self, node: nodes.ReversedNode):
        return self.execute_node(node.child).reverse()

    @_execute_node.register
    def compile_selection(self, node: nodes.SelectionNode):
        return self.execute_node(node.child).select(
            **{new: orig for orig, new in node.input_output_pairs}
        )

    @_execute_node.register
    def compile_projection(self, node: nodes.ProjectionNode):
        new_cols = [
            self.expr_compiler.compile_expression(ex).alias(name)
            for ex, name in node.assignments
        ]
        return self.execute_node(node.child).with_columns(new_cols)

    @_execute_node.register
    def compile_rowcount(self, node: nodes.RowCountNode):
        rows = self.execute_node(node.child).count()[0]
        return pl.DataFrame({"count": [rows]})

    @_execute_node.register
    def compile_reproject(self, node: nodes.ReprojectOpNode):
        # NOOP
        return self.execute_node(node.child)
