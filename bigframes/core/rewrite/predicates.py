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
from typing import Generator, Sequence, Tuple

import bigframes.core.expression
import bigframes.core.guid
import bigframes.core.identifiers
import bigframes.core.join_def
import bigframes.core.nodes
import bigframes.core.window_spec
import bigframes.operations.aggregations


def apply_mask_expr(
    expr: bigframes.core.expression.Expression,
    mask: bigframes.core.expression.Expression,
) -> bigframes.core.expression.Expression:
    return bigframes.operations.where_op.as_expr(
        expr, mask, bigframes.core.expression.const(None)
    )


def merge_predicates(
    predicates: Sequence[bigframes.core.expression.Expression],
) -> bigframes.core.expression.Expression:
    if len(predicates) == 0:
        raise ValueError("Must provide at least one predicate to merge")
    return functools.reduce(bigframes.operations.and_op.as_expr, predicates)


def decompose_conjunction(
    expr: bigframes.core.expression.Expression,
) -> Generator[bigframes.core.expression.Expression, None, None]:
    if isinstance(expr, bigframes.core.expression.OpExpression) and isinstance(
        expr.op, type(bigframes.operations.and_op)
    ):
        yield from decompose_conjunction(expr.inputs[0])
        yield from decompose_conjunction(expr.inputs[1])
    else:
        yield expr


# TODO: Support more node types.
@functools.singledispatch
def mask_node(
    node: bigframes.core.nodes.BigFrameNode,
    mask: Sequence[bigframes.core.expression.Expression],
) -> bigframes.core.nodes.BigFrameNode:
    raise ValueError(f"Unexpected node: {node}")


@mask_node.register
def _(
    node: bigframes.core.nodes.ProjectionNode,
    mask: Sequence[bigframes.core.expression.Expression],
) -> bigframes.core.nodes.BigFrameNode:
    if not mask:
        return node
    mask_expr = merge_predicates(mask)
    new_assignment = tuple(
        (apply_mask_expr(ex, mask_expr), name) for ex, name in node.assignments
    )
    return bigframes.core.nodes.ProjectionNode(node.child, new_assignment)


@mask_node.register
def _(
    node: bigframes.core.nodes.WindowOpNode,
    mask: Sequence[bigframes.core.expression.Expression],
) -> bigframes.core.nodes.BigFrameNode:
    if not mask:
        return node
    mask_expr = merge_predicates(mask)
    grouping_keys = node.window_spec.grouping_keys
    new_id = bigframes.core.identifiers.ColumnId.unique()
    new_child = bigframes.core.nodes.ProjectionNode(
        node.child, assignments=((mask_expr, new_id),)
    )
    new_window_spec = dataclasses.replace(
        node.window_spec,
        grouping_keys=(*grouping_keys, bigframes.core.expression.DerefOp(new_id)),
    )
    return dataclasses.replace(node, child=new_child, window_spec=new_window_spec)


def relative_predicates(
    left_predicates: Sequence[bigframes.core.expression.Expression],
    right_predicates: Sequence[bigframes.core.expression.Expression],
) -> Tuple[
    Sequence[bigframes.core.expression.Expression],
    Sequence[bigframes.core.expression.Expression],
]:
    left_relative = tuple(
        pred for pred in left_predicates if pred not in right_predicates
    )
    right_relative = tuple(
        pred for pred in right_predicates if pred not in left_predicates
    )
    return left_relative, right_relative
