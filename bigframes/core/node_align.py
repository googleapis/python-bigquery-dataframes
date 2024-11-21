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
from typing import Optional, Tuple

import bigframes.core.expression
import bigframes.core.guid
import bigframes.core.identifiers
import bigframes.core.join_def
import bigframes.core.nodes
import bigframes.core.window_spec
import bigframes.operations.aggregations

ADDITIVE_NODES = (
    bigframes.core.nodes.ProjectionNode,
    bigframes.core.nodes.WindowOpNode,
    bigframes.core.nodes.PromoteOffsetsNode,
)


@dataclasses.dataclass(frozen=True)
class ExpressionSpec:
    expression: bigframes.core.expression.Expression
    node: bigframes.core.nodes.BigFrameNode


def get_expression_spec(
    node: bigframes.core.nodes.BigFrameNode, id: bigframes.core.identifiers.ColumnId
) -> ExpressionSpec:
    """Normalizes column value by chaining expressions across multiple selection and projection nodes if possible"""
    expression: bigframes.core.expression.Expression = (
        bigframes.core.expression.DerefOp(id)
    )
    curr_node = node
    while True:
        if isinstance(curr_node, bigframes.core.nodes.SelectionNode):
            select_mappings = {
                col_id: ref for ref, col_id in curr_node.input_output_pairs
            }
            expression = expression.bind_refs(
                select_mappings, allow_partial_bindings=True
            )
        elif isinstance(curr_node, bigframes.core.nodes.ProjectionNode):
            proj_mappings = {col_id: expr for expr, col_id in curr_node.assignments}
            expression = expression.bind_refs(
                proj_mappings, allow_partial_bindings=True
            )
        elif isinstance(
            curr_node,
            (
                bigframes.core.nodes.WindowOpNode,
                bigframes.core.nodes.PromoteOffsetsNode,
            ),
        ):
            pass
        else:
            return ExpressionSpec(expression, curr_node)
        curr_node = curr_node.child


def _linearize_trees(
    base_tree: bigframes.core.nodes.BigFrameNode,
    append_tree: bigframes.core.nodes.BigFrameNode,
) -> bigframes.core.nodes.BigFrameNode:
    """Linearize two divergent tree who only diverge through different additive nodes."""
    assert append_tree.projection_base == base_tree.projection_base
    if append_tree == append_tree.projection_base:
        return base_tree
    else:
        assert isinstance(append_tree, ADDITIVE_NODES)
        return append_tree.replace_child(_linearize_trees(base_tree, append_tree.child))


def combine_nodes(
    l_node: bigframes.core.nodes.BigFrameNode,
    r_node: bigframes.core.nodes.BigFrameNode,
) -> bigframes.core.nodes.BigFrameNode:
    assert l_node.projection_base == r_node.projection_base
    l_node, l_selection = pull_up_selection(l_node)
    r_node, r_selection = pull_up_selection(
        r_node, rename_vars=True
    )  # Rename only right vars to avoid collisions with left vars
    combined_selection = (*l_selection, *r_selection)
    merged_node = _linearize_trees(l_node, r_node)
    return bigframes.core.nodes.SelectionNode(merged_node, combined_selection)


def join_as_projection(
    l_node: bigframes.core.nodes.BigFrameNode,
    r_node: bigframes.core.nodes.BigFrameNode,
    join_keys: Tuple[Tuple[str, str], ...],
) -> Optional[bigframes.core.nodes.BigFrameNode]:
    """Joins the two nodes"""
    if l_node.projection_base != r_node.projection_base:
        return None
    # check join key
    for l_key, r_key in join_keys:
        # Caller is block, so they still work with raw strings rather than ids
        left_id = bigframes.core.identifiers.ColumnId(l_key)
        right_id = bigframes.core.identifiers.ColumnId(r_key)
        if get_expression_spec(l_node, left_id) != get_expression_spec(
            r_node, right_id
        ):
            return None
    return combine_nodes(l_node, r_node)


def pull_up_selection(
    node: bigframes.core.nodes.BigFrameNode, rename_vars: bool = False
) -> Tuple[
    bigframes.core.nodes.BigFrameNode,
    Tuple[
        Tuple[bigframes.core.expression.DerefOp, bigframes.core.identifiers.ColumnId],
        ...,
    ],
]:
    """Remove all selection nodes above the base node. Returns stripped tree."""
    if node == node.projection_base:  # base case
        return node, tuple(
            (bigframes.core.expression.DerefOp(field.id), field.id)
            for field in node.fields
        )
    assert isinstance(node, (bigframes.core.nodes.SelectionNode, *ADDITIVE_NODES))
    child_node, child_selections = pull_up_selection(node.child)
    mapping = {out: ref.id for ref, out in child_selections}
    if isinstance(node, ADDITIVE_NODES):
        new_node: bigframes.core.nodes.BigFrameNode = node.replace_child(child_node)
        new_node = new_node.remap_refs(mapping)
        if rename_vars:
            var_renames = {
                field.id: bigframes.core.identifiers.ColumnId.unique()
                for field in node.added_fields
            }
            new_node = new_node.remap_vars(var_renames)
        assert isinstance(new_node, ADDITIVE_NODES)
        added_selections = (
            (bigframes.core.expression.DerefOp(field.id), field.id)
            for field in new_node.added_fields
        )
        new_selection = (*child_selections, *added_selections)
        assert all(ref.id in new_node.ids for ref, _ in new_selection)
        return new_node, new_selection
    elif isinstance(node, bigframes.core.nodes.SelectionNode):
        new_selection = tuple(
            (
                bigframes.core.expression.DerefOp(mapping[ref.id]),
                (bigframes.core.identifiers.ColumnId.unique() if rename_vars else out),
            )
            for ref, out in node.input_output_pairs
        )
        if not (all(ref.id in child_node.ids for ref, _ in new_selection)):
            raise ValueError()
        return child_node, new_selection
    raise ValueError(f"Couldn't pull up select from node: {node}")
