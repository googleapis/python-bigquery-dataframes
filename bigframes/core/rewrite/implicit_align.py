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
from typing import Optional, Tuple, Generator, Sequence, Literal

import bigframes.core.expression
import bigframes.core.guid
import bigframes.core.identifiers
import bigframes.core.join_def
import bigframes.core.nodes
import bigframes.core.window_spec
import bigframes.operations.aggregations
import bigframes.operations

import functools

# Additive nodes leave existing columns completely intact, and only add new columns to the end
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
    """Normalizes column value by chaining expressions across multiple selection and projection nodes if possible.
    This normalization helps identify whether columns are equivalent.
    """
    # TODO: While we chain expression fragments from different nodes
    # we could further normalize with constant folding and other scalar expression rewrites
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
            # we don't yet have a way of normalizing window ops into a ExpressionSpec, which only
            # handles normalizing scalar expressions at the moment.
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
    # base case: append tree does not have any additive nodes to linearize
    if append_tree == append_tree.projection_base:
        return base_tree
    else:
        assert isinstance(append_tree, ADDITIVE_NODES)
        return append_tree.replace_child(_linearize_trees(base_tree, append_tree.child))
    
def try_join_as_projection(
    l_node: bigframes.core.nodes.BigFrameNode,
    r_node: bigframes.core.nodes.BigFrameNode,
    join_keys: Tuple[Tuple[str, str], ...],
    how: Literal["inner", "outer", "left", "right"]
) -> Optional[bigframes.core.nodes.BigFrameNode]:
    """Joins the two nodes"""
    if l_node.projection_base != r_node.projection_base:
        return None
    # check join keys are equivalent by normalizing the expressions as much as posisble
    # instead of just comparing ids
    for l_key, r_key in join_keys:
        # Caller is block, so they still work with raw strings rather than ids
        left_id = bigframes.core.identifiers.ColumnId(l_key)
        right_id = bigframes.core.identifiers.ColumnId(r_key)
        if get_expression_spec(l_node, left_id) != get_expression_spec(
            r_node, right_id
        ):
            return None
        
    if how == "inner":
        return inner_join(l_node, r_node)
    elif how == "outer":
        return outer_join(l_node, r_node)
    elif how == "left":
        return left_join(l_node, r_node)
    elif how == "right":
        return right_join(l_node, r_node)    
    raise ValueError(f"Unrecognized join type: {how}")


def left_join(
    l_node: bigframes.core.nodes.BigFrameNode,
    r_node: bigframes.core.nodes.BigFrameNode,
) -> bigframes.core.nodes.BigFrameNode:
    assert l_node.projection_base == r_node.projection_base
    l_node, l_selection = pull_up_selection(l_node)
    r_node, r_selection = pull_up_selection(
        r_node, rename_vars=True
    )  # Rename only right vars to avoid collisions with left var

    r_node, r_mask = pull_up_filters(r_node)

    merged_node = _linearize_trees(l_node, r_node)
    combined_selection = (*l_selection, *r_selection)
    
    return bigframes.core.nodes.SelectionNode(merged_node, combined_selection)

def right_join(
    l_node: bigframes.core.nodes.BigFrameNode,
    r_node: bigframes.core.nodes.BigFrameNode,
) -> bigframes.core.nodes.BigFrameNode:
    assert l_node.projection_base == r_node.projection_base
    l_node, l_selection = pull_up_selection(l_node)
    r_node, r_selection = pull_up_selection(
        r_node, rename_vars=True
    )  # Rename only right vars to avoid collisions with left var

    merged_node = _linearize_trees(l_node, r_node)
    combined_selection = (*l_selection, *r_selection)
    
    return bigframes.core.nodes.SelectionNode(merged_node, combined_selection)

def inner_join(
    l_node: bigframes.core.nodes.BigFrameNode,
    r_node: bigframes.core.nodes.BigFrameNode,
) -> bigframes.core.nodes.BigFrameNode:
    assert l_node.projection_base == r_node.projection_base
    l_node, l_selection = pull_up_selection(l_node)
    r_node, r_selection = pull_up_selection(
        r_node, rename_vars=True
    )  # Rename only right vars to avoid collisions with left var

    merged_node = _linearize_trees(l_node, r_node)
    combined_selection = (*l_selection, *r_selection)
    
    return bigframes.core.nodes.SelectionNode(merged_node, combined_selection)

def outer_join(
    l_node: bigframes.core.nodes.BigFrameNode,
    r_node: bigframes.core.nodes.BigFrameNode,
) -> bigframes.core.nodes.BigFrameNode:
    assert l_node.projection_base == r_node.projection_base
    l_node, l_selection = pull_up_selection(l_node)
    r_node, r_selection = pull_up_selection(
        r_node, rename_vars=True
    )  # Rename only right vars to avoid collisions with left var

    merged_node = _linearize_trees(l_node, r_node)
    combined_selection = (*l_selection, *r_selection)
    
    return bigframes.core.nodes.SelectionNode(merged_node, combined_selection)

def pull_up_selection(
    node: bigframes.core.nodes.BigFrameNode, rename_vars: bool = False
) -> Tuple[
    bigframes.core.nodes.BigFrameNode,
    Tuple[
        Tuple[bigframes.core.expression.DerefOp, bigframes.core.identifiers.ColumnId],
        ...,
    ],
]:
    """Remove all selection nodes above the base node. Returns stripped tree.

    Args:
        node (BigFrameNode):
            The node from which to pull up SelectionNode ops
        rename_vars (bool):
            If true, will rename projected columns to new unique ids.

    Returns:
        BigFrameNode, Selections
    """
    if node == node.projection_base:  # base case
        return node, tuple(
            (bigframes.core.expression.DerefOp(field.id), field.id)
            for field in node.fields
        )
    assert isinstance(node, (bigframes.core.nodes.SelectionNode, *ADDITIVE_NODES))
    child_node, child_selections = pull_up_selection(
        node.child, rename_vars=rename_vars
    )
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
        else:
            var_renames = {}
        assert isinstance(new_node, ADDITIVE_NODES)
        added_selections = (
            (
                bigframes.core.expression.DerefOp(var_renames.get(field.id, field.id)),
                field.id,
            )
            for field in node.added_fields
        )
        new_selection = (*child_selections, *added_selections)
        return new_node, new_selection
    elif isinstance(node, bigframes.core.nodes.FilterNode):
        return node.replace_child(child_node).remap_refs(mapping), child_selections
    elif isinstance(node, bigframes.core.nodes.SelectionNode):
        new_selection = tuple(
            (
                bigframes.core.expression.DerefOp(mapping[ref.id]),
                out,
            )
            for ref, out in node.input_output_pairs
        )
        return child_node, new_selection
    raise ValueError(f"Couldn't pull up select from node: {node}")


def pull_up_filters(
    node: bigframes.core.nodes.BigFrameNode
) -> Tuple[bigframes.core.nodes.BigFrameNode, Tuple[bigframes.core.expression.Expression, ...]]:
    """
    Pull predicates up. May create extra variables, which caller will need to drop.
    """
    if node == node.projection_base:  # base case
        return node, tuple(
            (bigframes.core.expression.DerefOp(field.id), field.id)
            for field in node.fields
        )
    if isinstance(node, bigframes.core.nodes.ProjectionNode):
        child_node, child_predicates = pull_up_filters(node.child)
        node = node.replace_child(child_node)
        if len(child_predicates) > 0:
            mask_expr = merge_predicates(child_predicates)
            node = mask_projection(node, mask_expr)
        # when pulling up filters, need to mask projection so invalid
        # inputs don't get passed and create errors
        return node, child_predicates
    if isinstance(node, bigframes.core.nodes.FilterNode):
        # TODO: Do we need to go mask all existing variables??
        child_node, child_predicates = pull_up_filters(node.child)
        return (child_node, (*child_predicates, *decompose_conjunction(node.predicate)))
    else:
        raise ValueError(f"Unsupported node type f{type(node)}")


## Predicate helpers
def mask_projection(
    node: bigframes.core.nodes.ProjectionNode, mask: bigframes.core.expression.Expression
) -> bigframes.core.nodes.ProjectionNode:
    new_assignment = tuple(
        (bigframes.operations.where_op.as_expr(ex, mask, bigframes.core.expression.const(None)), name)
        for ex, name in node.assignments
    )
    return bigframes.core.nodes.ProjectionNode(node.child, new_assignment)

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
