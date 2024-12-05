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
from typing import cast, Generator, Literal, Optional, Sequence, Tuple

import bigframes.core.expression
import bigframes.core.guid
import bigframes.core.identifiers
import bigframes.core.join_def
import bigframes.core.nodes
import bigframes.core.window_spec
import bigframes.operations
import bigframes.operations.aggregations

Predicates = Sequence[bigframes.core.expression.Expression]
Selection = Tuple[
    Tuple[bigframes.core.expression.DerefOp, bigframes.core.identifiers.ColumnId],
    ...,
]

# Additive nodes leave existing columns completely intact, and only add new columns to the end
ADDITIVE_NODES = (
    bigframes.core.nodes.ProjectionNode,
    bigframes.core.nodes.WindowOpNode,
    bigframes.core.nodes.PromoteOffsetsNode,
)
ALIGNABLE_NODES = (
    *ADDITIVE_NODES,
    bigframes.core.nodes.SelectionNode,
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
                bigframes.core.nodes.FilterNode,
            ),
        ):
            # we don't yet have a way of normalizing window ops into a ExpressionSpec, which only
            # handles normalizing scalar expressions at the moment.
            pass
        else:
            return ExpressionSpec(expression, curr_node)
        curr_node = curr_node.child


def can_row_join(
    l_node: bigframes.core.nodes.BigFrameNode,
    r_node: bigframes.core.nodes.BigFrameNode,
    join_keys: Tuple[Tuple[str, str], ...],
):
    if l_node.projection_base != r_node.projection_base:
        return False

    # check join keys are equivalent by normalizing the expressions as much as posisble
    # instead of just comparing ids
    for l_key, r_key in join_keys:
        # Caller is block, so they still work with raw strings rather than ids
        left_id = bigframes.core.identifiers.ColumnId(l_key)
        right_id = bigframes.core.identifiers.ColumnId(r_key)
        if get_expression_spec(l_node, left_id) != get_expression_spec(
            r_node, right_id
        ):
            return False
    return True


def try_join_as_projection(
    l_node: bigframes.core.nodes.BigFrameNode,
    r_node: bigframes.core.nodes.BigFrameNode,
    join_keys: Tuple[Tuple[str, str], ...],
) -> Optional[bigframes.core.nodes.BigFrameNode]:
    """Joins the two nodes"""
    if not can_row_join(l_node, r_node, join_keys):
        return None

    divergent_node = first_common_descendant(l_node, r_node)

    l_node, l_selection = pull_up_selection(l_node, stop=divergent_node)
    r_node, r_selection = pull_up_selection(
        r_node, stop=divergent_node, rename_vars=True
    )

    def _linearize_trees(
        base_tree: bigframes.core.nodes.BigFrameNode,
        append_tree: bigframes.core.nodes.BigFrameNode,
    ) -> bigframes.core.nodes.BigFrameNode:
        """Linearize two divergent tree who only diverge through different additive nodes."""
        assert append_tree.projection_base == base_tree.projection_base
        # base case: append tree does not have any additive nodes to linearize
        if append_tree == divergent_node:
            return base_tree
        else:
            assert isinstance(append_tree, ADDITIVE_NODES)
            return append_tree.replace_child(
                _linearize_trees(base_tree, append_tree.child)
            )

    return bigframes.core.nodes.SelectionNode(
        _linearize_trees(l_node, r_node), (*l_selection, *r_selection)
    )


def convert_relational_join(
    node: bigframes.core.nodes.JoinNode,
) -> Optional[bigframes.core.nodes.BigFrameNode]:
    """
    Converts a true relational join into a projection.
    This is a compile-time rewrite only - and must preserve value exactly.
    """
    l_node = node.left_child
    r_node = node.right_child
    join_keys = node.conditions
    how = node.type

    if how not in ("inner", "left", "right", "outer"):
        return None

    # TODO: Include filter
    if l_node.projection_base != r_node.projection_base:
        return None

    # TODO: Check (joint) uniqueness of join keys

    divergent_node = first_common_descendant(l_node, r_node)

    # check join keys are equivalent by normalizing the expressions as much as posisble
    # instead of just comparing ids
    for l_key, r_key in join_keys:
        # Caller is block, so they still work with raw strings rather than ids
        if get_expression_spec(l_node, l_key.id) != get_expression_spec(
            r_node, r_key.id
        ):
            return None

    l_node, l_selection = pull_up_selection(l_node, stop=divergent_node)
    r_node, r_selection = pull_up_selection(
        r_node, stop=divergent_node, rename_vars=True
    )
    l_node, l_filters = pull_up_filters(l_node, stop=divergent_node)
    r_node, r_filters = pull_up_filters(r_node, stop=divergent_node)

    def _linearize_trees(
        base_tree: bigframes.core.nodes.BigFrameNode,
        append_tree: bigframes.core.nodes.BigFrameNode,
    ) -> bigframes.core.nodes.BigFrameNode:
        """Linearize two divergent tree who only diverge through different additive nodes."""
        assert append_tree.projection_base == base_tree.projection_base
        # base case: append tree does not have any additive nodes to linearize
        if append_tree == divergent_node:
            return base_tree
        else:
            assert isinstance(append_tree, ADDITIVE_NODES)
            return append_tree.replace_child(
                _linearize_trees(base_tree, append_tree.child)
            )

    merged_node = _linearize_trees(l_node, r_node)

    # add back all the destructive operators we pulled out according to join type
    return mask_filter_select(
        merged_node, l_selection, l_filters, r_selection, r_filters, how=how  # type: ignore
    )


def mask_filter_select(
    raw_node: bigframes.core.nodes.BigFrameNode,
    l_selection: Selection,
    l_mask: Predicates,
    r_selection: Selection,
    r_mask: Predicates,
    how: Literal["inner", "outer", "left", "right"],
) -> bigframes.core.nodes.BigFrameNode:
    node = raw_node
    # TODO: don't apply unnecessary row filters!!!!
    row_filter: Optional[bigframes.core.expression.Expression] = None
    if how == "outer" and (l_mask and r_mask):
        row_filter = bigframes.operations.or_op.as_expr(
            merge_predicates(l_mask), merge_predicates(r_mask)
        )
    elif (how == "left") and l_mask:
        row_filter = merge_predicates(l_mask)
    elif (how == "right") and r_mask:
        row_filter = merge_predicates(r_mask)
    elif (how == "inner") and (l_mask or r_mask):
        row_filter = merge_predicates([*l_mask, *r_mask])

    if row_filter:
        node = bigframes.core.nodes.FilterNode(node, row_filter)

    if l_mask and how in ("outer", "right"):
        node = bigframes.core.nodes.ProjectionNode(
            node,
            assignments=tuple(
                (
                    apply_mask_expr(ref, merge_predicates(l_mask)),
                    bigframes.core.identifiers.ColumnId.unique(),
                )
                for ref, _ in l_selection
            ),
        )
        l_selection = tuple(
            zip(
                map(
                    lambda x: bigframes.core.expression.DerefOp(x.id),
                    node.added_fields,
                ),
                map(lambda x: x[1], l_selection),
            )
        )
    if r_mask and how in ("outer", "left"):
        node = bigframes.core.nodes.ProjectionNode(
            node,
            assignments=tuple(
                (
                    apply_mask_expr(ref, merge_predicates(r_mask)),
                    bigframes.core.identifiers.ColumnId.unique(),
                )
                for ref, _ in r_selection
            ),
        )
        r_selection = tuple(
            zip(
                map(
                    lambda x: bigframes.core.expression.DerefOp(x.id),
                    node.added_fields,
                ),
                map(lambda x: x[1], r_selection),
            )
        )

    result = bigframes.core.nodes.SelectionNode(node, (*l_selection, *r_selection))
    result.validate_tree()
    return result


def pull_up_selection(
    node: bigframes.core.nodes.BigFrameNode,
    stop: bigframes.core.nodes.BigFrameNode,
    rename_vars: bool = False,
) -> Tuple[bigframes.core.nodes.BigFrameNode, Selection]:
    """Remove all selection nodes above the base node. Returns stripped tree.

    Args:
        node (BigFrameNode):
            The node from which to pull up SelectionNode ops
        rename_vars (bool):
            If true, will rename projected columns to new unique ids.

    Returns:
        BigFrameNode, Selections
    """
    if node == stop:  # base case
        return node, tuple(
            (bigframes.core.expression.DerefOp(field.id), field.id)
            for field in node.fields
        )
    assert isinstance(
        node,
        (
            bigframes.core.nodes.SelectionNode,
            bigframes.core.nodes.FilterNode,
            *ADDITIVE_NODES,
        ),
    )
    child_node, child_selections = pull_up_selection(
        node.child, stop, rename_vars=rename_vars
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
    root: bigframes.core.nodes.BigFrameNode,
    stop: bigframes.core.nodes.BigFrameNode,
) -> Tuple[bigframes.core.nodes.BigFrameNode, Predicates]:
    """Linearize two divergent tree who only diverge through different additive nodes."""
    # base case: append tree does not have any additive nodes to linearize
    if root == stop:
        return root, ()
    elif isinstance(root, bigframes.core.nodes.FilterNode):
        this_mask = decompose_conjunction(root.predicate)
        child_result, child_mask = pull_up_filters(root.child, stop)
        return child_result, (*child_mask, *this_mask)
    elif isinstance(root, ADDITIVE_NODES):
        assert isinstance(root, ADDITIVE_NODES)
        child_result, child_mask = pull_up_filters(root.child, stop)
        return (
            mask_node(root.replace_child(child_result), child_mask),
            child_mask,
        )
    raise ValueError(f"Unexpected node: {root}")


## Predicate helpers
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


@mask_node.register
def _(
    node: bigframes.core.nodes.PromoteOffsetsNode,
    mask: Sequence[bigframes.core.expression.Expression],
) -> bigframes.core.nodes.BigFrameNode:
    if not mask:
        return node
    mask_expr = merge_predicates(mask)
    new_id = bigframes.core.identifiers.ColumnId.unique()
    new_child = bigframes.core.nodes.ProjectionNode(
        node.child, assignments=((mask_expr, new_id),)
    )
    new_window_spec = bigframes.core.WindowSpec(
        grouping_keys=(bigframes.core.expression.DerefOp(new_id),)
    )
    return bigframes.core.nodes.WindowOpNode(
        new_child,
        column_name=bigframes.core.expression.DerefOp(new_id),
        output_name=node.col_id,
        op=bigframes.operations.aggregations.RowNumberOp(),
        window_spec=new_window_spec,
    )


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


def first_common_descendant(
    left: bigframes.core.nodes.BigFrameNode, right: bigframes.core.nodes.BigFrameNode
) -> bigframes.core.nodes.BigFrameNode:
    l_depth = projection_depth(left)
    r_depth = projection_depth(right)
    diff = r_depth - l_depth

    while left != right:
        if diff <= 0:
            left = cast(bigframes.core.nodes.UnaryNode, left).child
            diff += 1
        if diff > 0:
            right = cast(bigframes.core.nodes.UnaryNode, right).child
            diff -= 1
    return left


def projection_depth(node: bigframes.core.nodes.BigFrameNode) -> int:
    if node == node.projection_base:
        return 0
    else:
        assert isinstance(node, bigframes.core.nodes.UnaryNode)
        return 1 + projection_depth(node.child)
