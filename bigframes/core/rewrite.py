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

import functools
import itertools
from typing import (
    Callable,
    Generator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import bigframes.core.expression as scalar_exprs
import bigframes.core.join_def as join_defs
import bigframes.core.nodes as nodes
import bigframes.core.ordering as order
import bigframes.operations as ops

Selection = Tuple[Tuple[scalar_exprs.Expression, str], ...]


# These node types have certain properties that are required.
# 1. column ids created must be unique
# 2. other than selectionnode (which is rewritten), all nodes must by constructive
REWRITABLE_NODE_TYPES = (
    nodes.SelectionNode,
    nodes.ProjectionNode,
    nodes.FilterNode,
    nodes.ReversedNode,
    nodes.OrderByNode,
    nodes.ReprojectOpNode,
    # TODO: Support WindowOpNode
    # nodes.WindowOpNode,
)


def join_as_projection(
    l_node: nodes.BigFrameNode,
    r_node: nodes.BigFrameNode,
    join_keys: Tuple[join_defs.CoalescedColumnMapping, ...],
    mappings: Tuple[join_defs.JoinColumnMapping, ...],
    how: join_defs.JoinType,
) -> Optional[nodes.BigFrameNode]:
    if how == "cross":
        # Cross join requires true join
        return None

    # Prerequisities
    # 1. There is a common child node (self-join)
    # 2. The join condition can assumed to be considered to be a row-wise join
    common_node = common_selection_root(l_node, r_node)
    if common_node is None:
        return None

    # Simplify join type in the absence of filtering
    # Simplest case is inner join, can reduce to this if there are no filters on "inner side"
    l_filtered = span_has_filter(l_node, common_node)
    r_filtered = span_has_filter(r_node, common_node)
    # warning: cannot simply convert right join to inner, as ordering is different
    if not r_filtered and how == "left":
        how = "inner"
    if how == "outer":
        if not r_filtered and not l_filtered:
            how = "inner"
        if not l_filtered:
            how = "left"

    # these cases are simpler, can simply use outer side to provide join keys
    if how != "outer":
        result_node = rewrite_join(l_node, r_node, how, common_root=common_node)
        # can ignore join side, since assumption that new ids are unique
        if how != "right":
            join_keys_selection = (
                (ccm.left_source_id, ccm.destination_id) for ccm in join_keys
            )
        else:  # right join, use righ side as join keys
            join_keys_selection = (
                (ccm.right_source_id, ccm.destination_id) for ccm in join_keys
            )
        mappings_selection = ((jcm.source_id, jcm.destination_id) for jcm in mappings)
        w_mappings = nodes.SelectionNode(
            result_node, (*join_keys_selection, *mappings_selection)
        )
        return w_mappings
    else:  # cursed outer join special case
        # TODO: Complete the implementation before submitting
        result_node = rewrite_join(l_node, r_node, how, common_root=common_node)
        # can ignore join side, since assumption that new ids are unique
        join_keys_selection = (
            (ccm.left_source_id, ccm.destination_id) for ccm in join_keys
        )
        mappings_selection = ((jcm.source_id, jcm.destination_id) for jcm in mappings)
        w_mappings = nodes.SelectionNode(
            result_node, (*join_keys_selection, *mappings_selection)
        )
        return w_mappings


def span_has_filter(root: nodes.BigFrameNode, stop: nodes.BigFrameNode) -> bool:
    if root == stop:
        return False
    elif isinstance(root, nodes.FilterNode):
        return True
    assert isinstance(root, nodes.UnaryNode)
    return span_has_filter(root.child, stop)


def merge_predicates(
    predicates: Sequence[scalar_exprs.Expression],
) -> scalar_exprs.Expression:
    if len(predicates) == 0:
        raise ValueError("Must provide at least one predicate to merge")
    return functools.reduce(ops.and_op.as_expr, predicates)


def decompose_conjunction(
    expr: scalar_exprs.Expression,
) -> Generator[scalar_exprs.Expression, None, None]:
    if isinstance(expr, scalar_exprs.OpExpression) and isinstance(
        expr.op, type(ops.and_op)
    ):
        yield from decompose_conjunction(expr.inputs[0])
        yield from decompose_conjunction(expr.inputs[1])
    else:
        yield expr


def or_predicates(
    left: Tuple[scalar_exprs.Expression, ...],
    right: Tuple[scalar_exprs.Expression, ...],
) -> Optional[scalar_exprs.Expression]:
    if (len(left) == 0) or (len(right) == 0):
        return None
    return ops.or_op.as_expr(merge_predicates(left), merge_predicates(right))


def common_selection_root(
    l_tree: nodes.BigFrameNode, r_tree: nodes.BigFrameNode
) -> Optional[nodes.BigFrameNode]:
    """Find common subtree between join subtrees"""
    l_node = l_tree
    l_nodes: set[nodes.BigFrameNode] = set()
    while isinstance(l_node, REWRITABLE_NODE_TYPES):
        l_nodes.add(l_node)
        l_node = l_node.child
    l_nodes.add(l_node)

    r_node = r_tree
    while isinstance(r_node, REWRITABLE_NODE_TYPES):
        if r_node in l_nodes:
            return r_node
        r_node = r_node.child

    if r_node in l_nodes:
        return r_node
    return None


def rewrite_join(
    left: nodes.BigFrameNode,
    right: nodes.BigFrameNode,
    how: Literal["inner", "outer", "left", "right"],
    common_root: nodes.BigFrameNode,
):
    # TODO: Fast path in the absence of order_by and filters?
    def stop_cond(node):
        return node == common_root

    # All cases, pull up selections. This makes following steps a lot easier
    left, l_selection = pull_up_selections(left, stop_cond)
    right, r_selection = pull_up_selections(right, stop_cond)

    outer_join_predicate: Optional[scalar_exprs.Expression] = None

    # some combination of baking ordering, baking filters
    if how == "inner":  # just need to bake inner ordering
        # if extend to window ops, will need to pull up filters from both sides
        right, _ = bake_order_by(right, stop_cond)
    elif how == "outer":
        # Outer needs to pull up all filters
        left, left_filters = pull_up_filters(left, stop_cond)
        right, right_filters = pull_up_filters(right, stop_cond)
        right, _ = bake_order_by(right, stop_cond)
        outer_join_predicate = or_predicates(left_filters, right_filters)
    elif how == "left":
        right, _ = bake_order_by(right, stop_cond)
        right, _ = pull_up_filters(right, stop_cond)
    elif how == "right":
        left, _ = bake_order_by(left, stop_cond)
        left, _ = pull_up_filters(left, stop_cond)
    else:
        raise ValueError(f"Unsupported join type: {how}")

    merged = merge_all(left, right, common_root)
    if outer_join_predicate is not None:
        merged = nodes.FilterNode(merged, outer_join_predicate)

    joined_selection = (*l_selection, *r_selection)
    return nodes.SelectionNode(merged, joined_selection)


def merge_all(
    left: nodes.BigFrameNode, right: nodes.BigFrameNode, common_root: nodes.BigFrameNode
):
    """Apply all the nodes from right to the left tree, stopping at common_root"""
    if right != common_root:
        assert isinstance(right, nodes.UnaryNode)
        return replace_child(right, merge_all(left, right.child, common_root))
    return left


def bake_order_by(
    node: nodes.BigFrameNode, stop_at: Callable[[nodes.BigFrameNode], bool]
) -> Tuple[nodes.BigFrameNode, order.RowOrdering]:
    # TODO: Support every node type. This will be used as a preprocessor.
    if stop_at(node):
        return node, order.RowOrdering()
    if isinstance(node, nodes.UnaryNode):
        child, ordering = bake_order_by(node.child, stop_at)
        if isinstance(node, nodes.OrderByNode):
            return child, ordering.with_ordering_columns(node.by)
        if isinstance(node, nodes.ReversedNode):
            return child, ordering.with_reverse()
        if isinstance(node, nodes.ProjectionNode):
            return replace_child(node, child), ordering
        if isinstance(node, nodes.FilterNode):
            return replace_child(node, child), ordering
    raise ValueError(f"Unexpected node type f{type(node)}")


# Push down or pull up????
# The goal is to keep all available variables. Also, simplifies by removing messy seleciton, reproject nodes
def pull_up_selections(
    node: nodes.BigFrameNode, stop_at: Callable[[nodes.BigFrameNode], bool]
) -> Tuple[nodes.BigFrameNode, Tuple[Tuple[str, str], ...]]:
    """Pull selection nodes out of the tree."""
    if stop_at(node):
        return node, tuple((id, id) for id in node.schema.names)

    if isinstance(node, nodes.UnaryNode):
        child, selection = pull_up_selections(node.child, stop_at)
        mapping: Mapping[str, str] = {o: i for i, o in selection}
        if isinstance(node, nodes.OrderByNode):
            ordering = tuple(ex.remap_names(mapping) for ex in node.by)
            return nodes.OrderByNode(child, ordering), selection
        if isinstance(node, nodes.ReversedNode):
            return nodes.ReversedNode(child, reversed=node.reversed), selection
        if isinstance(node, nodes.ReprojectOpNode):
            # Essentially just removes this no-op node
            return pull_up_selections(node.child, stop_at)
        if isinstance(node, nodes.SelectionNode):
            selection = tuple(
                (mapping.get(input, input), output)
                for input, output in node.input_output_pairs
            )
            return child, selection
        if isinstance(node, nodes.ProjectionNode):
            assignments = tuple(
                (ex.rename(mapping), mapping.get(id, id)) for ex, id in node.assignments
            )
            return nodes.ProjectionNode(child, assignments), tuple(
                itertools.chain(selection, ((id, id) for _, id in node.assignments))
            )
        if isinstance(node, nodes.FilterNode):
            predicate = node.predicate.rename(mapping)
            return nodes.FilterNode(child, predicate), selection
    raise ValueError(f"Unexpected node type f{type(node)}")


def pull_up_filters(
    node: nodes.BigFrameNode, stop_at: Callable[[nodes.BigFrameNode], bool]
) -> Tuple[nodes.BigFrameNode, Tuple[scalar_exprs.Expression, ...]]:
    """
    Pull predicates up. May create extra variables, which caller will need to drop.

    This will create new variables when it would otherwise drop a predicate
    """
    if stop_at(node):
        return node, ()
    if isinstance(node, nodes.OrderByNode):
        child_node, child_predicates = pull_up_filters(node.child, stop_at)
        return replace_child(node, child_node), child_predicates
    if isinstance(node, nodes.ReversedNode):
        child_node, child_predicates = pull_up_filters(node.child, stop_at)
        return replace_child(node, child_node), child_predicates
    if isinstance(node, nodes.ReprojectOpNode):
        # Reproject nodes need to be removed by another step.
        raise ValueError("Unsupported reproject node.")
    if isinstance(node, nodes.SelectionNode):
        # Selection nodes need to be removed by another step.
        raise ValueError("Unsupported selection node.")
    if isinstance(node, nodes.ProjectionNode):
        child_node, child_predicates = pull_up_filters(node.child, stop_at)
        node = replace_child(node, child_node)
        if len(child_predicates) > 0:
            mask_expr = merge_predicates(child_predicates)
            node = apply_mask(node, mask_expr)
        # when pulling up filters, need to mask projection so invalid
        # inputs don't get passed and create errors
        return node, child_predicates
    if isinstance(node, nodes.FilterNode):
        # TODO: Do we need to go mask all existing variables??
        child_node, child_predicates = pull_up_filters(node.child, stop_at)
        return (child_node, (*child_predicates, *decompose_conjunction(node.predicate)))
    else:
        raise ValueError(f"Unsupported node type f{type(node)}")


def apply_mask(
    node: nodes.ProjectionNode, mask: scalar_exprs.Expression
) -> nodes.ProjectionNode:
    new_assignment = tuple(
        (ops.where_op.as_expr(ex, mask, scalar_exprs.const(None)), name)
        for ex, name in node.assignments
    )
    return nodes.ProjectionNode(node.child, new_assignment)


T = TypeVar("T", bound=nodes.UnaryNode)


def replace_child(node: T, new_child: nodes.BigFrameNode) -> T:
    if node.child == new_child:
        return node
    return node.transform_children(lambda x: new_child)  # type: ignore
