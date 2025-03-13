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
import dataclasses
import functools
from typing import AbstractSet

from bigframes.core import identifiers
import bigframes.core.nodes


def column_pruning(
    root: bigframes.core.nodes.BigFrameNode,
) -> bigframes.core.nodes.BigFrameNode:
    return bigframes.core.nodes.top_down(root, prune_columns)


def to_fixed(max_iterations: int = 100):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            previous_result = None
            current_result = func(*args, **kwargs)
            attempts = 1

            while attempts < max_iterations:
                if current_result == previous_result:
                    return current_result
                previous_result = current_result
                current_result = func(current_result)
                attempts += 1

            return current_result

        return wrapper

    return decorator


@to_fixed(max_iterations=100)
def prune_columns(node: bigframes.core.nodes.BigFrameNode):
    if isinstance(node, bigframes.core.nodes.SelectionNode):
        result = prune_selection_child(node)
    elif isinstance(node, bigframes.core.nodes.AggregateNode):
        result = node.replace_child(prune_node(node.child, node.consumed_ids))
    elif isinstance(node, bigframes.core.nodes.InNode):
        result = dataclasses.replace(
            node,
            right_child=prune_node(node.right_child, frozenset([node.right_col.id])),
        )
    else:
        result = node
    return result


def prune_selection_child(
    selection: bigframes.core.nodes.SelectionNode,
) -> bigframes.core.nodes.BigFrameNode:
    child = selection.child

    # Important to check this first
    if list(selection.ids) == list(child.ids):
        return child

    if isinstance(child, bigframes.core.nodes.SelectionNode):
        return selection.remap_refs(
            {id: ref.id for ref, id in child.input_output_pairs}
        ).replace_child(child.child)
    elif isinstance(child, bigframes.core.nodes.AdditiveNode):
        if not set(field.id for field in child.added_fields) & selection.consumed_ids:
            return selection.replace_child(child.additive_base)
        needed_ids = selection.consumed_ids | child.referenced_ids
        if isinstance(child, bigframes.core.nodes.ProjectionNode):
            # Projection expressions are independent, so can be individually removed from the node
            child = dataclasses.replace(
                child,
                assignments=tuple(
                    (ex, id) for (ex, id) in child.assignments if id in needed_ids
                ),
            )
        return selection.replace_child(
            child.replace_additive_base(prune_node(child.additive_base, needed_ids))
        )
    elif isinstance(child, bigframes.core.nodes.ConcatNode):
        indices = [
            list(child.ids).index(ref.id) for ref, _ in selection.input_output_pairs
        ]
        new_children = []
        for concat_node in child.child_nodes:
            cc_ids = tuple(concat_node.ids)
            sub_selection = tuple(
                bigframes.core.nodes.AliasedRef.identity(cc_ids[i]) for i in indices
            )
            new_children.append(
                bigframes.core.nodes.SelectionNode(concat_node, sub_selection)
            )
        return bigframes.core.nodes.ConcatNode(
            children=tuple(new_children), output_ids=tuple(selection.ids)
        )
    # Nodes that pass through input columns
    elif isinstance(
        child,
        (
            bigframes.core.nodes.RandomSampleNode,
            bigframes.core.nodes.ReversedNode,
            bigframes.core.nodes.OrderByNode,
            bigframes.core.nodes.FilterNode,
            bigframes.core.nodes.SliceNode,
            bigframes.core.nodes.JoinNode,
            bigframes.core.nodes.ExplodeNode,
        ),
    ):
        ids = selection.consumed_ids | child.referenced_ids
        return selection.replace_child(
            child.transform_children(lambda x: prune_node(x, ids))
        )
    elif isinstance(child, bigframes.core.nodes.AggregateNode):
        return selection.replace_child(prune_aggregate(child, selection.consumed_ids))
    elif isinstance(child, bigframes.core.nodes.LeafNode):
        return selection.replace_child(prune_leaf(child, selection.consumed_ids))
    return selection


def prune_node(
    node: bigframes.core.nodes.BigFrameNode,
    ids: AbstractSet[identifiers.ColumnId],
):
    # This clause is important, ensures idempotency, so can reach fixed point
    if not (set(node.ids) - ids):
        return node
    else:
        return bigframes.core.nodes.SelectionNode(
            node,
            tuple(
                bigframes.core.nodes.AliasedRef.identity(id)
                for id in node.ids
                if id in ids
            ),
        )


def prune_aggregate(
    node: bigframes.core.nodes.AggregateNode,
    used_cols: AbstractSet[identifiers.ColumnId],
) -> bigframes.core.nodes.AggregateNode:
    pruned_aggs = tuple(agg for agg in node.aggregations if agg[1] in used_cols)
    return dataclasses.replace(node, aggregations=pruned_aggs)


@functools.singledispatch
def prune_leaf(
    node: bigframes.core.nodes.BigFrameNode,
    used_cols: AbstractSet[identifiers.ColumnId],
):
    ...


@prune_leaf.register
def prune_readlocal(
    node: bigframes.core.nodes.ReadLocalNode,
    selection: AbstractSet[identifiers.ColumnId],
) -> bigframes.core.nodes.ReadLocalNode:
    new_scan_list = filter_scanlist(node.scan_list, selection)
    return dataclasses.replace(
        node,
        scan_list=new_scan_list,
        offsets_col=node.offsets_col if (node.offsets_col in selection) else None,
    )


@prune_leaf.register
def prune_readtable(
    node: bigframes.core.nodes.ReadTableNode,
    selection: AbstractSet[identifiers.ColumnId],
) -> bigframes.core.nodes.ReadTableNode:
    new_scan_list = filter_scanlist(node.scan_list, selection)
    return dataclasses.replace(node, scan_list=new_scan_list)


def filter_scanlist(
    scanlist: bigframes.core.nodes.ScanList,
    ids: AbstractSet[identifiers.ColumnId],
):
    result = bigframes.core.nodes.ScanList(
        tuple(item for item in scanlist.items if item.id in ids)
    )
    if len(result.items) == 0:
        # We need to select something, or stuff breaks
        result = bigframes.core.nodes.ScanList(scanlist.items[:1])
    return result
