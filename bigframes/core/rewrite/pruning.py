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
import bigframes.core.nodes
import functools
import bigframes.core.identifiers
import dataclasses
from typing import Tuple, TypeVar, Iterable

# TODO: Need to preserve column sequence as Concat node is sensitive to column order
# Could increase performance if column order becomes unnecessary, and can store as set instead
COLUMN_SEQUENCE = Tuple[bigframes.core.identifiers.ColumnId, ...]


def column_pruning(root: bigframes.core.nodes.BigFrameNode) -> bigframes.core.nodes.BigFrameNode:
    return bigframes.core.nodes.top_down(root, prune_columns)

def to_fixed(max_iterations=100):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            current_value = f(*args, **kwargs)
            for _ in range(max_iterations):
                next_value = f(*args, **kwargs)
                if current_value == next_value:
                    return next_value
                current_value = next_value
            raise ValueError(f"Could not reach fixed point within {max_iterations} iterations.")
        return wrapper
    return decorator

@to_fixed(max_iterations=20)
def prune_columns(node: bigframes.core.nodes.BigFrameNode):
    if isinstance(node, bigframes.core.nodes.SelectionNode):
        selection = node.input_output_pairs
        child = node.child
        if selection.is_ambiguous:
            new_selection = bigframes.core.nodes.Selection.from_cols(list(dedupe(id for _, id in selection)))
            return node.replace_child(prune_node(child, new_selection))
        # TODO: if selection 
        if isinstance(child, bigframes.core.nodes.SelectionNode):
            merged_selection = node.input_output_pairs.chain(child.input_output_pairs)
            return bigframes.core.nodes.SelectionNode(child.child, merged_selection)
        if isinstance(child, bigframes.core.nodes.AdditiveNode):
            if not set(child.added_fields) & set(selection.consumed_ids):
                return node.replace_child(child.additive_base)
            else:
                return node.remap_ids(selection.get_id_mapping).replace_child(prune_columns(child.additive_base))

        # case join: push down to both sides
        if isinstance(node.child, bigframes.core.nodes.JoinNode):
            l_ids = tuple(id for id in node.child.consumed_ids if id in node.child.left_child.ids)
            r_ids = tuple(id for id in node.child.consumed_ids if id in node.child.right_child.ids)
            l_node = prune_node(node.child.left_child, l_ids)
            r_node = prune_node(node.child.right_child, r_ids)
            return dataclasses.replace(node.child, left_child=l_node, right_child=r_node)

        if isinstance(child, bigframes.core.nodes.ConcatNode):
            indices = [list(child.ids).index(ref.id) for ref, _ in selection]
            new_children = []
            for concat_node in child.child_nodes:
                cc_ids = tuple(concat_node.ids)
                sub_selection = bigframes.core.nodes.Selection.from_cols(cc_ids[i] for i in indices)
                new_children.append(bigframes.core.nodes.Selection(concat_node, sub_selection))
            return bigframes.core.nodes.ConcatNode(children=new_children, output_ids=tuple(node.ids))
        # Nodes that purely pass through
        if isinstance(child, (bigframes.core.nodes.RandomSampleNode, bigframes.core.nodes.ReversedNode, bigframes.core.nodes.OrderByNode, bigframes.core.nodes.FilterNode, bigframes.core.nodes.SliceNode)):
            # TODO: Append referenced ids
            return dataclasses.replace(child.remap_refs(selection.get_id_mapping), child=node.replace_child(child.child))
        # TODO: OrderBy, Filter may create double selection
        if isinstance(child, bigframes.core.nodes.ExplodeNode):
            if not set(selection.consumed_ids) & set(child.column_ids):
                return node.replace_child(prune_node(child))
            return node.replace_child(prune_node(child, child.column_ids))
        if isinstance(child, bigframes.core.nodes.LeafNode):
            return prune_leaf(child, selection)
        else: # from range
    elif isinstance(node, bigframes.core.nodes.AggregateNode):
        consumed_ids = node.consumed_ids
        return node.replace_child(prune_node(node.child, consumed_ids))
    else:
        return node
        




# Might be two versions, one preserves column order and one doesn't?
def prune_node(node: bigframes.core.nodes.ReadLocalNode, selection: bigframes.core.nodes.Selection):
    if selection.is_identity and tuple(node.ids) == selection.ids:
        return node
    else:
        return bigframes.core.nodes.SelectionNode(node, selection)
    

def prune_aggregate(node: bigframes.core.nodes.AggregateNode, used_cols: bigframes.core.nodes.Selection) -> bigframes.core.nodes.AggregateNode:
    pruned_aggs = (
        tuple(agg for agg in node.aggregations if agg[1] in used_cols)
        or node.aggregations[:1]
    )
    return dataclasses.replace(aggregations=pruned_aggs)



@functools.singledispatch
def prune_leaf(node: bigframes.core.nodes.BigFrameNode, used_cols: bigframes.core.nodes.Selection):
    ...
    

@prune_leaf.register
def prune_readlocal(node: bigframes.core.nodes.ReadLocalNode, selection: bigframes.core.nodes.Selection) -> bigframes.core.nodes.ReadLocalNode:
    new_scan_list = push_selection_scanlist(node.scan_list, selection)
    return dataclasses.replace(node, scan_list=new_scan_list)

@prune_leaf.register
def prune_readtable(node: bigframes.core.nodes.ReadTableNode, selection: bigframes.core.nodes.Selection) -> bigframes.core.nodes.ReadTableNode:
    new_scan_list = push_selection_scanlist(node.scan_list, selection)
    return dataclasses.replace(node, scan_list=new_scan_list)


def push_selection_scanlist(scanlist: bigframes.core.nodes.ScanList, selection: bigframes.core.nodes.Selection):
    scan_item_by_id = {item.id: item for item in scanlist.items}
    new_items = tuple(scan_item_by_id[ref].with_id(id) for ref, id in selection)
    return bigframes.core.nodes.ScanList(new_items)

T = TypeVar["T"]
def dedupe(
  items: Iterable[T]      
) -> Iterable[T]:
    seen = set() 

    for item in items:
        if item not in seen:
            seen.add(item)
            yield item
