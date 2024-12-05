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
from typing import Callable, Dict, Optional, Sequence, Set

import bigframes.core.identifiers
import bigframes.core.nodes as nodes


def is_trivially_executable(node: nodes.BigFrameNode) -> bool:
    if local_only(node):
        return True
    children_trivial = all(is_trivially_executable(child) for child in node.child_nodes)
    self_trivial = (not node.non_local) and (node.row_preserving)
    return children_trivial and self_trivial


def local_only(node: nodes.BigFrameNode) -> bool:
    return all(isinstance(node, nodes.ReadLocalNode) for node in node.roots)


def can_fast_peek(node: nodes.BigFrameNode) -> bool:
    if local_only(node):
        return True
    children_peekable = all(can_fast_peek(child) for child in node.child_nodes)
    self_peekable = not node.non_local
    return children_peekable and self_peekable


def can_fast_head(node: nodes.BigFrameNode) -> bool:
    """Can get head fast if can push head operator down to leafs and operators preserve rows."""
    # To do fast head operation:
    # (1) the underlying data must be arranged/indexed according to the logical ordering
    # (2) transformations must support pushing down LIMIT or a filter on row numbers
    return has_fast_offset_address(node) or has_fast_offset_address(node)


def has_fast_orderby_limit(node: nodes.BigFrameNode) -> bool:
    """True iff ORDER BY LIMIT can be performed without a large full table scan."""
    # TODO: In theory compatible with some Slice nodes, potentially by adding OFFSET
    if isinstance(node, nodes.LeafNode):
        return node.fast_ordered_limit
    if isinstance(node, (nodes.ProjectionNode, nodes.SelectionNode)):
        return has_fast_orderby_limit(node.child)
    return False


def has_fast_offset_address(node: nodes.BigFrameNode) -> bool:
    """True iff specific offsets can be scanned without a large full table scan."""
    # TODO: In theory can push offset lookups through slice operators by translating indices
    if isinstance(node, nodes.LeafNode):
        return node.fast_offsets
    if isinstance(node, (nodes.ProjectionNode, nodes.SelectionNode)):
        return has_fast_offset_address(node.child)
    return False


def row_count(node: nodes.BigFrameNode) -> Optional[int]:
    """Determine row count from local metadata, return None if unknown."""
    return node.row_count


# Replace modified_cost(node) = cost(apply_cache(node))
def select_cache_target(
    root: nodes.BigFrameNode,
    min_complexity: float,
    max_complexity: float,
    cache: dict[nodes.BigFrameNode, nodes.BigFrameNode],
    heuristic: Callable[[int, int], float],
) -> Optional[nodes.BigFrameNode]:
    """Take tree, and return candidate nodes with (# of occurences, post-caching planning complexity).

    heurstic takes two args, node complexity, and node occurence count, in that order
    """

    @functools.cache
    def _with_caching(subtree: nodes.BigFrameNode) -> nodes.BigFrameNode:
        return replace_nodes(subtree, cache)

    def _combine_counts(
        left: Dict[nodes.BigFrameNode, int], right: Dict[nodes.BigFrameNode, int]
    ) -> Dict[nodes.BigFrameNode, int]:
        return {
            key: left.get(key, 0) + right.get(key, 0)
            for key in itertools.chain(left.keys(), right.keys())
        }

    @functools.cache
    def _node_counts_inner(
        subtree: nodes.BigFrameNode,
    ) -> Dict[nodes.BigFrameNode, int]:
        """Helper function to count occurences of duplicate nodes in a subtree. Considers only nodes in a complexity range"""
        empty_counts: Dict[nodes.BigFrameNode, int] = {}
        subtree_complexity = _with_caching(subtree).planning_complexity
        if subtree_complexity >= min_complexity:
            child_counts = [_node_counts_inner(child) for child in subtree.child_nodes]
            node_counts = functools.reduce(_combine_counts, child_counts, empty_counts)
            if subtree_complexity <= max_complexity:
                return _combine_counts(node_counts, {subtree: 1})
            else:
                return node_counts
        return empty_counts

    node_counts = _node_counts_inner(root)

    if len(node_counts) == 0:
        raise ValueError("node counts should be non-zero")

    return max(
        node_counts.keys(),
        key=lambda node: heuristic(
            _with_caching(node).planning_complexity, node_counts[node]
        ),
    )


def count_nodes(forest: Sequence[nodes.BigFrameNode]) -> dict[nodes.BigFrameNode, int]:
    """
    Counts the number of instances of each subtree present within a forest.

    Memoizes internally to accelerate execution, but cache not persisted (not reused between invocations).

    Args:
        forest (Sequence of BigFrameNode):
            The roots of each tree in the forest

    Returns:
        dict[BigFramesNode, int]: The number of occurences of each subtree.
    """

    def _combine_counts(
        left: Dict[nodes.BigFrameNode, int], right: Dict[nodes.BigFrameNode, int]
    ) -> Dict[nodes.BigFrameNode, int]:
        return {
            key: left.get(key, 0) + right.get(key, 0)
            for key in itertools.chain(left.keys(), right.keys())
        }

    empty_counts: Dict[nodes.BigFrameNode, int] = {}

    @functools.cache
    def _node_counts_inner(
        subtree: nodes.BigFrameNode,
    ) -> Dict[nodes.BigFrameNode, int]:
        """Helper function to count occurences of duplicate nodes in a subtree. Considers only nodes in a complexity range"""
        child_counts = [_node_counts_inner(child) for child in subtree.child_nodes]
        node_counts = functools.reduce(_combine_counts, child_counts, empty_counts)
        return _combine_counts(node_counts, {subtree: 1})

    counts = [_node_counts_inner(root) for root in forest]
    return functools.reduce(_combine_counts, counts, empty_counts)


def replace_nodes(
    root: nodes.BigFrameNode,
    replacements: dict[nodes.BigFrameNode, nodes.BigFrameNode],
):
    @functools.cache
    def apply_substition(node: nodes.BigFrameNode) -> nodes.BigFrameNode:
        if node in replacements.keys():
            return replacements[node]
        else:
            return node.transform_children(apply_substition)

    return apply_substition(root)


def is_unique_key(
    root: nodes.BigFrameNode, key: Set[bigframes.core.identifiers.ColumnId]
) -> bool:
    """
    Does the set of columns together from a unique key for the rows.

    False negatives may occur, but won't give false positives.

    Not very strict, extra column ids are ok and will just be ignored.
    """
    if root.row_count == 1:
        return True
    if key.issuperset(root.ids):
        return True
    if isinstance(root, bigframes.core.nodes.AggregateNode):
        return key.issuperset(set(root.by_column_ids))
    if isinstance(root, bigframes.core.nodes.SelectionNode):
        mapping = {v: k.id for k, v in root.input_output_pairs}
        remapped_key = set(mapping[k] for k in key if k in mapping)
        return is_unique_key(root.child, remapped_key)
    if isinstance(root, bigframes.core.nodes.ReadTableNode):
        ordering = root.source.ordering
        if (ordering is not None) and isinstance(
            ordering, bigframes.core.orderings.TotalOrdering
        ):
            return key.issuperset(ref.id for ref in ordering.total_ordering_columns)
        return False
    if isinstance(root, bigframes.core.nodes.PromoteOffsetsNode):
        return key.issuperset({root.col_id}) or is_unique_key(root.child, key)
    if isinstance(root, bigframes.core.nodes.FromRangeNode):
        return key.issuperset({root.output_id})
    if isinstance(root, bigframes.core.nodes.ConcatNode):
        return False
    if isinstance(root, bigframes.core.nodes.JoinNode):
        return is_unique_key(root.left_child, key) and is_unique_key(
            root.right_child, key
        )
    if isinstance(root, bigframes.core.nodes.ExplodeNode):
        return key.issuperset(set(root.column_ids)) and is_unique_key(root.child, key)
    elif isinstance(
        root,
        (
            bigframes.core.nodes.WindowOpNode,  # row id preserving
            bigframes.core.nodes.ProjectionNode,
            bigframes.core.nodes.RandomSampleNode,
            bigframes.core.nodes.OrderByNode,
            bigframes.core.nodes.ReversedNode,
            bigframes.core.nodes.SliceNode,
            bigframes.core.nodes.FilterNode,
        ),
    ):
        return is_unique_key(root.child, key)
    return False
