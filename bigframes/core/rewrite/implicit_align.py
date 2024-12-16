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

from typing import Iterable, Optional, Tuple

import bigframes.core.expression
import bigframes.core.guid
import bigframes.core.identifiers
import bigframes.core.join_def
import bigframes.core.nodes
import bigframes.core.window_spec
import bigframes.operations.aggregations

# Additive nodes leave existing columns completely intact, and only add new columns to the end
ADDITIVE_NODES = (
    bigframes.core.nodes.ProjectionNode,
    bigframes.core.nodes.WindowOpNode,
    bigframes.core.nodes.PromoteOffsetsNode,
)
# Combination of selects and additive nodes can be merged as an explicit keyless "row join"
ALIGNABLE_NODES = (
    *ADDITIVE_NODES,
    bigframes.core.nodes.SelectionNode,
)


def rewrite_row_join(
    node: bigframes.core.nodes.BigFrameNode,
):
    if not isinstance(node, bigframes.core.nodes.RowJoinNode):
        return node

    l_node = node.left_child
    r_node = node.right_child
    divergent_node = first_shared_descendent(
        l_node, r_node, descendable_types=ALIGNABLE_NODES
    )
    assert divergent_node is not None
    l_node, l_selection = pull_up_selection(l_node, stop=divergent_node)
    r_node, r_selection = pull_up_selection(
        r_node, stop=divergent_node, rename_vars=True
    )  # Rename only right vars to avoid collisions with left vars
    combined_selection = (*l_selection, *r_selection)

    def _linearize_trees(
        base_tree: bigframes.core.nodes.BigFrameNode,
        append_tree: bigframes.core.nodes.BigFrameNode,
    ) -> bigframes.core.nodes.BigFrameNode:
        """Linearize two divergent tree who only diverge through different additive nodes."""
        # base case: append tree does not have any divergent nodes to linearize
        if append_tree == divergent_node:
            return base_tree
        else:
            assert isinstance(append_tree, ADDITIVE_NODES)
            return append_tree.replace_child(
                _linearize_trees(base_tree, append_tree.child)
            )

    merged_node = _linearize_trees(l_node, r_node)
    return bigframes.core.nodes.SelectionNode(merged_node, combined_selection)


def pull_up_selection(
    node: bigframes.core.nodes.BigFrameNode,
    stop: bigframes.core.nodes.BigFrameNode,
    rename_vars: bool = False,
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
    if node == stop:  # base case
        return node, tuple(
            (bigframes.core.expression.DerefOp(field.id), field.id)
            for field in node.fields
        )
    assert isinstance(node, (bigframes.core.nodes.SelectionNode, *ADDITIVE_NODES))
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


## Traversal helpers
def first_shared_descendent(
    left: bigframes.core.nodes.BigFrameNode,
    right: bigframes.core.nodes.BigFrameNode,
    descendable_types: Tuple[type[bigframes.core.nodes.BigFrameNode], ...],
) -> Optional[bigframes.core.nodes.BigFrameNode]:
    l_path = tuple(descend_left(left, descendable_types))
    r_path = tuple(descend_left(right, descendable_types))
    if l_path[-1] != r_path[-1]:
        return None

    for l_node, r_node in zip(l_path[-len(r_path) :], r_path[-len(l_path) :]):
        if l_node == r_node:
            return l_node
    # should be impossible, as l_path[-1] == r_path[-1]
    raise ValueError()


def descend_left(
    root: bigframes.core.nodes.BigFrameNode,
    descendable_types: Tuple[type[bigframes.core.nodes.BigFrameNode], ...],
) -> Iterable[bigframes.core.nodes.BigFrameNode]:
    yield root
    if isinstance(root, descendable_types):
        yield from descend_left(root.child_nodes[0], descendable_types)
