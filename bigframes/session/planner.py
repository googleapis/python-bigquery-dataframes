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

from typing import Optional, Sequence, Tuple

import bigframes.core.expression as ex
import bigframes.core.nodes as nodes
import bigframes.core.pruning as predicate_pruning
import bigframes.core.tree_properties as traversals


def session_aware_cache_plan(
    root: nodes.BigFrameNode, session_forest: Sequence[nodes.BigFrameNode]
) -> Tuple[nodes.BigFrameNode, Optional[str]]:
    """
    Determines the best node to cache given a target and a list of object roots for objects in a session.

    Returns the node to cache, and optionally a clustering column.
    """
    node_counts = traversals.count_nodes(session_forest)
    # These node types are cheap to re-compute
    de_cachable_types = (nodes.FilterNode, nodes.ProjectionNode)
    caching_target = cur_node = root
    caching_target_refs = node_counts.get(caching_target, 0)

    filters: list[
        ex.Expression
    ] = []  # accumulate filters into this as traverse downwards
    cluster_col: Optional[str] = None
    while isinstance(cur_node, de_cachable_types):
        if isinstance(cur_node, nodes.FilterNode):
            filters.append(cur_node.predicate)
        elif isinstance(cur_node, nodes.ProjectionNode):
            bindings = {name: expr for expr, name in cur_node.assignments}
            filters = [i.bind_all_variables(bindings) for i in filters]

        cur_node = cur_node.child
        cur_node_refs = node_counts.get(cur_node, 0)
        if cur_node_refs > caching_target_refs:
            caching_target, caching_target_refs = cur_node, cur_node_refs
            cluster_col = None
            # Just pick the first cluster-compatible predicate
            for predicate in filters:
                # Cluster cols only consider the target object and not other sesssion objects
                cluster_cols = predicate_pruning.cluster_cols_for_predicate(predicate)
                if len(cluster_cols) > 0:
                    cluster_col = cluster_cols[0]
                    continue
    return caching_target, cluster_col
