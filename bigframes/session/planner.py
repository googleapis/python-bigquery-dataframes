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

import itertools
from typing import Sequence

import bigframes.core.identifiers as ids
import bigframes.core.nodes as nodes
import bigframes.core.pruning as predicate_pruning
import functools
import bigframes.dtypes
import bigframes.core.expression

def select_cluster_cols(
    cache_node: nodes.BigFrameNode, session_forest: Sequence[nodes.BigFrameNode]
) -> set[ids.ColumnId]:
    """
    Determines the best cluster cols for materializing a target node give a list of session trees.
    """
    @functools.cache
    def find_direct_predicates(cache_node, root_node: nodes.BigFrameNode) -> set[bigframes.core.expression.Expression]:
        if isinstance(root_node, nodes.FilterNode):
            # Filter node doesn't define any variables, so no need to chain expressions
            filters.append(root_node.predicate)
        elif isinstance(root_node, nodes.ProjectionNode):
            # Projection defines the variables that are used in the filter expressions, need to substitute variables with their scalar expressions
            # that instead reference variables in the child node.
            bindings = {name: expr for expr, name in root_node.assignments}
            filters = [
                i.bind_refs(bindings, allow_partial_bindings=True) for i in filters
            ]
        elif isinstance(root_node, nodes.SelectionNode):
            bindings = {output: input for input, output in root_node.input_output_pairs}
            filters = [i.bind_refs(bindings) for i in filters]
        else:
            return frozenset().union(find_direct_predicates(cache_node, child) for child in root_node.child_nodes)


    cluster_compatible_cols = {
                    field.id
                    for field in cur_node.fields
                    if bigframes.dtypes.is_clusterable(field.dtype)
                }
    clusterable_cols = set(
        itertools.chain.from_iterable(
            map(
                lambda f: predicate_pruning.cluster_cols_for_predicate(
                    f, cluster_compatible_cols
                ),
                filters,
            )
        )
    )
    # BQ supports up to 4 cluster columns, just prioritize by alphabetical ordering
    # TODO: Prioritize caching columns by estimated filter selectivity
    return sorted(list(clusterable_cols))[:4]
