# Copyright 2023 Google LLC
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

import bigframes.core.nodes as nodes


def squash_projections(node: nodes.BigFrameNode) -> nodes.BigFrameNode:
    if not isinstance(node, nodes.ProjectionNode):
        return node
    child = node.child
    if not isinstance(child, nodes.ProjectionNode):
        return node
    substitutions = {variable: expr for expr, variable in child.assignments}
    squashed_projection = tuple(
        (expr.substitute(substitutions), id) for expr, id in node.assignments
    )
    return nodes.ProjectionNode(child.child, assignments=squashed_projection)


# Rewrites are applied in this order per-node from bottom up
DEFAULT_REWRITES = [
    squash_projections,
]


class ExpressionTreeRewriter:
    """Simple recursive context-free tree rewriter."""

    def __init__(self, rewrites=DEFAULT_REWRITES):
        self._rewrites = rewrites

    def rewrite(self, node: nodes.BigFrameNode) -> nodes.BigFrameNode:
        children = [self.rewrite(child) for child in node.child_nodes]
        node = node.swap_children(tuple(children))
        for transform in self._rewrites:
            node = transform(node)
        return node
