# Copyright 2025 Google LLC
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

from collections import defaultdict

from bigframes.core import nodes


def extract_ctes(root: nodes.BigFrameNode) -> nodes.BigFrameNode:
    # identify candidates
    # candidates
    node_parents: dict[nodes.BigFrameNode, int] = defaultdict(int)
    for parent, child in root.edges():
        node_parents[child] += 1

    # ok time to replace via extract
    # we just mark in place, rather than pull out of the tree.
    # if we did pull out of tree, we'd want to make sure to extract bottom-up
    def insert_cte_markers(node: nodes.BigFrameNode) -> nodes.BigFrameNode:
        if node_parents[node] > 1:
            return nodes.CteNode(node)
        return node

    return root.top_down(insert_cte_markers)
