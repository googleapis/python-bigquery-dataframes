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
import dataclasses
import functools
from typing import Optional

from bigframes.core import nodes


def try_reduce_to_table_scan(root: nodes.BigFrameNode) -> Optional[nodes.ReadTableNode]:
    for node in root.unique_nodes():
        if not isinstance(node, (nodes.ReadTableNode, nodes.SelectionNode)):
            return None
    result = root.bottom_up(merge_scan)
    if isinstance(result, nodes.ReadTableNode):
        return result
    return None


def try_reduce_to_local_scan(node: nodes.BigFrameNode) -> Optional[nodes.ReadLocalNode]:
    # Top-level slice is supported.
    start: Optional[int] = None
    stop: Optional[int] = None

    if isinstance(node, nodes.SliceNode):
        # These slice features are not supported by pyarrow.Table.slice. Must
        # have a non-negative start and stop and no custom step size.
        if (
            (node.step is not None and node.step != 1)
            or (node.start is not None and node.start < 0)
            or (node.stop is not None and node.stop < 0)
        ):
            return None

        start = node.start
        stop = node.stop
        node = node.child

    if not all(
        map(
            lambda x: isinstance(x, (nodes.ReadLocalNode, nodes.SelectionNode)),
            node.unique_nodes(),
        )
    ):
        return None

    result = node.bottom_up(merge_scan)
    if isinstance(result, nodes.ReadLocalNode):
        return dataclasses.replace(
            result,
            slice_start=start,
            slice_stop=stop,
        )
    return None


@functools.singledispatch
def merge_scan(node: nodes.BigFrameNode) -> nodes.BigFrameNode:
    return node


@merge_scan.register
def _(node: nodes.SelectionNode) -> nodes.BigFrameNode:
    if not isinstance(node.child, (nodes.ReadTableNode, nodes.ReadLocalNode)):
        return node
    if node.has_multi_referenced_ids:
        return node
    if isinstance(node, nodes.ReadLocalNode) and node.offsets_col is not None:
        return node
    selection = {
        aliased_ref.ref.id: aliased_ref.id for aliased_ref in node.input_output_pairs
    }
    new_scan_list = node.child.scan_list.project(selection)
    return dataclasses.replace(node.child, scan_list=new_scan_list)
