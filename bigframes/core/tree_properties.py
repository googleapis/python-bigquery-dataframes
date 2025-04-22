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

import abc
import dataclasses
import functools
import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
import weakref

import bigframes.core.nodes as nodes


@dataclasses.dataclass(frozen=True)
class GetPropRequest:
    property: Property
    node: nodes.BigFrameNode


T = TypeVar("T", bound=Hashable)


@dataclasses.dataclass(frozen=True)
class Property(abc.ABC, Generic[T]):
    id: str

    @abc.abstractmethod
    def get(self, node: nodes.BigFrameNode) -> Generator[GetPropRequest, Any, T]:
        ...


@dataclasses.dataclass(frozen=True)
class Height(Property[int]):
    id: str = "Height"

    def get(self, node: nodes.BigFrameNode) -> Generator[GetPropRequest, int, int]:
        if len(node.child_nodes) == 0:
            return 0
        heights = []
        for child in node.child_nodes:
            child_height = yield GetPropRequest(self, child)
            heights.append(child_height)
        return min(heights) + 1


class PropertyScope:
    _memoized: weakref.WeakKeyDictionary[
        nodes.BigFrameNode, Dict[Property, Any]
    ] = weakref.WeakKeyDictionary()

    def get_property(
        self, node: nodes.BigFrameNode, property_instance: Property[T]
    ) -> T:
        """
        Gets the value of a property for a given node, using memoization and
        an iterative approach with an explicit stack to avoid deep Python recursion
        in this method.

        Args:
            node: The node for which to compute the property.
            property_instance: The specific Property object (e.g., an instance of Height).

        Returns:
            The computed value of the property for the node.

        Raises:
            RecursionError: If a cycle in property dependencies is detected.
            TypeError: If a property's get() method yields an unexpected value type.
            Exception: Propagates exceptions raised during property computation.
        """
        # Stack to manage generators that are waiting for a dependency result.
        generator_stack: List[Tuple[Generator, nodes.BigFrameNode, Property]] = []

        # Value computed by the most recent dependency, to be sent into the parent generator.
        result_to_send: Any = None

        # The current node and property being processed. Start with the initial request.
        current_node: nodes.BigFrameNode = node
        current_prop: Property = property_instance
        current_gen: Optional[Generator[Any, None, None]] = None

        while True:
            if self._cache_contains(current_node, current_prop):
                result_to_send = self._get_cache(current_node, current_prop)
                if not generator_stack:
                    # No generators are waiting, this is the final result.
                    return result_to_send
                else:
                    current_gen, current_node, current_prop = generator_stack.pop()
                    continue
            elif current_gen is None:
                current_gen = current_prop.get(current_node)
                result_to_send = None  # Send None initially, gen.send(None) = next(gen)
            try:
                # Send the result from a completed dependency (or None initially).
                request = current_gen.send(result_to_send)
                result_to_send = None  # Consume the result after sending

                # Generator yielded a request for another property.
                assert isinstance(request, GetPropRequest)
                # Push the current generator onto the stack, as it's now waiting.
                generator_stack.append((current_gen, current_node, current_prop))
                # Set the new dependency as the current task.
                current_node, current_prop = request.node, request.property
                current_gen = None  # Clear current generator before looping
                continue

            except StopIteration as e:
                self._insert_cache(current_node, current_prop, e.value)
                result_to_send = e.value
                # Check if this completes the overall request or a dependency.
                if not generator_stack:
                    return result_to_send  # Final result
                else:
                    current_gen, current_node, current_prop = generator_stack.pop()
                    continue

    def _insert_cache(self, node, property, value):
        if node not in self._memoized:
            self._memoized[node] = {}
        self._memoized[node][property] = value

    def _cache_contains(self, node, property: Property[T]) -> bool:
        node_cache = self._memoized.get(node)
        if node_cache is not None and property in node_cache:
            return True
        return False

    def _get_cache(self, node, property: Property[T]) -> T:
        node_cache = self._memoized.get(node)
        if node_cache is not None and property in node_cache:
            return node_cache[property]  # Get cached result
        raise ValueError("no value")


HEIGHT_PROP = Height()
GLOBAL_PROPERTY_SCOPE = PropertyScope()


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
        return nodes.top_down(subtree, lambda x: cache.get(x, x))

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
