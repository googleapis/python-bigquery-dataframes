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

import dataclasses
import itertools
from typing import Optional, Sequence, Tuple

import bigframes.core.expression as scalar_exprs
import bigframes.core.identifiers as ids
import bigframes.core.join_def as join_defs
import bigframes.core.nodes as nodes
import bigframes.core.ordering as order
import bigframes.core.rewrite.implicit_align
import bigframes.core.rewrite.predicates
import bigframes.operations as ops

Selection = Tuple[Tuple[scalar_exprs.Expression, ids.ColumnId], ...]

LEGACY_REWRITER_NODES = (
    bigframes.core.nodes.ProjectionNode,
    bigframes.core.nodes.SelectionNode,
    bigframes.core.nodes.ReversedNode,
    bigframes.core.nodes.OrderByNode,
    bigframes.core.nodes.FilterNode,
)


@dataclasses.dataclass(frozen=True)
class SquashedSelect:
    """Squash nodes together until target node, separating out the projection, filter and reordering expressions."""

    root: nodes.BigFrameNode
    columns: Tuple[Tuple[scalar_exprs.Expression, ids.ColumnId], ...]
    predicates: Tuple[scalar_exprs.Expression, ...]
    ordering: Tuple[order.OrderingExpression, ...]
    reverse_root: bool = False

    @classmethod
    def from_node_span(
        cls, node: nodes.BigFrameNode, target: nodes.BigFrameNode
    ) -> SquashedSelect:
        node, selection = bigframes.core.rewrite.implicit_align.pull_up_selection(
            node, target, rename_vars=True
        )
        node, filters = bigframes.core.rewrite.implicit_align.pull_up_filters(
            node, target
        )
        node, ordering = bigframes.core.rewrite.implicit_align.pull_up_order(
            node, target
        )
        node, reverse_root = bigframes.core.rewrite.implicit_align.pull_up_reverse(
            node, target
        )
        return SquashedSelect(node, selection, filters, ordering, reverse_root)

    def merge(
        self,
        right: SquashedSelect,
        join_type: join_defs.JoinType,
        join_keys: Tuple[join_defs.CoalescedColumnMapping, ...],
        mappings: Tuple[join_defs.JoinColumnMapping, ...],
    ) -> SquashedSelect:
        new_root = bigframes.core.rewrite.implicit_align.linearize_trees(
            self.root, right.root
        )
        # Mask columns and remap names to expected schema
        lselection = self.columns
        rselection = right.columns
        if join_type == "outer":
            new_predicate = or_predicates(self.predicates, right.predicates)
            new_predicates = (new_predicate,) if new_predicate else ()
        elif join_type == "inner":
            new_predicate = and_predicates(self.predicates, right.predicates)
            new_predicates = (new_predicate,) if new_predicate else ()
        elif join_type == "left":
            new_predicates = self.predicates
        elif join_type == "right":
            new_predicates = right.predicates

        l_relative, r_relative = bigframes.core.rewrite.predicates.relative_predicates(
            self.predicates, right.predicates
        )
        if join_type in {"right", "outer"} and len(l_relative) > 0:
            lmask = bigframes.core.rewrite.predicates.merge_predicates(l_relative)
        else:
            lmask = None
        if join_type in {"left", "outer"} and len(r_relative) > 0:
            rmask = bigframes.core.rewrite.predicates.merge_predicates(r_relative)
        else:
            rmask = None

        new_columns = merge_expressions(
            join_keys, mappings, lselection, rselection, lmask, rmask
        )

        # Reconstruct ordering
        reverse_root = self.reverse_root
        if join_type == "right":
            new_ordering = right.ordering
            reverse_root = right.reverse_root
        elif join_type == "outer":
            if lmask is not None:
                prefix = order.OrderingExpression(lmask, order.OrderingDirection.DESC)
                left_ordering = tuple(
                    order.OrderingExpression(
                        bigframes.core.rewrite.predicates.apply_mask_expr(
                            ref.scalar_expression, lmask
                        ),
                        ref.direction,
                        ref.na_last,
                    )
                    for ref in self.ordering
                )
                right_ordering = (
                    tuple(
                        order.OrderingExpression(
                            bigframes.core.rewrite.predicates.apply_mask_expr(
                                ref.scalar_expression, rmask
                            ),
                            ref.direction,
                            ref.na_last,
                        )
                        for ref in right.ordering
                    )
                    if rmask
                    else right.ordering
                )
                new_ordering = (prefix, *left_ordering, *right_ordering)
            else:
                new_ordering = self.ordering
        elif join_type in {"inner", "left"}:
            new_ordering = self.ordering
        else:
            raise ValueError(f"Unexpected join type {join_type}")
        return SquashedSelect(
            new_root, new_columns, new_predicates, new_ordering, reverse_root
        )

    def expand(self) -> nodes.BigFrameNode:
        # Safest to apply predicates first, as it may filter out inputs that cannot be handled by other expressions
        root = self.root
        if self.reverse_root:
            root = nodes.ReversedNode(child=root)
        if len(self.predicates) > 0:
            root = nodes.FilterNode(
                child=root,
                predicate=bigframes.core.rewrite.predicates.merge_predicates(
                    self.predicates
                ),
            )
        if self.ordering:
            root = nodes.OrderByNode(child=root, by=self.ordering)
        selection = tuple((scalar_exprs.DerefOp(id), id) for _, id in self.columns)
        return nodes.SelectionNode(
            child=nodes.ProjectionNode(child=root, assignments=self.columns),
            input_output_pairs=selection,
        )


def legacy_join_as_projection(
    l_node: nodes.BigFrameNode,
    r_node: nodes.BigFrameNode,
    join_keys: Tuple[join_defs.CoalescedColumnMapping, ...],
    mappings: Tuple[join_defs.JoinColumnMapping, ...],
    how: join_defs.JoinType,
) -> Optional[nodes.BigFrameNode]:
    rewrite_common_node = bigframes.core.rewrite.implicit_align.first_shared_descendent(
        l_node, r_node, descendable_types=LEGACY_REWRITER_NODES
    )
    if rewrite_common_node is None:
        return None

    # check join keys are equivalent by normalizing the expressions as much as posisble
    # instead of just comparing ids
    for join_key in join_keys:
        # Caller is block, so they still work with raw strings rather than ids
        left_id = ids.ColumnId(join_key.left_source_id)
        right_id = ids.ColumnId(join_key.right_source_id)
        if bigframes.core.rewrite.implicit_align.get_expression_spec(
            l_node, left_id
        ) != bigframes.core.rewrite.implicit_align.get_expression_spec(
            r_node, right_id
        ):
            return None

    left_side = SquashedSelect.from_node_span(l_node, rewrite_common_node)
    right_side = SquashedSelect.from_node_span(r_node, rewrite_common_node)
    merged = left_side.merge(right_side, how, join_keys, mappings)
    return merged.expand()


def merge_expressions(
    join_keys: Tuple[join_defs.CoalescedColumnMapping, ...],
    mappings: Tuple[join_defs.JoinColumnMapping, ...],
    lselection: Selection,
    rselection: Selection,
    lmask: Optional[scalar_exprs.Expression],
    rmask: Optional[scalar_exprs.Expression],
) -> Selection:
    new_selection: Selection = tuple()
    # Assumption is simple ids
    l_exprs_by_id = {id.name: expr for expr, id in lselection}
    r_exprs_by_id = {id.name: expr for expr, id in rselection}
    for key in join_keys:
        expr = l_exprs_by_id[key.left_source_id]
        id = key.destination_id
        new_selection = (*new_selection, (expr, ids.ColumnId(id)))
    for mapping in mappings:
        if mapping.source_table == join_defs.JoinSide.LEFT:
            expr = l_exprs_by_id[mapping.source_id]
            if lmask:
                expr = bigframes.core.rewrite.predicates.apply_mask_expr(expr, lmask)
        else:  # Right
            expr = r_exprs_by_id[mapping.source_id]
            if rmask:
                expr = bigframes.core.rewrite.predicates.apply_mask_expr(expr, rmask)
        new_selection = (*new_selection, (expr, ids.ColumnId(mapping.destination_id)))
    return new_selection


def and_predicates(
    left_predicates: Sequence[scalar_exprs.Expression],
    right_predicates: Sequence[scalar_exprs.Expression],
) -> Optional[scalar_exprs.Expression]:
    if not (left_predicates or right_predicates):
        return None
    # remove common predicates
    all_predicates = itertools.chain(
        left_predicates, [p for p in right_predicates if p not in left_predicates]
    )
    return bigframes.core.rewrite.predicates.merge_predicates(list(all_predicates))


def or_predicates(
    l_predicates: Sequence[scalar_exprs.Expression],
    r_predicates: Sequence[scalar_exprs.Expression],
) -> Optional[scalar_exprs.Expression]:
    if (not l_predicates) or (not r_predicates):
        return None
    # TODO(tbergeron): Factor out common predicates
    return ops.or_op.as_expr(
        bigframes.core.rewrite.predicates.merge_predicates(l_predicates),
        bigframes.core.rewrite.predicates.merge_predicates(r_predicates),
    )
