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

import fractions

import bigframes.core.agg_expressions as agg_ex
import bigframes.core.expression as ex
import bigframes.core.identifiers as ids
import bigframes.core.nodes as nodes
import bigframes.core.ordering as orderings
import bigframes.core.window_spec as window_spec
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops

# a shuffle is a key assignment (random or not), followed by a


def rewrite_random_sample(node: nodes.BigFrameNode) -> nodes.BigFrameNode:
    if not isinstance(node, nodes.RandomSampleNode):
        return node
    if node.seed is not None:
        return hash_based_sample(node.child, node.fraction, node.shuffle, node.seed)
    else:
        return rand_based_sample(node.child, node.fraction, node.shuffle)


def rand_based_sample(
    root: nodes.BigFrameNode, fraction: float, shuffle: bool
) -> nodes.BigFrameNode:
    """
    Shuffle data using volatile random uuid generation
    """
    # Note: Using uuid rather than float rand() to avoid collisions at high cardinalities
    rand_col_id = ids.ColumnId.unique()
    result: nodes.BigFrameNode = nodes.ProjectionNode(
        root, ((ops.gen_uuid_op.as_expr(), rand_col_id),)
    )
    if fraction < 1:
        result = nodes.FilterNode(
            result, ops.lt_op.as_expr(ex.DerefOp(rand_col_id), ex.const(fraction))
        )
    if shuffle:
        result = nodes.OrderByNode(
            result, by=(orderings.ascending_over(rand_col_id),), is_total_order=True
        )
    return nodes.SelectionNode(
        result, tuple(nodes.AliasedRef.identity(id) for id in root.ids)
    )


MAX_DENOM = 2**58


def hash_based_sample(
    root: nodes.BigFrameNode, fraction: float, shuffle: bool, seed: int
) -> nodes.BigFrameNode:
    """
    Shuffle data using hash of row contents

    Sort_key = hash(rowhash, dupe_count, seed)
    """
    rowhash_col_id = ids.ColumnId.unique()
    dupe_count_col_id = ids.ColumnId.unique()
    unique_row_id_col_id = ids.ColumnId.unique()
    # Note rowhash is actually does two different hashes in order to get 128 bits, for collision-avoidance
    result: nodes.BigFrameNode = nodes.ProjectionNode(
        root,
        (
            (
                ops.RowHash().as_expr(*(ex.DerefOp(id) for id in root.ids)),
                rowhash_col_id,
            ),
        ),
    )

    dupe_count_expr = agg_ex.NullaryAggregation(agg_ops.RowNumberOp())
    dupe_count_window = window_spec.WindowSpec((ex.DerefOp(rowhash_col_id),))

    result = nodes.WindowOpNode(
        result,
        (nodes.ColumnDef(dupe_count_expr, dupe_count_col_id),),
        dupe_count_window,
    )

    uniq_row_id = ops.RowHash().as_expr(
        ex.DerefOp(rowhash_col_id), ex.DerefOp(dupe_count_col_id), ex.const(seed)
    )
    result = nodes.ProjectionNode(root, ((uniq_row_id, unique_row_id_col_id),))

    if fraction < 1:
        # The filtering is correlated with the ordering, but thats fine because the ordering is pseudo-random
        true_fraction = fractions.Fraction.from_float(fraction).limit_denominator(
            MAX_DENOM
        )
        modulo_expr = ops.lt_op.as_expr(
            ops.mod_op.as_expr(
                ex.DerefOp(unique_row_id_col_id), ex.const(true_fraction.denominator)
            ),
            ex.const(true_fraction.numerator),
        )
        result = nodes.FilterNode(result, modulo_expr)
    if shuffle:
        result = nodes.OrderByNode(
            result,
            by=(orderings.ascending_over(unique_row_id_col_id),),
            is_total_order=True,
        )
    return nodes.SelectionNode(
        result, tuple(nodes.AliasedRef.identity(id) for id in root.ids)
    )
