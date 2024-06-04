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

import ibis
import pandas as pd

import bigframes.core as core
import bigframes.core.expression as ex
import bigframes.core.ordering as bf_order
import bigframes.operations as ops
import bigframes.session.planner as planner

ibis_table = ibis.memtable({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
cols = [ibis_table["col_a"], ibis_table["col_b"]]
LEAF: core.ArrayValue = core.ArrayValue.from_ibis(
    session=None,  # type: ignore
    table=ibis_table,
    columns=cols,
    hidden_ordering_columns=[],
    ordering=bf_order.ExpressionOrdering((bf_order.ascending_over("col_a"),)),
)


def test_session_aware_caching_project_filter():
    """
    Test that if a node is filtered by a column, the node is cached pre-filter and clustered by the filter column.
    """
    session_objects = [LEAF, LEAF.assign_constant("col_c", 4, pd.Int64Dtype())]
    target = LEAF.assign_constant("col_c", 4, pd.Int64Dtype()).filter(
        ops.gt_op.as_expr("col_a", ex.const(3))
    )
    result, cluster_cols = planner.session_aware_cache_plan(
        target.node, [obj.node for obj in session_objects]
    )
    assert result == LEAF.node
    assert cluster_cols == ["col_a"]


def test_session_aware_caching_unusable_filter():
    """
    Test that if a node is filtered by multiple columns in the same comparison, the node is cached pre-filter and not clustered by either column.

    Most filters with multiple column references cannot be used for scan pruning, as they cannot be converted to fixed value ranges.
    """
    session_objects = [LEAF, LEAF.assign_constant("col_c", 4, pd.Int64Dtype())]
    target = LEAF.assign_constant("col_c", 4, pd.Int64Dtype()).filter(
        ops.gt_op.as_expr("col_a", "col_b")
    )
    result, cluster_cols = planner.session_aware_cache_plan(
        target.node, [obj.node for obj in session_objects]
    )
    assert result == LEAF.node
    assert cluster_cols == []


def test_session_aware_caching_fork_after_window_op():
    """
    Test that caching happens only after an windowed operation, but before filtering, projecting.

    Windowing is expensive, so caching should always compute the window function, in order to avoid later recomputation.
    """
    other = LEAF.promote_offsets("offsets_col").assign_constant(
        "col_d", 5, pd.Int64Dtype()
    )
    target = (
        LEAF.promote_offsets("offsets_col")
        .assign_constant("col_c", 4, pd.Int64Dtype())
        .filter(
            ops.eq_op.as_expr("col_a", ops.add_op.as_expr(ex.const(4), ex.const(3)))
        )
    )
    result, cluster_cols = planner.session_aware_cache_plan(
        target.node,
        [
            other.node,
        ],
    )
    assert result == LEAF.promote_offsets("offsets_col").node
    assert cluster_cols == ["col_a"]
