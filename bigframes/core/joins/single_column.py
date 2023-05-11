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

"""Helpers to join BigFramesExpr objects."""

from __future__ import annotations

from typing import Callable, Literal, Tuple

import ibis
import ibis.expr.types as ibis_types

import bigframes.core as core
import bigframes.core.joins.row_identity
import bigframes.core.ordering
import bigframes.guid


def join_by_column(
    left: core.BigFramesExpr,
    left_column_id: str,
    right: core.BigFramesExpr,
    right_column_id: str,
    *,
    how: Literal[
        "inner",
        "left",
        "outer",
        "right",
    ],
    sort: bool = False,
) -> Tuple[
    core.BigFramesExpr,
    str,
    Tuple[Callable[[str], ibis_types.Value], Callable[[str], ibis_types.Value]],
]:
    """Join two expressions by column equality.

    Arguments:
        left: Expression for left table to join.
        left_column_id: Column ID (not label) to join by.
        right: Expression for right table to join.
        right_column_id: Column ID (not label) to join by.
        how: The type of join to perform.

    Returns:
        The joined expression and the objects needed to interpret it.

        * BigFramesExpr: Joined table with all columns from left and right.
        * str: Column ID of the coalesced join column. Sometimes either the
          left/right table will have missing rows. This column pulls the
          non-NULL value from either left/right.
        * Tuple[Callable, Callable]: For a given column ID from left or right,
          respectively, return the new column from the combined expression.
    """

    if (
        how in bigframes.core.joins.row_identity.SUPPORTED_ROW_IDENTITY_HOW
        and left.table.equals(right.table)
        # Compare ibis expressions for left/right column because its possible that
        # they both have the same name but were modified in different ways.
        and left.get_any_column(left_column_id).equals(
            right.get_any_column(right_column_id)
        )
    ):
        combined_expr, (
            get_column_left,
            get_column_right,
        ) = bigframes.core.joins.row_identity.join_by_row_identity(left, right, how=how)
        original_ordering = combined_expr._ordering
        new_order_id = original_ordering.ordering_id if original_ordering else None
    else:
        # Generate offsets if non-default ordering is applied
        # Assumption, both sides are totally ordered, otherwise offsets will be nondeterministic
        left_table = left.to_ibis_expr(
            ordering_mode="ordered_col", order_col_name=core.ORDER_ID_COLUMN
        )
        left_index = left_table[left_column_id]
        right_table = right.to_ibis_expr(
            ordering_mode="ordered_col", order_col_name=core.ORDER_ID_COLUMN
        )
        right_index = right_table[right_column_id]
        join_condition = left_index == right_index

        # TODO(swast): Handle duplicate column names with suffixs, see "merge"
        # in DaPandas.
        combined_table = ibis.join(
            left_table, right_table, predicates=join_condition, how=how
        )

        def get_column_left(key: str) -> ibis_types.Value:
            if how == "inner" and key == left_column_id:
                # Don't rename the column if it's the index on an inner
                # join.
                pass
            elif key in right_table.columns:
                key = f"{key}_x"

            return combined_table[key]

        def get_column_right(key: str) -> ibis_types.Value:
            if how == "inner" and key == right_column_id:
                # Don't rename the column if it's the index on an inner
                # join.
                pass
            elif key in left_table.columns:
                key = f"{key}_y"

            return combined_table[key]

        left_ordering_encoding_size = (
            left._ordering.ordering_encoding_size
            or bigframes.core.ordering.DEFAULT_ORDERING_ID_LENGTH
        )
        right_ordering_encoding_size = (
            right._ordering.ordering_encoding_size
            or bigframes.core.ordering.DEFAULT_ORDERING_ID_LENGTH
        )

        # Preserve original ordering accross joins.
        left_order_id = get_column_left(core.ORDER_ID_COLUMN)
        right_order_id = get_column_right(core.ORDER_ID_COLUMN)
        new_order_id_col = _merge_order_ids(
            left_order_id,
            left_ordering_encoding_size,
            right_order_id,
            right_ordering_encoding_size,
            how,
        )
        new_order_id = new_order_id_col.get_name()
        if new_order_id is None:
            raise ValueError("new_order_id unexpectedly has no name")
        metadata_columns = (new_order_id_col,)
        original_ordering = core.ExpressionOrdering(
            ordering_id_column=core.OrderingColumnReference(new_order_id)
            if (new_order_id_col is not None)
            else None,
            ordering_encoding_size=left_ordering_encoding_size
            + right_ordering_encoding_size,
        )
        combined_expr = core.BigFramesExpr(
            left._session,
            combined_table,
            meta_columns=metadata_columns,
        )

    join_key_col = (
        # The left index and the right index might contain null values, for
        # example due to an outer join with different numbers of rows. Coalesce
        # these to take the index value from either column.
        ibis.coalesce(
            get_column_left(left_column_id),
            get_column_right(right_column_id),
        )
        # Use a random name in case the left index and the right index have the
        # same name. In such a case, _x and _y suffixes will already be used.
        .name(bigframes.guid.generate_guid(prefix="index_"))
    )

    # We could filter out the original join columns, but predicates/ordering
    # might still reference them in implicit joins.
    columns = (
        [join_key_col]
        + [get_column_left(key) for key in left.column_names.keys()]
        + [get_column_right(key) for key in right.column_names.keys()]
    )

    if sort:
        ordering = original_ordering.with_ordering_columns(
            [core.OrderingColumnReference(join_key_col.get_name())]
        )
    else:
        ordering = original_ordering

    combined_expr_builder = combined_expr.builder()
    combined_expr_builder.columns = columns
    combined_expr_builder.ordering = ordering
    combined_expr = combined_expr_builder.build()
    return (
        combined_expr,
        join_key_col.get_name(),
        (get_column_left, get_column_right),
    )


def _merge_order_ids(
    left_id: ibis_types.Value,
    left_encoding_size: int,
    right_id: ibis_types.Value,
    right_encoding_size: int,
    how: str,
) -> ibis_types.StringValue:
    if how == "right":
        return _merge_order_ids(
            right_id, right_encoding_size, left_id, left_encoding_size, "left"
        )
    return (
        (
            bigframes.core.ordering.stringify_order_id(left_id, left_encoding_size)
            + bigframes.core.ordering.stringify_order_id(right_id, right_encoding_size)
        )
    ).name(bigframes.guid.generate_guid(prefix="bigframes_ordering_id_"))
