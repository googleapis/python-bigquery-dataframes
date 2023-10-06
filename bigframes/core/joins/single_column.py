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

"""Helpers to join ArrayValue objects."""

from __future__ import annotations

import typing
from typing import Literal, Mapping

import ibis
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types

import bigframes.constants as constants
import bigframes.core as core
import bigframes.core.guid as guid
import bigframes.core.joins.name_resolution as naming
import bigframes.core.joins.row_identity
import bigframes.core.ordering


def join_by_column(
    left: core.ArrayValue,
    left_column_ids: typing.Sequence[str],
    right: core.ArrayValue,
    right_column_ids: typing.Sequence[str],
    *,
    how: Literal[
        "inner",
        "left",
        "outer",
        "right",
    ],
    allow_row_identity_join: bool = True,
) -> core.ArrayValue:
    """Join two expressions by column equality.

    Arguments:
        left: Expression for left table to join.
        left_column_ids: Column IDs (not label) to join by.
        right: Expression for right table to join.
        right_column_ids: Column IDs (not label) to join by.
        how: The type of join to perform.
        allow_row_identity_join (bool):
            If True, allow matching by row identity. Set to False to always
            perform a true JOIN in generated SQL.
    Returns:
        The joined expression. The resulting columns will be, in order,
        first the coalesced join keys, then, all the left columns, and
        finally, all the right columns.
    """
    # Value column mapping must use JOIN_NAME_REMAPPER to stay in sync with consumers of join result
    lmapping, rmapping = naming.JOIN_NAME_REMAPPER(left.column_ids, right.column_ids)
    if (
        allow_row_identity_join
        and how in bigframes.core.joins.row_identity.SUPPORTED_ROW_IDENTITY_HOW
        and left._table.equals(right._table)
        # Make sure we're joining on exactly the same column(s), at least with
        # regards to value its possible that they both have the same names but
        # were modified in different ways. Ignore differences in the names.
        and all(
            left._get_any_column(lcol)
            .name("index")
            .equals(right._get_any_column(rcol).name("index"))
            for lcol, rcol in zip(left_column_ids, right_column_ids)
        )
    ):
        return bigframes.core.joins.row_identity.join_by_row_identity(
            left, right, how=how
        )
    else:
        lhiddenmapping, rhiddenmapping = naming.JoinNameRemapper(namespace="hidden")(
            left._hidden_column_ids, right._hidden_column_ids
        )

        left_table = left._to_ibis_expr(
            "unordered",
            expose_hidden_cols=True,
            col_id_overrides={**lmapping, **lhiddenmapping},
        )
        right_table = right._to_ibis_expr(
            "unordered",
            expose_hidden_cols=True,
            col_id_overrides={**rmapping, **rhiddenmapping},
        )
        join_conditions = [
            value_to_join_key(left_table[lmapping[left_index]])
            == value_to_join_key(right_table[rmapping[right_index]])
            for left_index, right_index in zip(left_column_ids, right_column_ids)
        ]

        combined_table = ibis.join(
            left_table,
            right_table,
            predicates=join_conditions,
            how=how,
        )

        # Preserve ordering accross joins.
        ordering = join_orderings(
            left._ordering,
            right._ordering,
            {**lmapping, **lhiddenmapping},
            {**rmapping, **rhiddenmapping},
            left_order_dominates=(how != "right"),
        )

        # We could filter out the original join columns, but predicates/ordering
        # might still reference them in implicit joins.
        columns = [combined_table[lmapping[col.get_name()]] for col in left.columns] + [
            combined_table[rmapping[col.get_name()]] for col in right.columns
        ]
        hidden_ordering_columns = [
            *[
                combined_table[lhiddenmapping[col.get_name()]]
                for col in left._hidden_ordering_columns
            ],
            *[
                combined_table[rhiddenmapping[col.get_name()]]
                for col in right._hidden_ordering_columns
            ],
        ]
        return core.ArrayValue(
            left._session,
            combined_table,
            columns=columns,
            hidden_ordering_columns=hidden_ordering_columns,
            ordering=ordering,
        )


def get_coalesced_join_cols(
    left_join_cols: typing.Iterable[ibis_types.Value],
    right_join_cols: typing.Iterable[ibis_types.Value],
    how: str,
) -> typing.List[ibis_types.Value]:
    join_key_cols: list[ibis_types.Value] = []
    for left_col, right_col in zip(left_join_cols, right_join_cols):
        if how == "left" or how == "inner":
            join_key_cols.append(left_col.name(guid.generate_guid(prefix="index_")))
        elif how == "right":
            join_key_cols.append(right_col.name(guid.generate_guid(prefix="index_")))
        elif how == "outer":
            # The left index and the right index might contain null values, for
            # example due to an outer join with different numbers of rows. Coalesce
            # these to take the index value from either column.
            # Use a random name in case the left index and the right index have the
            # same name. In such a case, _x and _y suffixes will already be used.
            # Don't need to coalesce if they are exactly the same column.
            if left_col.name("index").equals(right_col.name("index")):
                join_key_cols.append(left_col.name(guid.generate_guid(prefix="index_")))
            else:
                join_key_cols.append(
                    ibis.coalesce(
                        left_col,
                        right_col,
                    ).name(guid.generate_guid(prefix="index_"))
                )
        else:
            raise ValueError(f"Unexpected join type: {how}. {constants.FEEDBACK_LINK}")
    return join_key_cols


def value_to_join_key(value: ibis_types.Value):
    """Converts nullable values to non-null string SQL will not match null keys together - but pandas does."""
    if not value.type().is_string():
        value = value.cast(ibis_dtypes.str)
    return value.fillna(ibis_types.literal("$NULL_SENTINEL$"))


def join_orderings(
    left: core.ExpressionOrdering,
    right: core.ExpressionOrdering,
    left_id_mapping: Mapping[str, str],
    right_id_mapping: Mapping[str, str],
    left_order_dominates: bool = True,
) -> core.ExpressionOrdering:
    left_ordering_refs = [
        ref.with_name(left_id_mapping[ref.column_id])
        for ref in left.all_ordering_columns
    ]
    right_ordering_refs = [
        ref.with_name(right_id_mapping[ref.column_id])
        for ref in right.all_ordering_columns
    ]
    if left_order_dominates:
        joined_refs = [*left_ordering_refs, *right_ordering_refs]
    else:
        joined_refs = [*right_ordering_refs, *left_ordering_refs]

    left_total_order_cols = frozenset(
        [left_id_mapping[id] for id in left.total_ordering_columns]
    )
    right_total_order_cols = frozenset(
        [right_id_mapping[id] for id in right.total_ordering_columns]
    )
    return core.ExpressionOrdering(
        ordering_value_columns=joined_refs,
        total_ordering_columns=left_total_order_cols | right_total_order_cols,
    )
