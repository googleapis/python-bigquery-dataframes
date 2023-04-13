"""An index based on a single column."""

from __future__ import annotations

import typing
from typing import Callable, Optional, Tuple

import ibis
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types

from bigframes.core import BigFramesExpr, ExpressionOrdering, ORDER_ID_COLUMN
from bigframes.core.indexes.implicitjoiner import ImplicitJoiner
import bigframes.guid


class Index(ImplicitJoiner):
    """An index based on a single column."""

    # TODO(swast): Handle more than 1 index column, possibly in a separate
    # MultiIndex class.
    # TODO(swast): Include ordering here?
    def __init__(
        self, expr: BigFramesExpr, index_column: str, name: Optional[str] = None
    ):
        super().__init__(expr, name=name)
        self._index_column = index_column

    def copy(self) -> Index:
        """Make a copy of this object."""
        return Index(self._expr, self._index_column, name=self.name)

    def join(
        self, other: ImplicitJoiner, *, how="left", sort=False
    ) -> Tuple[
        ImplicitJoiner,
        Tuple[Callable[[str], ibis_types.Value], Callable[[str], ibis_types.Value]],
    ]:
        if not isinstance(other, Index):
            # TODO(swast): We need to improve this error message to be more
            # actionable for the user. For example, it's possible they
            # could call set_index and try again to resolve this error.
            raise ValueError(
                "Can't mixed objects with explicit Index and ImpliedJoiner"
            )

        # TODO(swast): Support cross-joins (requires reindexing).
        if how not in {"outer", "left", "right", "inner"}:
            raise NotImplementedError(
                "Only how='outer','left','right','inner' currently supported"
            )

        try:
            # TOOD(swast): We need to check that the indexes are the same
            # before falling back to row identity matching.
            combined_joiner, (get_column_left, get_column_right) = super().join(
                other, how=how
            )
            combined_expr = combined_joiner._expr
            original_ordering = combined_joiner._expr.builder().ordering
            new_order_id = original_ordering.ordering_id if original_ordering else None
        except (ValueError, NotImplementedError):
            # TODO(swast): Catch a narrower exception than ValueError.
            # If the more efficient implicit join can't be performed, try an explicit join.

            # TODO(swast): Consider refactoring to allow re-use in cases where an
            # explicit join key is used.

            # Generate offsets if non-default ordering is applied
            # Assumption, both sides are totally ordered, otherwise offsets will be nondeterministic
            left_table = self._expr.to_ibis_expr(
                ordering_mode="ordered_col", order_col_name=ORDER_ID_COLUMN
            )
            left_index = left_table[self._index_column]
            right_table = other._expr.to_ibis_expr(
                ordering_mode="ordered_col", order_col_name=ORDER_ID_COLUMN
            )
            right_index = right_table[other._index_column]
            join_condition = left_index == right_index

            # TODO(swast): Handle duplicate column names with suffixs, see "merge"
            # in DaPandas.
            combined_table = ibis.join(
                left_table, right_table, predicates=join_condition, how=how
            )

            def get_column_left(key: str) -> ibis_types.Value:
                if how == "inner" and key == self._index_column:
                    # Don't rename the column if it's the index on an inner
                    # join.
                    pass
                elif key in right_table.columns:
                    key = f"{key}_x"

                return combined_table[key]

            def get_column_right(key: str) -> ibis_types.Value:
                if how == "inner" and key == typing.cast(Index, other)._index_column:
                    # Don't rename the column if it's the index on an inner
                    # join.
                    pass
                elif key in left_table.columns:
                    key = f"{key}_y"

                return combined_table[key]

            # Preserve original ordering accross joins.
            left_order_id = get_column_left(ORDER_ID_COLUMN)
            right_order_id = get_column_right(ORDER_ID_COLUMN)
            new_order_id_col = (
                _merge_order_ids(left_order_id, right_order_id)
                if how in ["left", "inner", "outer"]
                else _merge_order_ids(right_order_id, left_order_id)
            )
            new_order_id = new_order_id_col.get_name()
            metadata_columns = (new_order_id_col,)
            original_ordering = ExpressionOrdering(
                ordering_id_column=new_order_id
                if (new_order_id_col is not None)
                else None,
            )
            combined_expr = BigFramesExpr(
                self._expr._session,
                combined_table,
                meta_columns=metadata_columns,
            )

        index_name_orig = self._index_column

        joined_index_col = (
            # The left index and the right index might contain null values, for
            # example due to an outer join with different numbers of rows. Coalesce
            # these to take the index value from either column.
            ibis.coalesce(
                get_column_left(self._index_column),
                get_column_right(other._index_column),
            )
            # Add a suffix in case the left index and the right index have the
            # same name. In such a case, _x and _y suffixes will already be
            # used.
            .name(index_name_orig + "_z")
        )

        # TODO: Can actually ignore original index values post-join
        columns = (
            [joined_index_col]
            + [get_column_left(key) for key in self._expr.column_names.keys()]
            + [get_column_right(key) for key in other._expr.column_names.keys()]
        )

        if sort:
            order_cols = [joined_index_col.get_name()]
            ordering: Optional[ExpressionOrdering] = ExpressionOrdering(
                ordering_value_columns=order_cols,
                ordering_id_column=new_order_id if (new_order_id is not None) else None,
            )
        else:
            ordering = original_ordering

        combined_expr_builder = combined_expr.builder()
        combined_expr_builder.columns = columns
        combined_expr_builder.ordering = ordering
        combined_expr = combined_expr_builder.build()
        combined_index_name = self.name if self.name == other.name else None
        return (
            Index(combined_expr, joined_index_col.get_name(), name=combined_index_name),
            (get_column_left, get_column_right),
        )


def _merge_order_ids(left_id: ibis_types.Value, right_id: ibis_types.Value):
    return ((_stringify_order_id(left_id) + _stringify_order_id(right_id))).name(
        bigframes.guid.generate_guid(prefix="bigframes_ordering_id_")
    )


def _stringify_order_id(order_id: ibis_types.Value) -> ibis_types.StringValue:
    """Conversts an order id value to string if it is not already a string. MUST produced fixed-length strings."""
    if order_id.type().is_int64():
        # This is very inefficient encoding base-10 string uses only 10 characters per byte(out of 256 bit combinations)
        # Furthermore, if know tighter bounds on order id are known, can produce smaller strings.
        # 19 characters chosen as it can represent any positive Int64 in base-10
        # For missing values, ":" * 19 is used as it is larger than any other value this function produces, so null values will be last.
        string_order_id = (
            typing.cast(
                ibis_types.StringValue,
                typing.cast(ibis_types.IntegerValue, order_id).cast(ibis_dtypes.string),
            )
            .lpad(19, "0")
            .fillna(ibis_types.literal(":" * 19))
        )
        return typing.cast(ibis_types.StringValue, string_order_id)
    return typing.cast(ibis_types.StringValue, order_id)
