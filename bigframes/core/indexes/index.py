"""An index based on a single column."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import ibis
import ibis.expr.types as ibis_types

from bigframes.core import BigFramesExpr
from bigframes.core.indexes.implicitjoiner import ImplicitJoiner


class Index(ImplicitJoiner):
    """An index based on a single column."""

    # TODO(swast): Handle more than 1 index column, possibly in a separate
    # MultiIndex class.
    # TODO(swast): Include ordering here?
    def __init__(
        self, expr: BigFramesExpr, index_column: str, name: Optional[str] = None
    ):
        index_name = name if name is not None else index_column
        super().__init__(expr, index_name)
        self._index_column = index_column

    def copy(self) -> Index:
        """Make a copy of this object."""
        return Index(self._expr, self._index_column, name=self.name)

    def join(
        self,
        other: ImplicitJoiner,
        *,
        how="left",
    ) -> Tuple[
        ImplicitJoiner,
        Tuple[Callable[[str], ibis_types.Value], Callable[[str], ibis_types.Value]],
    ]:
        try:
            # TOOD(swast): We need to check that the indexes are the same
            # (including ordering) before falling back to row identity
            # matching. Though maybe the index itself will validate that?
            joined, column_getters = super().join(other, how=how)
            return Index(joined._expr, self._index_column), column_getters
        except (ValueError, NotImplementedError):
            # TODO(swast): Catch a narrower exception than ValueError.
            # If the more efficient implicit join can't be performed, try an explicit join.
            pass

        if not isinstance(other, Index):
            # TODO(swast): We need to improve this error message to be more
            # actionable for the user. For example, it's possible they
            # could call set_index and try again to resolve this error.
            raise ValueError(
                "Can't mixed objects with explicit Index and ImpliedJoiner"
            )

        # TODO(swast): Consider refactoring to allow re-use in cases where an
        # explicit join key is used.
        left_table = self._expr.to_ibis_expr(order_results=False)
        left_index = left_table[self._index_column]
        right_table = other._expr.to_ibis_expr(order_results=False)
        right_index = right_table[other._index_column]
        join_condition = left_index == right_index

        index_name_orig = self._index_column
        index_name = index_name_orig
        if index_name in right_table.columns:
            index_name = f"{index_name}_x"

        # TODO(swast): Handle duplicate column names with suffixs, see "merge"
        # in DaPandas.
        combined_table = ibis.join(
            left_table, right_table, predicates=join_condition, how=how
        )
        combined_expr = BigFramesExpr(self._expr._session, combined_table)

        # Always sort by the join key. Note: This differs from pandas, in which
        # the sort order can differ unless explicitly sorted with sort=True.
        index_value = combined_table[index_name].name(index_name_orig)
        combined_expr = combined_expr.order_by([index_value])

        def get_column_left(key: str) -> ibis_types.Value:
            if key in right_table.columns:
                key = f"{key}_x"

            return combined_table[key]

        def get_column_right(key: str) -> ibis_types.Value:
            if key in left_table.columns:
                key = f"{key}_y"

            return combined_table[key]

        return (
            Index(combined_expr, index_name, name=self._name),
            (get_column_left, get_column_right),
        )
