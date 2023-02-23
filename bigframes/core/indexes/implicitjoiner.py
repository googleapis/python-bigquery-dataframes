"""Classes that handle row labels, ordering, and implicit joins."""

from __future__ import annotations

import typing
from typing import Callable, Optional, Tuple

import ibis
import ibis.expr.types as ibis_types

if typing.TYPE_CHECKING:
    from bigframes.core import BigFramesExpr


class ImplicitJoiner:
    """Allow implicit joins without row labels on related table expressions."""

    def __init__(self, expr: BigFramesExpr):
        self._expr = expr

    # TODO(swast): In pandas, "left_indexer" and "right_indexer" are numpy
    # arrays that indicate where the rows line up. Do we want to wrap ibis to
    # emulate arrays? How might this change if we're doing a real join on the
    # respective table expressions? See:
    # https://pandas.pydata.org/docs/reference/api/pandas.Index.get_indexer.html
    def join(
        self,
        other: ImplicitJoiner,
        *,
        how="left",
    ) -> Tuple[
        ImplicitJoiner,
        Tuple[Callable[[str], ibis_types.Value], Callable[[str], ibis_types.Value]],
    ]:
        """Compute join_index and indexers to conform data structures to the new index."""
        if how != "outer":
            raise NotImplementedError("Only how='outer' currently supported")

        # TODO(swast): Allow different expressions in the cases where we can
        # emulate the desired kind of join.
        # TODO(swast): How will our validation change when we allow for mutable
        # cells and inplace methods?
        if self._expr.table != other._expr.table:
            raise ValueError(
                "Cannot combine objects without an explicit join/merge key. "
                f"Left based on: {self._expr.table.compile()}, but "
                f"right based on: {other._expr.table.compile()}"
            )

        combined_table = self._expr.table

        if self._expr._predicates != other._expr._predicates:
            # TODO(tbergeron): Implement join on tables with predicates.
            raise NotImplementedError(
                "Join not yet supported on differently filtered tables."
            )

        combined_expr = self._expr.builder()
        # TODO(swast): We assume an outer join, but the pandas default is actually a left join.
        combined_limit, (left_has_value, right_has_value) = _outer_join_limits(
            self._expr, other._expr
        )
        combined_expr.limit = combined_limit

        def get_column_left(key: str) -> ibis_types.Value:
            column = combined_table[key]
            # TODO(swast): We can avoid the case statement if left_has_value is always True.
            return ibis.case().when(left_has_value, column).else_(ibis.null()).end()

        def get_column_right(key: str) -> ibis_types.Value:
            column = combined_table[key]
            # TODO(swast): We can avoid the case statement if right_has_value is always True.
            return ibis.case().when(right_has_value, column).else_(ibis.null()).end()

        return ImplicitJoiner(combined_expr.build()), (
            get_column_left,
            get_column_right,
        )


def _outer_join_limits(
    left_expr, right_expr
) -> Tuple[Optional[int], Tuple[ibis_types.BooleanValue, ibis_types.BooleanValue]]:
    left_limit = left_expr.limit
    right_limit = right_expr.limit

    if left_limit is None or right_limit is None:
        combined_limit = None
    else:
        combined_limit = max(left_limit, right_limit)

    if left_limit is None:
        left_has_values = typing.cast(ibis_types.BooleanValue, ibis.literal(True))
    else:
        # ibis.row_number() starts at 0, not 1 like native BigQuery.
        left_has_values = ibis.row_number() < left_limit

    if right_limit is None:
        right_has_values = typing.cast(ibis_types.BooleanValue, ibis.literal(True))
    else:
        right_has_values = ibis.row_number() < right_limit

    return combined_limit, (left_has_values, right_has_values)
