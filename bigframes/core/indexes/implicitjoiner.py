"""Classes that handle row labels, ordering, and implicit joins."""

from __future__ import annotations

import typing
from typing import Callable, Tuple

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

        def get_column_left(key: str) -> ibis_types.Value:
            # TODO(swast): We can emulate an outer join for predicates using
            # something like the following:
            # return ibis.case().when(left_has_value, column).else_(ibis.null()).end()
            return combined_table[key]

        def get_column_right(key: str) -> ibis_types.Value:
            # TODO(swast): We can emulate an outer join for predicates using
            # something like the following:
            # return ibis.case().when(right_has_value, column).else_(ibis.null()).end()
            return combined_table[key]

        return ImplicitJoiner(combined_expr.build()), (
            get_column_left,
            get_column_right,
        )
