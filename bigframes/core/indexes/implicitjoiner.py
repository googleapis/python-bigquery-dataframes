"""Classes that handle row labels, ordering, and implicit joins."""

from __future__ import annotations

import functools
import typing
from typing import Callable, Tuple

import ibis
import ibis.expr.types as ibis_types

if typing.TYPE_CHECKING:
    from bigframes.core import BigFramesExpr


class ImplicitJoiner:
    """Allow implicit joins without row labels on related table expressions."""

    def __init__(self, expr: BigFramesExpr):
        self._expr = expr

    def copy(self) -> ImplicitJoiner:
        """Make a copy of this object."""
        return ImplicitJoiner(self._expr)

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
        if how not in ["outer", "left"]:
            raise NotImplementedError(
                "Only how='outer' and how='left' currently supported"
            )

        # TODO(swast): Allow different expressions in the cases where we can
        # emulate the desired kind of join.
        # TODO(swast): How will our validation change when we allow for mutable
        # cells and inplace methods?
        if self._expr.table != other._expr.table:
            # TODO(swast): Raise a more specific exception (subclass of
            # ValueError, though) to make it easier to catch only the intended
            # failures.
            raise ValueError(
                "Cannot combine objects without an explicit join/merge key. "
                f"Left based on: {self._expr.table.compile()}, but "
                f"right based on: {other._expr.table.compile()}"
            )

        left_expr = self._expr
        right_expr = other._expr
        combined_expr = left_expr.builder()

        left_predicates = left_expr._predicates
        right_predicates = right_expr._predicates
        # TODO(tbergeron): Skip generating these for inner part of join
        (
            left_relative_predicates,
            right_relative_predicates,
        ) = _get_relative_predicates(left_predicates, right_predicates)

        if left_predicates or right_predicates:
            joined_predicates = _join_predicates(
                self._expr.predicates, other._expr.predicates, join_type=how
            )
            combined_expr.predicates = list(
                joined_predicates
            )  # builder expects mutable list

        def get_column_left(key: str) -> ibis_types.Value:
            if left_relative_predicates and how in ["right", "outer"]:
                left_reduce_rel_pred = _reduce_predicate_list(left_relative_predicates)
                return (
                    ibis.case()
                    .when(left_reduce_rel_pred, left_expr.get_column(key))
                    .else_(ibis.null())
                    .end()
                )
            else:
                return left_expr.get_column(key)

        def get_column_right(key: str) -> ibis_types.Value:
            if right_relative_predicates and how in ["left", "outer"]:
                right_reduce_rel_pred = _reduce_predicate_list(
                    right_relative_predicates
                )
                return (
                    ibis.case()
                    .when(right_reduce_rel_pred, right_expr.get_column(key))
                    .else_(ibis.null())
                    .end()
                )
            else:
                return right_expr.get_column(key)

        return ImplicitJoiner(combined_expr.build()), (
            get_column_left,
            get_column_right,
        )


def _join_predicates(
    left_predicates: typing.Collection[ibis_types.BooleanValue],
    right_predicates: typing.Collection[ibis_types.BooleanValue],
    join_type: str = "outer",
) -> typing.Tuple[ibis_types.BooleanValue, ...]:
    """Combines predicates lists for each side of a join."""
    if join_type == "outer":
        if not left_predicates:
            return ()
        if not right_predicates:
            return ()
        # TODO(tbergeron): Investigate factoring out common predicates
        joined_predicates = _reduce_predicate_list(left_predicates).__or__(
            _reduce_predicate_list(right_predicates)
        )
        return (joined_predicates,)
    if join_type == "left":
        return tuple(left_predicates)
    else:
        raise ValueError("Unsupported join_type: " + join_type)


def _get_relative_predicates(
    left_predicates: typing.Collection[ibis_types.BooleanValue],
    right_predicates: typing.Collection[ibis_types.BooleanValue],
) -> tuple[
    typing.Tuple[ibis_types.BooleanValue, ...],
    typing.Tuple[ibis_types.BooleanValue, ...],
]:
    """Get predicates that apply to only one side of the join. Not strictly necessary but simplifies resulting query."""
    left_relative_predicates = tuple(left_predicates) or ()
    right_relative_predicates = tuple(right_predicates) or ()
    if left_predicates and right_predicates:
        # Factor out common predicates needed for left/right column masking
        left_relative_predicates = tuple(set(left_predicates) - set(right_predicates))
        right_relative_predicates = tuple(set(right_predicates) - set(left_predicates))
    return (left_relative_predicates, right_relative_predicates)


def _reduce_predicate_list(
    predicate_list: typing.Collection[ibis_types.BooleanValue],
) -> ibis_types.BooleanValue:
    """Converts a list of predicates BooleanValues into a single BooleanValue."""
    if len(predicate_list) == 0:
        raise ValueError("Cannot reduce empty list of predicates")
    if len(predicate_list) == 1:
        (item,) = predicate_list
        return item
    return functools.reduce(lambda acc, pred: acc.__and__(pred), predicate_list)
