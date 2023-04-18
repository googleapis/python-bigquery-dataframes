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

"""Classes that handle row labels, ordering, and implicit joins."""

from __future__ import annotations

import functools
import typing
from typing import Callable, Tuple

import ibis
import ibis.expr.types as ibis_types

import bigframes.core as core
import bigframes.core.blocks as blocks


class ImplicitJoiner:
    """Allow implicit joins without row labels on related table expressions."""

    def __init__(self, block: blocks.Block, name: typing.Optional[str] = None):
        self._block = block
        self._name = name

    def copy(self) -> ImplicitJoiner:
        """Make a copy of this object."""
        # TODO(swast): Should this make a copy of block?
        return ImplicitJoiner(self._block, self._name)

    @property
    def _expr(self) -> core.BigFramesExpr:
        return self._block.expr

    @property
    def name(self) -> typing.Optional[str]:
        """Name of the Index."""
        # This introduces a level of indirection over Ibis to allow for more
        # accurate pandas behavior (such as allowing for unnamed or
        # non-uniquely named objects) without breaking SQL generation.
        return self._name

    @name.setter
    def name(self, value: typing.Optional[str]):
        self._name = value

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
        if how not in {"outer", "left", "inner"}:
            raise NotImplementedError(
                "Only how='outer','left','inner' currently supported"
            )

        # TODO(swast): Allow different expressions in the cases where we can
        # emulate the desired kind of join.
        # TODO(swast): How will our validation change when we allow for mutable
        # cells and inplace methods?
        if not self._expr.table.equals(other._expr.table):
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

        left_predicates = left_expr._predicates
        right_predicates = right_expr._predicates
        # TODO(tbergeron): Skip generating these for inner part of join
        (
            left_relative_predicates,
            right_relative_predicates,
        ) = _get_relative_predicates(left_predicates, right_predicates)

        combined_predicates = []
        if left_predicates or right_predicates:
            joined_predicates = _join_predicates(
                left_predicates, right_predicates, join_type=how
            )
            combined_predicates = list(
                joined_predicates
            )  # builder expects mutable list

        left_mask = left_relative_predicates if how in ["right", "outer"] else None
        right_mask = right_relative_predicates if how in ["left", "outer"] else None
        joined_columns = [
            _mask_value(left_expr.get_column(key), left_mask).name(map_left_id(key))
            for key in left_expr.column_names.keys()
        ] + [
            _mask_value(right_expr.get_column(key), right_mask).name(map_right_id(key))
            for key in right_expr.column_names.keys()
        ]

        new_ordering = core.ExpressionOrdering()
        if left_expr._ordering and right_expr._ordering:
            meta_columns = [
                left_expr.get_any_column(key).name(map_left_id(key))
                for key in left_expr._meta_column_names.keys()
            ] + [
                right_expr.get_any_column(key).name(map_right_id(key))
                for key in right_expr._meta_column_names.keys()
            ]
            new_ordering_id = (
                map_left_id(left_expr._ordering.ordering_id)
                if (left_expr._ordering.ordering_id)
                else None
            )
            new_ordering = core.ExpressionOrdering(
                ordering_value_columns=(
                    [
                        map_left_id(key)
                        for key in left_expr._ordering.ordering_value_columns
                    ]
                    + [
                        map_right_id(key)
                        for key in right_expr._ordering.ordering_value_columns
                    ]
                ),
                ordering_id_column=new_ordering_id,
                ascending=left_expr._ordering.is_ascending,
            )

        joined_expr = core.BigFramesExpr(
            left_expr._session,
            left_expr.table,
            columns=joined_columns,
            meta_columns=meta_columns,
            ordering=new_ordering,
            predicates=combined_predicates,
        )
        block = blocks.Block(joined_expr)
        return ImplicitJoiner(block, name=self.name), (
            lambda key: joined_expr.get_any_column(map_left_id(key)),
            lambda key: joined_expr.get_any_column(map_right_id(key)),
        )


def map_left_id(left_side_id):
    return f"{left_side_id}_x"


def map_right_id(right_side_id):
    return f"{right_side_id}_y"


def _mask_value(
    value: ibis_types.Value,
    predicates: typing.Optional[typing.Sequence[ibis_types.BooleanValue]] = None,
):
    if predicates:
        return (
            ibis.case()
            .when(_reduce_predicate_list(predicates), value)
            .else_(ibis.null())
            .end()
        )
    return value


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
    if join_type == "inner":
        _, right_relative_predicates = _get_relative_predicates(
            left_predicates, right_predicates
        )
        return (*left_predicates, *right_relative_predicates)
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
