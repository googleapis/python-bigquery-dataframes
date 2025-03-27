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
from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
from typing import Literal, Mapping, Optional, Set, Tuple, Union

import bigframes.core.expression as ex
import bigframes.core.identifiers as ids
import bigframes.core.ordering as orderings


# Unbound Windows
def unbound(
    grouping_keys: Tuple[str, ...] = (),
    min_periods: int = 0,
    ordering: Tuple[orderings.OrderingExpression, ...] = (),
) -> WindowSpec:
    """
    Create an unbound window.

    Args:
        grouping_keys:
            Columns ids of grouping keys
        min_periods (int, default 0):
            Minimum number of input rows to generate output.
        ordering:
            Orders the rows within the window.

    Returns:
        WindowSpec
    """
    return WindowSpec(
        grouping_keys=tuple(map(ex.deref, grouping_keys)),
        min_periods=min_periods,
        ordering=ordering,
    )


### Rows-based Windows
def rows(
    grouping_keys: Tuple[str, ...] = (),
    start: Optional[int] = None,
    end: Optional[int] = None,
    min_periods: int = 0,
    ordering: Tuple[orderings.OrderingExpression, ...] = (),
) -> WindowSpec:
    """
    Create a row-bounded window.

    Args:
        grouping_keys:
            Columns ids of grouping keys
        start:
            The window's starting boundary relative to the current row. For example, "-1" means one row prior
            "1" means one row after, and "0" means the current row. If None, the window is unbounded from the start.
        following:
            The window's ending boundary relative to the current row. For example, "-1" means one row prior
            "1" means one row after, and "0" means the current row. If None, the window is unbounded until the end.
        min_periods (int, default 0):
            Minimum number of input rows to generate output.
        ordering:
            Ordering to apply on top of based dataframe ordering
    Returns:
        WindowSpec
    """
    bounds = RowsWindowBounds(
        start=start,
        end=end,
    )
    return WindowSpec(
        grouping_keys=tuple(map(ex.deref, grouping_keys)),
        bounds=bounds,
        min_periods=min_periods,
        ordering=ordering,
    )


def cumulative_rows(
    grouping_keys: Tuple[str, ...] = (), min_periods: int = 0
) -> WindowSpec:
    """
    Create a expanding window that includes all preceding rows

    Args:
        grouping_keys:
            Columns ids of grouping keys
        min_periods (int, default 0):
            Minimum number of input rows to generate output.
    Returns:
        WindowSpec
    """
    bounds = RowsWindowBounds(end=0)
    return WindowSpec(
        grouping_keys=tuple(map(ex.deref, grouping_keys)),
        bounds=bounds,
        min_periods=min_periods,
    )


def inverse_cumulative_rows(
    grouping_keys: Tuple[str, ...] = (), min_periods: int = 0
) -> WindowSpec:
    """
    Create a shrinking window that includes all following rows

    Args:
        grouping_keys:
            Columns ids of grouping keys
        min_periods (int, default 0):
            Minimum number of input rows to generate output.
    Returns:
        WindowSpec
    """
    bounds = RowsWindowBounds(start=0)
    return WindowSpec(
        grouping_keys=tuple(map(ex.deref, grouping_keys)),
        bounds=bounds,
        min_periods=min_periods,
    )


### Struct Classes


@dataclass(frozen=True)
class RowsWindowBounds:
    start: Optional[int] = None
    end: Optional[int] = None

    @classmethod
    def from_window_size(
        cls, window: int, closed: Literal["right", "left", "both", "neither"]
    ) -> RowsWindowBounds:
        if closed == "right":
            return cls(-(window - 1), 0)
        elif closed == "left":
            return cls(-window, -1)
        elif closed == "both":
            return cls(-window, 0)
        elif closed == "neither":
            return cls(-(window - 1), -1)
        else:
            raise ValueError(f"Unsupported value for 'closed' parameter: {closed}")

    def __post_init__(self):
        if self.start is None:
            return
        if self.end is None:
            return
        if self.start > self.end:
            raise ValueError(
                f"Invalid window: start({self.start}) is greater than end({self.end})"
            )


@dataclass(frozen=True)
class RangeWindowBounds:
    # TODO(b/388916840) Support range rolling on timeseries with timedeltas.
    start: Optional[int] = None
    end: Optional[int] = None

    def __post_init__(self):
        if self.start is None:
            return
        if self.end is None:
            return
        if self.start > self.end:
            raise ValueError(
                f"Invalid window: start({self.start}) is greater than end({self.end})"
            )


@dataclass(frozen=True)
class WindowSpec:
    """
    Specifies a window over which aggregate and analytic function may be applied.
    grouping_keys: set of column ids to group on
    preceding: Number of preceding rows in the window
    following: Number of preceding rows in the window
    ordering: List of columns ids and ordering direction to override base ordering
    """

    grouping_keys: Tuple[ex.DerefOp, ...] = tuple()
    ordering: Tuple[orderings.OrderingExpression, ...] = tuple()
    bounds: Union[RowsWindowBounds, RangeWindowBounds, None] = None
    min_periods: int = 0

    @property
    def row_bounded(self):
        """
        Whether the window is bounded by row offsets.

        This is relevant for determining whether the window requires a total order
        to calculate deterministically.
        """
        return isinstance(self.bounds, RowsWindowBounds)

    @property
    def all_referenced_columns(self) -> Set[ids.ColumnId]:
        """
        Return list of all variables reference ind the window.
        """
        ordering_vars = itertools.chain.from_iterable(
            item.scalar_expression.column_references for item in self.ordering
        )
        return set(itertools.chain((i.id for i in self.grouping_keys), ordering_vars))

    def without_order(self) -> WindowSpec:
        """Removes ordering clause if ordering isn't required to define bounds."""
        if self.row_bounded:
            raise ValueError("Cannot remove order from row-bounded window")
        return replace(self, ordering=())

    def remap_column_refs(
        self,
        mapping: Mapping[ids.ColumnId, ids.ColumnId],
        allow_partial_bindings: bool = False,
    ) -> WindowSpec:
        return WindowSpec(
            grouping_keys=tuple(
                key.remap_column_refs(mapping, allow_partial_bindings)
                for key in self.grouping_keys
            ),
            ordering=tuple(
                order_part.remap_column_refs(mapping, allow_partial_bindings)
                for order_part in self.ordering
            ),
            bounds=self.bounds,
            min_periods=self.min_periods,
        )
