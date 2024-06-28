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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

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
        grouping_keys=grouping_keys, min_periods=min_periods, ordering=ordering
    )


### Rows-based Windows
def rows(
    grouping_keys: Tuple[str, ...] = (),
    preceding: Optional[int] = None,
    following: Optional[int] = None,
    min_periods: int = 0,
    ordering: Tuple[orderings.OrderingExpression, ...] = (),
) -> WindowSpec:
    """
    Create a row-bounded window.

    Args:
        grouping_keys:
            Columns ids of grouping keys
        preceding:
            number of preceding rows to include. If None, include all preceding rows
        following:
            number of following rows to include. If None, include all following rows
        min_periods (int, default 0):
            Minimum number of input rows to generate output.
        ordering:
            Ordering to apply on top of based dataframe ordering
    Returns:
        WindowSpec
    """
    assert (preceding is not None) or (following is not None)
    bounds = RowsWindowBounds(preceding=preceding, following=following)
    return WindowSpec(
        grouping_keys=grouping_keys,
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
    bounds = RowsWindowBounds(following=0)
    return WindowSpec(
        grouping_keys=grouping_keys, bounds=bounds, min_periods=min_periods
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
    bounds = RowsWindowBounds(preceding=0)
    return WindowSpec(
        grouping_keys=grouping_keys, bounds=bounds, min_periods=min_periods
    )


### Struct Classes


@dataclass(frozen=True)
class RowsWindowBounds:
    preceding: Optional[int] = None
    following: Optional[int] = None


# TODO: Expand to datetime offsets
OffsetType = Union[float, int]


@dataclass(frozen=True)
class RangeWindowBounds:
    preceding: Optional[OffsetType] = None
    following: Optional[OffsetType] = None


@dataclass(frozen=True)
class WindowSpec:
    """
    Specifies a window over which aggregate and analytic function may be applied.
    grouping_keys: set of column ids to group on
    preceding: Number of preceding rows in the window
    following: Number of preceding rows in the window
    ordering: List of columns ids and ordering direction to override base ordering
    """

    grouping_keys: Tuple[str, ...] = tuple()
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
