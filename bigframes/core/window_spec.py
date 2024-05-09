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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import bigframes.core.ordering as orderings


# Unbound Windows
def grouping(grouping_keys: Tuple[str, ...] = ()):
    return WindowSpec(grouping_keys=grouping_keys)


def unbound(grouping_keys: Tuple[str, ...] = (), min_periods: int = 0):
    return WindowSpec(grouping_keys=grouping_keys, min_periods=min_periods)


### Range-based Windows
def range_over(
    ordering: Tuple[orderings.OrderingExpression, ...],
    grouping_keys: Tuple[str, ...] = (),
):
    bounds = RangeWindowBounds()
    return WindowSpec(grouping_keys=grouping_keys, bounds=bounds, ordering=ordering)


### Rows-based Windows
def rows(
    grouping_keys: Tuple[str, ...] = (),
    preceding: Optional[int] = None,
    following: Optional[int] = None,
    min_periods: int = 0,
    ordering: Tuple[orderings.OrderingExpression, ...] = (),
):
    bounds = RowsWindowBounds(preceding=preceding, following=following)
    return WindowSpec(
        grouping_keys=grouping_keys,
        bounds=bounds,
        min_periods=min_periods,
        ordering=ordering,
    )


def cumulative_rows(grouping_keys: Tuple[str, ...] = (), min_periods: int = 0):
    bounds = RowsWindowBounds(following=0)
    return WindowSpec(
        grouping_keys=grouping_keys, bounds=bounds, min_periods=min_periods
    )


def inverse_cumulative_rows(grouping_keys: Tuple[str, ...] = (), min_periods: int = 0):
    bounds = RowsWindowBounds(preceding=0)
    return WindowSpec(
        grouping_keys=grouping_keys, bounds=bounds, min_periods=min_periods
    )


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
