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

import typing

import bigframes.aggregations as agg_ops
import bigframes.core as core
import bigframes.core.blocks as blocks

if typing.TYPE_CHECKING:
    from bigframes.series import Series


class Window:
    """Represents a window applied over a dataframe."""

    # TODO(tbergeron): Windows with groupings should create multi-indexed results

    def __init__(
        self,
        block: blocks.Block,
        window_spec: core.WindowSpec,
        value_column_id: str,
        label: typing.Optional[str] = None,
    ):
        self._block = block
        self._window_spec = window_spec
        self._value_column_id = value_column_id
        self._label = label

    def min(self) -> Series:
        """Calculate the windowed min of values in the dataset."""
        return self._apply_aggregate(agg_ops.min_op)

    def max(self) -> Series:
        """Calculate the windowed max of values in the dataset."""
        return self._apply_aggregate(agg_ops.max_op)

    def sum(self) -> Series:
        """Calculate the windowed sum of values in the dataset."""
        return self._apply_aggregate(agg_ops.sum_op)

    def count(self) -> Series:
        """Calculate the windowed count of values in the dataset."""
        return self._apply_aggregate(agg_ops.count_op)

    def mean(self) -> Series:
        """Calculate the windowed mean of values in the dataset."""
        return self._apply_aggregate(agg_ops.mean_op)

    def std(self) -> Series:
        """Calculate the windowed standard deviation of values in the dataset."""
        return self._apply_aggregate(agg_ops.std_op)

    def var(self) -> Series:
        """Calculate the windowed variance of values in the dataset."""
        return self._apply_aggregate(agg_ops.var_op)

    def _apply_aggregate(
        self,
        op: agg_ops.AggregateOp,
    ) -> Series:
        block = self._block.copy()
        block.apply_window_op(self._value_column_id, op, self._window_spec)
        from bigframes.series import Series

        return Series(block, self._value_column_id, name=self._label)
