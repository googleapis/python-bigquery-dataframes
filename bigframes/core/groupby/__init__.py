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
import bigframes.core.blocks as blocks
import bigframes.core.window
import bigframes.dataframe as df
import bigframes.dtypes


class DataFrameGroupBy:
    """Represents a deferred dataframe with a grouping expression."""

    def __init__(
        self,
        block: blocks.Block,
        by_col_ids: typing.Sequence[str],
        *,
        dropna: bool = True,
        as_index: bool = True,
    ):
        if len(by_col_ids) > 1 and as_index:
            raise ValueError(
                "Set as_index=False if grouping by multiple values. Mutli-index not yet supported"
            )
        # TODO(tbergeron): Support more group-by expression types
        self._block = block
        self._col_id_labels = {
            value_column: column_label
            for value_column, column_label in zip(
                block.value_columns, block.column_labels
            )
        }
        self._by_col_ids = by_col_ids
        self._dropna = dropna  # Applies to aggregations but not windowing
        self._as_index = as_index

    def sum(
        self,
        numeric_only: bool = False,
    ) -> df.DataFrame:
        """Sums the numeric values for each group in the dataframe. Only supports 'numeric_only'=True."""
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        return self._aggregate(agg_ops.sum_op, numeric_only=True)

    def mean(
        self,
        numeric_only: bool = False,
    ) -> df.DataFrame:
        """Calculates the mean of non-null values in each group. Only supports 'numeric_only'=True."""
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        return self._aggregate(agg_ops.mean_op, numeric_only=True)

    def min(
        self,
        numeric_only: bool = False,
    ) -> df.DataFrame:
        """Calculates the minimum value in each group. Only supports 'numeric_only'=True."""
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        return self._aggregate(agg_ops.min_op, numeric_only=True)

    def max(
        self,
        numeric_only: bool = False,
    ) -> df.DataFrame:
        """Calculates the maximum value in each group. Only supports 'numeric_only'=True."""
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        return self._aggregate(agg_ops.max_op, numeric_only=True)

    def std(
        self,
        numeric_only: bool = False,
    ) -> df.DataFrame:
        """Calculates the standard deviation of values in each group. Only supports 'numeric_only'=True."""
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        return self._aggregate(agg_ops.std_op, numeric_only=True)

    def var(
        self,
        numeric_only: bool = False,
    ) -> df.DataFrame:
        """Calculates the variance of values in each group. Only supports 'numeric_only'=True."""
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        return self._aggregate(agg_ops.var_op, numeric_only=True)

    def all(self) -> df.DataFrame:
        """Returns true if any non-null value evalutes to true for each group."""
        return self._aggregate(agg_ops.all_op)

    def any(self) -> df.DataFrame:
        """Returns true if all non-null values evaluate to true for each group."""
        return self._aggregate(agg_ops.any_op)

    def count(self) -> df.DataFrame:
        """Counts the non-null values in each group."""
        return self._aggregate(agg_ops.count_op)

    def cumsum(
        self,
        *,
        numeric_only: bool = False,
    ) -> df.DataFrame:
        """Calculate the cumulative sum of values in each grouping. Only supports 'numeric_only'=True."""
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        window = bigframes.core.WindowSpec(grouping_keys=self._by_col_ids, following=0)
        return self._apply_window_op(agg_ops.sum_op, window, numeric_only=True)

    def cummin(
        self,
        *,
        numeric_only: bool = False,
    ) -> df.DataFrame:
        """Calculate the cumulative minimum of values in each grouping. Only supports 'numeric_only'=True."""
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        window = bigframes.core.WindowSpec(grouping_keys=self._by_col_ids, following=0)
        return self._apply_window_op(agg_ops.min_op, window, numeric_only=True)

    def cummax(
        self,
        *,
        numeric_only: bool = False,
    ) -> df.DataFrame:
        """Calculate the cumulative maximum of values in each grouping. Only supports 'numeric_only'=True."""
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        window = bigframes.core.WindowSpec(grouping_keys=self._by_col_ids, following=0)
        return self._apply_window_op(agg_ops.max_op, window, numeric_only=True)

    def cumprod(self) -> df.DataFrame:
        """Calculate the cumulative product of values in each grouping. Drops non-numeric columns always."""
        window = bigframes.core.WindowSpec(grouping_keys=self._by_col_ids, following=0)
        return self._apply_window_op(agg_ops.product_op, window, numeric_only=True)

    def _aggregated_columns(self, numeric_only: bool = False):
        return [
            col_id
            for col_id, dtype in zip(self._block.value_columns, self._block.dtypes)
            if col_id not in self._by_col_ids
            and (
                (not numeric_only)
                or (dtype in bigframes.dtypes.NUMERIC_BIGFRAMES_TYPES)
            )
        ]

    def _aggregate(
        self, aggregate_op: agg_ops.AggregateOp, numeric_only: bool = False
    ) -> df.DataFrame:
        aggregated_col_ids = self._aggregated_columns(numeric_only=numeric_only)
        aggregations = [
            (col_id, aggregate_op, col_id + "_bf_aggregated")
            for col_id in aggregated_col_ids
        ]
        result_block = self._block.aggregate(
            self._by_col_ids,
            aggregations,
            as_index=self._as_index,
            dropna=self._dropna,
        )
        return df.DataFrame(result_block)

    def _apply_window_op(
        self,
        op: agg_ops.WindowOp,
        window_spec: bigframes.core.WindowSpec,
        numeric_only: bool = False,
    ):
        columns = self._aggregated_columns(numeric_only=numeric_only)
        pruned_block = self._block.copy().drop_columns(
            [
                col
                for col in self._block.value_columns
                if col not in [*columns, *window_spec.grouping_keys]
            ]
        )
        for col_id in columns[:-1]:
            pruned_block.apply_window_op(
                col_id,
                op,
                window_spec=window_spec,
                skip_null_groups=self._dropna,
                skip_reproject_unsafe=True,
            )
        # Reproject after applying final independent window operation.
        pruned_block.apply_window_op(columns[-1], op, window_spec=window_spec)
        final_block = pruned_block.drop_columns(window_spec.grouping_keys)
        return df.DataFrame(final_block)
