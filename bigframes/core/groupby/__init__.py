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
import bigframes.dataframe as df
import bigframes.dtypes


class DataFrameGroupBy:
    """Represents a deferred dataframe with a grouping expression."""

    def __init__(
        self,
        block: blocks.Block,
        col_id_labels: typing.Mapping[str, str],
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
        self._col_id_labels = col_id_labels
        self._by_col_ids = by_col_ids
        self._dropna = dropna  # Applies to aggregations but not windowing
        self._as_index = as_index

    def sum(self) -> df.DataFrame:
        """Sums the numeric values for each group in the dataframe. Drops non-numeric columns always (like numeric_only=True in Pandas)."""
        aggregated_col_ids = [
            col_id
            for col_id, dtype in zip(self._block.value_columns, self._block.dtypes)
            if col_id not in self._by_col_ids
            and dtype in bigframes.dtypes.NUMERIC_BIGFRAMES_TYPES
        ]
        return self._aggregate(agg_ops.sum_op, aggregated_col_ids)

    def _aggregate(
        self,
        aggregate_op: agg_ops.AggregateOp,
        aggregated_col_ids: typing.Sequence[str],
    ) -> df.DataFrame:
        aggregations = [
            (col_id, aggregate_op, col_id + "_bf_aggregated")
            for col_id in aggregated_col_ids
        ]

        result_block = self._block.aggregate(
            self._by_col_ids,
            aggregations,
            dropna=self._dropna,
        )
        if self._as_index:
            # Promote 'by' column to index.
            # TODO(tbergeron): generalize for multi-index (once multi-index introduced)
            by_col_id = self._by_col_ids[0]
            result_block.index_columns = self._by_col_ids
            index_label = self._col_id_labels[by_col_id]
            result_block.index.name = index_label
            labels = [self._col_id_labels[col_id] for col_id in aggregated_col_ids]
            return df.DataFrame(result_block.index, labels)
        else:
            result_block = result_block.reset_index()
            by_col_labels = [self._col_id_labels[col_id] for col_id in self._by_col_ids]
            aggregate_labels = [
                self._col_id_labels[col_id] for col_id in aggregated_col_ids
            ]
            return df.DataFrame(result_block.index, by_col_labels + aggregate_labels)
