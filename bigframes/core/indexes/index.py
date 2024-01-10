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

"""An index based on a single column."""

from __future__ import annotations

import typing
from typing import Sequence, Union

import numpy as np
import pandas

import bigframes.core.block_transforms as block_ops
import bigframes.core.blocks as blocks
import bigframes.core.ordering as order
import bigframes.core.utils as utils
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops
import third_party.bigframes_vendored.pandas.core.indexes.base as vendored_pandas_index


class Index(vendored_pandas_index.Index):
    __doc__ = vendored_pandas_index.Index.__doc__

    def __init__(self, data: blocks.BlockHolder):
        self._data = data

    @property
    def name(self) -> blocks.Label:
        return self.names[0]

    @name.setter
    def name(self, value: blocks.Label):
        self.names = [value]

    @property
    def names(self) -> typing.Sequence[blocks.Label]:
        """Returns the names of the Index."""
        return self._data._get_block()._index_labels

    @names.setter
    def names(self, values: typing.Sequence[blocks.Label]):
        return self._data._set_block(self._block.with_index_labels(values))

    @property
    def nlevels(self) -> int:
        return len(self._data._get_block().index_columns)

    @property
    def values(self) -> np.ndarray:
        return self.to_numpy()

    @property
    def ndim(self) -> int:
        return 1

    @property
    def shape(self) -> typing.Tuple[int]:
        return (self._data._get_block().shape[0],)

    @property
    def dtype(self):
        return self._block.index.dtypes[0] if self.nlevels == 1 else np.dtype("O")

    @property
    def dtypes(self) -> pandas.Series:
        return pandas.Series(
            data=self._block.index.dtypes, index=self._block.index.names  # type:ignore
        )

    @property
    def size(self) -> int:
        """Returns the size of the Index."""
        return self.shape[0]

    @property
    def empty(self) -> bool:
        """Returns True if the Index is empty, otherwise returns False."""
        return self.shape[0] == 0

    @property
    def is_monotonic_increasing(self) -> bool:
        """
        Return a boolean if the values are equal or increasing.

        Returns:
            bool
        """
        return typing.cast(
            bool,
            self._data._get_block().is_monotonic_increasing(
                self._data._get_block().index_columns
            ),
        )

    @property
    def is_monotonic_decreasing(self) -> bool:
        """
        Return a boolean if the values are equal or decreasing.

        Returns:
            bool
        """
        return typing.cast(
            bool,
            self._data._get_block().is_monotonic_decreasing(
                self._data._get_block().index_columns
            ),
        )

    @property
    def is_unique(self) -> bool:
        # TODO: Cache this at block level
        # Avoid circular imports
        return not self.has_duplicates

    @property
    def has_duplicates(self) -> bool:
        # TODO: Cache this at block level
        # Avoid circular imports
        import bigframes.core.block_transforms as block_ops
        import bigframes.dataframe as df

        duplicates_block, indicator = block_ops.indicate_duplicates(
            self._block, self._block.index_columns
        )
        duplicates_block = duplicates_block.select_columns(
            [indicator]
        ).with_column_labels(["is_duplicate"])
        duplicates_df = df.DataFrame(duplicates_block)
        return duplicates_df["is_duplicate"].any()

    @property
    def _block(self) -> blocks.Block:
        return self._data._get_block()

    @property
    def T(self) -> Index:
        return self.transpose()

    def _memory_usage(self) -> int:
        (n_rows,) = self.shape
        return sum(
            self.dtypes.map(
                lambda dtype: bigframes.dtypes.DTYPE_BYTE_SIZES.get(dtype, 8) * n_rows
            )
        )

    def transpose(self) -> Index:
        return self

    def sort_values(self, *, ascending: bool = True, na_position: str = "last"):
        if na_position not in ["first", "last"]:
            raise ValueError("Param na_position must be one of 'first' or 'last'")
        direction = (
            order.OrderingDirection.ASC if ascending else order.OrderingDirection.DESC
        )
        na_last = na_position == "last"
        index_columns = self._block.index_columns
        ordering = [
            order.OrderingColumnReference(column, direction=direction, na_last=na_last)
            for column in index_columns
        ]
        return Index._from_block(self._block.order_by(ordering))

    def astype(
        self,
        dtype: Union[bigframes.dtypes.DtypeString, bigframes.dtypes.Dtype],
    ) -> Index:
        if self.nlevels > 1:
            raise TypeError("Multiindex does not support 'astype'")
        return self._apply_unary_op(ops.AsTypeOp(to_type=dtype))

    def all(self) -> bool:
        if self.nlevels > 1:
            raise TypeError("Multiindex does not support 'all'")
        return typing.cast(bool, self._apply_aggregation(agg_ops.all_op))

    def any(self) -> bool:
        if self.nlevels > 1:
            raise TypeError("Multiindex does not support 'any'")
        return typing.cast(bool, self._apply_aggregation(agg_ops.any_op))

    def nunique(self) -> int:
        return typing.cast(int, self._apply_aggregation(agg_ops.nunique_op))

    def max(self) -> typing.Any:
        return self._apply_aggregation(agg_ops.max_op)

    def min(self) -> typing.Any:
        return self._apply_aggregation(agg_ops.min_op)

    def argmax(self) -> int:
        block, row_nums = self._block.promote_offsets()
        block = block.order_by(
            [
                *[
                    order.OrderingColumnReference(
                        col, direction=order.OrderingDirection.DESC
                    )
                    for col in self._block.index_columns
                ],
                order.OrderingColumnReference(row_nums),
            ]
        )
        import bigframes.series as series

        return typing.cast(int, series.Series(block.select_column(row_nums)).iloc[0])

    def argmin(self) -> int:
        block, row_nums = self._block.promote_offsets()
        block = block.order_by(
            [
                *[
                    order.OrderingColumnReference(col)
                    for col in self._block.index_columns
                ],
                order.OrderingColumnReference(row_nums),
            ]
        )
        import bigframes.series as series

        return typing.cast(int, series.Series(block.select_column(row_nums)).iloc[0])

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        *,
        dropna: bool = True,
    ):
        block = block_ops.value_counts(
            self._block,
            self._block.index_columns,
            normalize=normalize,
            ascending=ascending,
            dropna=dropna,
        )
        import bigframes.series as series

        return series.Series(block)

    def fillna(self, value=None) -> Index:
        if self.nlevels > 1:
            raise TypeError("Multiindex does not support 'fillna'")
        return self._apply_unary_op(ops.partial_right(ops.fillna_op, value))

    def rename(self, name: Union[str, Sequence[str]]) -> Index:
        names = [name] if isinstance(name, str) else list(name)
        if len(names) != self.nlevels:
            raise ValueError("'name' must be same length as levels")
        return Index._from_block(self._block.with_index_labels(names))

    def drop(
        self,
        labels: typing.Any,
    ) -> Index:
        # ignore axis, columns params
        block = self._block
        level_id = self._block.index_columns[0]
        if utils.is_list_like(labels):
            block, inverse_condition_id = block.apply_unary_op(
                level_id, ops.IsInOp(values=tuple(labels), match_nulls=True)
            )
            block, condition_id = block.apply_unary_op(
                inverse_condition_id, ops.invert_op
            )
        else:
            block, condition_id = block.apply_unary_op(
                level_id, ops.partial_right(ops.ne_op, labels)
            )
        block = block.filter(condition_id, keep_null=True)
        block = block.drop_columns([condition_id])
        return Index._from_block(block)

    def dropna(self, how: str = "any") -> Index:
        if how not in ("any", "all"):
            raise ValueError("'how' must be one of 'any', 'all'")
        result = block_ops.dropna(self._block, self._block.index_columns, how=how)  # type: ignore
        return Index._from_block(result)

    def drop_duplicates(self, *, keep: str = "first") -> Index:
        block = block_ops.drop_duplicates(self._block, self._block.index_columns, keep)
        return Index._from_block(block)

    def isin(self, values) -> Index:
        if not utils.is_list_like(values):
            raise TypeError(
                "only list-like objects are allowed to be passed to "
                f"isin(), you passed a [{type(values).__name__}]"
            )

        return self._apply_unary_op(
            ops.IsInOp(values=tuple(values), match_nulls=True)
        ).fillna(value=False)

    def _apply_unary_op(
        self,
        op: ops.UnaryOp,
    ) -> Index:
        """Applies a unary operator to the index."""
        block = self._block
        result_ids = []
        for col in self._block.index_columns:
            block, result_id = block.apply_unary_op(col, op)
            result_ids.append(result_id)

        block = block.set_index(result_ids, index_labels=self._block.index.names)
        return Index._from_block(block)

    def _apply_aggregation(self, op: agg_ops.AggregateOp) -> typing.Any:
        if self.nlevels > 1:
            raise NotImplementedError(f"Multiindex does not yet support {op.name}")
        column_id = self._block.index_columns[0]
        return self._block.get_stat(column_id, op)

    def __getitem__(self, key: int) -> typing.Any:
        if isinstance(key, int):
            if key != -1:
                result_pd_df, _ = self._block.slice(key, key + 1, 1).to_pandas()
            else:  # special case, want [-1:] instead of [-1:0]
                result_pd_df, _ = self._block.slice(key).to_pandas()
            if result_pd_df.empty:
                raise IndexError("single positional indexer is out-of-bounds")
            return result_pd_df.index[0]
        else:
            raise NotImplementedError(f"Index key not supported {key}")

    def to_pandas(self) -> pandas.Index:
        """Gets the Index as a pandas Index.

        Returns:
            pandas.Index:
                A pandas Index with all of the labels from this Index.
        """
        return self._block.index.to_pandas()

    def to_numpy(self, dtype=None, **kwargs) -> np.ndarray:
        return self.to_pandas().to_numpy(dtype, **kwargs)

    __array__ = to_numpy

    def __len__(self):
        return self.shape[0]

    @classmethod
    def _from_block(cls, block: blocks.Block) -> Index:
        import bigframes.dataframe as df

        return Index(df.DataFrame(block))
