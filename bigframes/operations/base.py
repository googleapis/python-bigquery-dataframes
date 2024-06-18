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

import functools
import typing
from typing import List, Sequence

import bigframes_vendored.pandas.pandas._typing as vendored_pandas_typing
import numpy
import pandas as pd

import bigframes.constants as constants
import bigframes.core.blocks as blocks
import bigframes.core.convert
import bigframes.core.expression as ex
import bigframes.core.indexes as indexes
import bigframes.core.scalar as scalars
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops
import bigframes.series as series
import bigframes.session


def requires_index(meth):
    @functools.wraps(meth)
    def guarded_meth(df: SeriesMethods, *args, **kwargs):
        df._throw_if_null_index(meth.__name__)
        return meth(df, *args, **kwargs)

    return guarded_meth


class SeriesMethods:
    def __init__(
        self,
        data=None,
        index: vendored_pandas_typing.Axes | None = None,
        dtype: typing.Optional[
            bigframes.dtypes.DtypeString | bigframes.dtypes.Dtype
        ] = None,
        name: str | None = None,
        copy: typing.Optional[bool] = None,
        *,
        session: typing.Optional[bigframes.session.Session] = None,
    ):
        import bigframes.pandas

        # just ignore object dtype if provided
        if dtype in {numpy.dtypes.ObjectDType, "object"}:
            dtype = None

        read_pandas_func = (
            session.read_pandas
            if (session is not None)
            else (lambda x: bigframes.pandas.read_pandas(x))
        )

        block: typing.Optional[blocks.Block] = None
        if (name is not None) and not isinstance(name, typing.Hashable):
            raise ValueError(
                f"BigQuery DataFrames only supports hashable series names. {constants.FEEDBACK_LINK}"
            )
        if copy is not None and not copy:
            raise ValueError(
                f"Series constructor only supports copy=True. {constants.FEEDBACK_LINK}"
            )
        if isinstance(data, blocks.Block):
            # Constructing from block is for internal use only - shouldn't use parameters, block encompasses all state
            assert len(data.value_columns) == 1
            assert len(data.column_labels) == 1
            assert index is None
            assert name is None
            assert dtype is None
            block = data

        # interpret these cases as both index and data
        elif isinstance(data, bigframes.pandas.Series) or pd.api.types.is_dict_like(
            data
        ):  # includes pd.Series
            if isinstance(data, bigframes.pandas.Series):
                data = data.copy()
                if name is not None:
                    data.name = name
                if dtype is not None:
                    data = data.astype(dtype)
            else:  # local dict-like data
                data = read_pandas_func(pd.Series(data, name=name, dtype=dtype))  # type: ignore
            data_block = data._block
            if index is not None:
                # reindex
                bf_index = indexes.Index(index, session=session)
                idx_block = bf_index._block
                idx_cols = idx_block.value_columns
                block_idx, _ = idx_block.join(data_block, how="left")
                data_block = block_idx.with_index_labels(bf_index.names)
            block = data_block

        # list-like data that will get default index
        elif isinstance(data, indexes.Index) or pd.api.types.is_list_like(data):
            data = indexes.Index(data, dtype=dtype, name=name, session=session)
            # set to none as it has already been applied, avoid re-cast later
            if data.nlevels != 1:
                raise NotImplementedError("Cannot interpret multi-index as Series.")
            # Reset index to promote index columns to value columns, set default index
            data_block = data._block.reset_index(drop=False).with_column_labels(
                data.names
            )
            if index is not None:
                # Align by offset
                bf_index = indexes.Index(index, session=session)
                idx_block = bf_index._block.reset_index(
                    drop=False
                )  # reset to align by offsets, and then reset back
                idx_cols = idx_block.value_columns
                data_block, (l_mapping, _) = idx_block.join(data_block, how="left")
                data_block = data_block.set_index([l_mapping[col] for col in idx_cols])
                data_block = data_block.with_index_labels(bf_index.names)
            block = data_block

        else:  # Scalar case
            if index is not None:
                bf_index = indexes.Index(index, session=session)
            else:
                bf_index = indexes.Index(
                    [] if (data is None) else [0],
                    session=session,
                    dtype=bigframes.dtypes.INT_DTYPE,
                )
            block, _ = bf_index._block.create_constant(data, dtype)
            block = block.with_column_labels([name])

        assert block is not None
        self._block: blocks.Block = block

    @property
    def _value_column(self) -> str:
        return self._block.value_columns[0]

    @property
    def _name(self) -> blocks.Label:
        return self._block.column_labels[0]

    @property
    def _dtype(self):
        return self._block.dtypes[0]

    def _set_block(self, block: blocks.Block):
        self._block = block

    def _get_block(self) -> blocks.Block:
        return self._block

    def _apply_unary_op(
        self,
        op: ops.UnaryOp,
    ) -> series.Series:
        """Applies a unary operator to the series."""
        block, result_id = self._block.apply_unary_op(
            self._value_column, op, result_label=self._name
        )
        return series.Series(block.select_column(result_id))

    def _apply_binary_op(
        self,
        other: typing.Any,
        op: ops.BinaryOp,
        alignment: typing.Literal["outer", "left"] = "outer",
        reverse: bool = False,
    ) -> series.Series:
        """Applies a binary operator to the series and other."""
        if bigframes.core.convert.is_series_convertible(other):
            self_index = indexes.Index(self._block)
            other_series = bigframes.core.convert.to_bf_series(
                other, self_index, self._block.session
            )
            (self_col, other_col, block) = self._align(other_series, how=alignment)

            name = self._name
            if (
                hasattr(other, "name")
                and other.name != self._name
                and alignment == "outer"
            ):
                name = None
            expr = op.as_expr(
                other_col if reverse else self_col, self_col if reverse else other_col
            )
            block, result_id = block.project_expr(expr, name)
            return series.Series(block.select_column(result_id))

        else:  # Scalar binop
            name = self._name
            expr = op.as_expr(
                ex.const(other) if reverse else self._value_column,
                self._value_column if reverse else ex.const(other),
            )
            block, result_id = self._block.project_expr(expr, name)
            return series.Series(block.select_column(result_id))

    def _apply_nary_op(
        self,
        op: ops.NaryOp,
        others: Sequence[typing.Union[series.Series, scalars.Scalar]],
        ignore_self=False,
    ):
        """Applies an n-ary operator to the series and others."""
        values, block = self._align_n(others, ignore_self=ignore_self)
        block, result_id = block.apply_nary_op(
            values,
            op,
            self._name,
        )
        return series.Series(block.select_column(result_id))

    def _apply_binary_aggregation(
        self, other: series.Series, stat: agg_ops.BinaryAggregateOp
    ) -> float:
        (left, right, block) = self._align(other, how="outer")

        return block.get_binary_stat(left, right, stat)

    def _align(self, other: series.Series, how="outer") -> tuple[str, str, blocks.Block]:  # type: ignore
        """Aligns the series value with another scalar or series object. Returns new left column id, right column id and joined tabled expression."""
        values, block = self._align_n(
            [
                other,
            ],
            how,
        )
        return (values[0], values[1], block)

    def _align_n(
        self,
        others: typing.Sequence[typing.Union[series.Series, scalars.Scalar]],
        how="outer",
        ignore_self=False,
    ) -> tuple[typing.Sequence[str], blocks.Block]:
        if ignore_self:
            value_ids: List[str] = []
        else:
            value_ids = [self._value_column]

        block = self._block
        for other in others:
            if isinstance(other, series.Series):
                block, (
                    get_column_left,
                    get_column_right,
                ) = block.join(other._block, how=how)
                value_ids = [
                    *[get_column_left[value] for value in value_ids],
                    get_column_right[other._value_column],
                ]
            else:
                # Will throw if can't interpret as scalar.
                dtype = typing.cast(bigframes.dtypes.Dtype, self._dtype)
                block, constant_col_id = block.create_constant(other, dtype=dtype)
                value_ids = [*value_ids, constant_col_id]
        return (value_ids, block)

    def _throw_if_null_index(self, opname: str):
        if len(self._block.index_columns) == 0:
            raise bigframes.exceptions.NullIndexError(
                f"Series cannot perform {opname} as it has no index. Set an index using set_index."
            )
