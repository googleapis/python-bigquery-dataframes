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

import ibis

import bigframes.core.blocks as blocks
import bigframes.core.indexes.index
import bigframes.series


class LocSeriesIndexer:
    def __init__(self, series: bigframes.series.Series):
        self._series = series

    def __setitem__(self, key, value) -> None:
        index = self._series.index
        # TODO(swast): support MultiIndex
        if not isinstance(index, bigframes.core.indexes.index.Index):
            raise ValueError("loc requires labeled index")

        if isinstance(key, slice):
            # TODO(swast): Implement loc with slices.
            raise NotImplementedError("loc does not yet support slices")
        elif isinstance(key, list):
            # TODO(tbergeron): Implement loc for index label list.
            raise NotImplementedError("loc does not yet support index label lists")

        # Assume the key is for the index label.
        block = self._series._block
        value_column = self._series._value
        index_column = block.expr.get_column(index._index_column)
        new_value = (
            ibis.case()
            .when(
                index_column == ibis.literal(key, index_column.type()),
                ibis.literal(value, value_column.type()),
            )
            .else_(value_column)
            .end()
            .name(value_column.get_name())
        )
        all_columns = []
        for column in block.expr.columns:
            if column.get_name() != value_column.get_name():
                all_columns.append(column)
            else:
                all_columns.append(new_value)
        block.expr = block.expr.projection(all_columns)


class IlocSeriesIndexer:
    def __init__(self, series: bigframes.series.Series):
        self._series = series

    def __getitem__(self, key) -> bigframes.scalar.Scalar | bigframes.series.Series:
        if isinstance(key, slice):
            return _slice_series(self._series, key.start, key.stop, key.step)
        if isinstance(key, list):
            # TODO(tbergeron): Implement list, may require fixing ibis table literal support
            raise NotImplementedError("iloc does not yet support single offsets")
        elif isinstance(key, int):
            # TODO(tbergeron): Implement iloc for single offset returning deferred scalar
            raise NotImplementedError("iloc does not yet support single offsets")
        else:
            raise TypeError("Invalid argument type.")


class _iLocIndexer:
    def __init__(self, dataframe: bigframes.DataFrame):
        self._dataframe = dataframe

    def __getitem__(self, key) -> bigframes.scalar.Scalar | bigframes.DataFrame:
        """
        Only slice type is supported currently for indexing the iloc object.
        """
        if isinstance(key, slice):
            return _slice_dataframe(self._dataframe, key.start, key.stop, key.step)
        if isinstance(key, list):
            raise NotImplementedError("iloc does not yet support indexing with a list")
        elif isinstance(key, int):
            raise NotImplementedError("iloc does not yet support single offsets")
        elif isinstance(key, tuple):
            raise NotImplementedError(
                "iloc does not yet support indexing with a (row, column) tuple"
            )
        elif callable(key):
            raise NotImplementedError(
                "iloc does not yet support indexing with a callable"
            )
        else:
            raise TypeError("Invalid argument type.")


def _slice_block(
    block: bigframes.core.blocks.Block,
    start: typing.Optional[int] = None,
    stop: typing.Optional[int] = None,
    step: typing.Optional[int] = None,
) -> bigframes.core.blocks.Block:
    expr_with_offsets = block.expr.project_offsets()
    cond_list = []
    # TODO(tbergeron): Handle negative indexing
    if start:
        cond_list.append(expr_with_offsets.offsets >= start)
    if stop:
        cond_list.append(expr_with_offsets.offsets < stop)
    if step:
        # TODO(tbergeron): Reverse the ordering if negative step
        start = start if start else 0
        cond_list.append((expr_with_offsets.offsets - start) % step == 0)
    if not cond_list:
        return block
    original_name = block.index.name
    block = blocks.Block(
        expr_with_offsets.filter(functools.reduce(lambda x, y: x & y, cond_list)),
        index_columns=block.index_columns,
    )
    # TODO(swast): Support MultiIndex.
    block.index.name = original_name
    return block


def _slice_series(
    series: bigframes.Series,
    start: typing.Optional[int] = None,
    stop: typing.Optional[int] = None,
    step: typing.Optional[int] = None,
) -> bigframes.Series:
    return bigframes.Series(
        _slice_block(series._block, start=start, stop=stop, step=step),
        series._value_column,
        name=series.name,
    )


def _slice_dataframe(
    dataframe: bigframes.DataFrame,
    start: typing.Optional[int] = None,
    stop: typing.Optional[int] = None,
    step: typing.Optional[int] = None,
) -> bigframes.DataFrame:
    block = _slice_block(dataframe._block, start=start, stop=stop, step=step)
    result = dataframe._copy()
    result._block = block
    return result
