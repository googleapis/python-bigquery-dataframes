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
            return self._slice(key.start, key.stop, key.step)
        if isinstance(key, list):
            # TODO(tbergeron): Implement list, may require fixing ibis table literal support
            raise NotImplementedError("iloc does not yet support single offsets")
        elif isinstance(key, int):
            # TODO(tbergeron): Implement iloc for single offset returning deferred scalar
            raise NotImplementedError("iloc does not yet support single offsets")
        else:
            raise TypeError("Invalid argument type.")

    def _slice(
        self,
        start: typing.Optional[int] = None,
        stop: typing.Optional[int] = None,
        step: typing.Optional[int] = None,
    ):
        expr_with_offsets = self._series._block.expr.project_offsets()
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
            return self._series
        block = blocks.Block(
            expr_with_offsets.filter(functools.reduce(lambda x, y: x & y, cond_list)),
            index_columns=self._series._block.index_columns,
        )
        return bigframes.Series(
            block, self._series._value_column, name=self._series.name
        )
