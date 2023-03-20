from __future__ import annotations

import ibis

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
