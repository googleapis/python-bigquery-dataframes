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

import ibis
import pandas as pd

import bigframes.core.indexes.index
import bigframes.series


class LocSeriesIndexer:
    def __init__(self, series: bigframes.series.Series):
        self._series = series

    def __getitem__(self, key) -> bigframes.Series:
        """
        Only indexing by a boolean bigframes.Series is currently supported
        """
        return _loc_getitem_series_or_dataframe(self._series, key)

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
        """
        Index series using integer offsets. Currently supports index by key type:

        slice: i.e. series.iloc[2:5] returns values at index 2, 3, and 4 as a series
        individual offset: i.e. series.iloc[0] returns value at index 0 as a scalar

        Other key types are not yet supported.
        """
        return _iloc_getitem_series_or_dataframe(self._series, key)


class _LocIndexer:
    def __init__(self, dataframe: bigframes.DataFrame):
        self._dataframe = dataframe

    def __getitem__(self, key) -> bigframes.DataFrame:
        """
        Only indexing by a boolean bigframes.Series is currently supported
        """
        return _loc_getitem_series_or_dataframe(self._dataframe, key)


class _iLocIndexer:
    def __init__(self, dataframe: bigframes.DataFrame):
        self._dataframe = dataframe

    def __getitem__(self, key) -> bigframes.DataFrame | pd.Series:
        """
        Index dataframe using integer offsets. Currently supports index by key type:

        slice: i.e. df.iloc[2:5] returns rows at index 2, 3, and 4 as a dataframe
        individual offset: i.e. df.iloc[0] returns row at index 0 as a pandas Series

        Other key types are not yet supported.
        """
        return _iloc_getitem_series_or_dataframe(self._dataframe, key)


@typing.overload
def _loc_getitem_series_or_dataframe(
    series_or_dataframe: bigframes.DataFrame, key
) -> bigframes.DataFrame:
    ...


@typing.overload
def _loc_getitem_series_or_dataframe(
    series_or_dataframe: bigframes.Series, key
) -> bigframes.Series:
    ...


def _loc_getitem_series_or_dataframe(
    series_or_dataframe: bigframes.DataFrame | bigframes.Series, key
) -> bigframes.DataFrame | bigframes.Series:
    if isinstance(key, bigframes.Series):
        return series_or_dataframe[key]
    elif isinstance(key, list):
        raise NotImplementedError(
            "loc does not yet support indexing with a list of labels"
        )
    elif isinstance(key, slice):
        raise NotImplementedError("loc does not yet support indexing with a slice")
    elif callable(key):
        raise NotImplementedError("loc does not yet support indexing with a callable")
    else:
        raise TypeError(
            "Invalid argument type. loc currently only supports indexing with a boolean bigframes Series."
        )


@typing.overload
def _iloc_getitem_series_or_dataframe(
    series_or_dataframe: bigframes.Series, key
) -> bigframes.Series | bigframes.scalar.Scalar:
    ...


@typing.overload
def _iloc_getitem_series_or_dataframe(
    series_or_dataframe: bigframes.DataFrame, key
) -> bigframes.DataFrame | pd.Series:
    ...


def _iloc_getitem_series_or_dataframe(
    series_or_dataframe: bigframes.DataFrame | bigframes.Series, key
) -> bigframes.DataFrame | bigframes.Series | bigframes.scalar.Scalar | pd.Series:
    if isinstance(key, int):
        if key < 0:
            raise NotImplementedError(
                "iloc does not yet support negative single positional index"
            )
        internal_slice_result = series_or_dataframe._slice(key, key + 1, 1)
        result_pd_df = internal_slice_result.compute()
        if result_pd_df.empty:
            raise IndexError("single positional indexer is out-of-bounds")
        return result_pd_df.iloc[0]
    elif isinstance(key, slice):
        return series_or_dataframe._slice(key.start, key.stop, key.step)
    elif isinstance(key, list):
        raise NotImplementedError("iloc does not yet support indexing with a list")
    elif isinstance(key, tuple):
        raise NotImplementedError(
            "iloc does not yet support indexing with a (row, column) tuple"
        )
    elif callable(key):
        raise NotImplementedError("iloc does not yet support indexing with a callable")
    else:
        raise TypeError("Invalid argument type.")
