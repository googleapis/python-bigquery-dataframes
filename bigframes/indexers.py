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

import bigframes.core as core
import bigframes.core.indexes.index
import bigframes.series


class LocSeriesIndexer:
    def __init__(self, series: bigframes.series.Series):
        self._series = series

    def __getitem__(self, key) -> bigframes.Series:
        """
        Only indexing by a boolean bigframes.Series or list of index entries is currently supported
        """
        return typing.cast(
            bigframes.Series, _loc_getitem_series_or_dataframe(self._series, key)
        )

    def __setitem__(self, key, value) -> None:
        # TODO(swast): support MultiIndex
        if isinstance(key, slice):
            # TODO(swast): Implement loc with slices.
            raise NotImplementedError("loc does not yet support slices")
        elif isinstance(key, list):
            # TODO(tbergeron): Implement loc for index label list.
            raise NotImplementedError("loc does not yet support index label lists")

        # Assume the key is for the index label.
        block = self._series._block
        value_column = self._series._value
        index_column = block.expr.get_column(block.index_columns[0])
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
        new_expr = block.expr.projection(all_columns)

        # TODO(tbergeron): Use block operators rather than directly building desired ibis expressions.
        self._series._set_block(
            core.blocks.Block(
                new_expr,
                self._series._block.index_columns,
                self._series._block.column_labels,
                [self._series._block.index.name],
            )
        )


class IlocSeriesIndexer:
    def __init__(self, series: bigframes.series.Series):
        self._series = series

    def __getitem__(self, key) -> bigframes.scalar.Scalar | bigframes.series.Series:
        """
        Index series using integer offsets. Currently supports index by key type:

        slice: ex. series.iloc[2:5] returns values at index 2, 3, and 4 as a series
        individual offset: ex. series.iloc[0] returns value at index 0 as a scalar
        list: ex. series.iloc[1, 1, 2, 0] returns a series with the index 1 item repeated
        twice, followed by the index 2 and then and 0 items in that order.

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
        return typing.cast(
            bigframes.DataFrame, _loc_getitem_series_or_dataframe(self._dataframe, key)
        )


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


def _loc_getitem_series_or_dataframe(
    series_or_dataframe: bigframes.DataFrame | bigframes.Series, key
) -> bigframes.DataFrame | bigframes.Series:
    if isinstance(key, bigframes.Series) and key.dtype == "boolean":
        return series_or_dataframe[key]
    elif isinstance(key, bigframes.Series):
        # TODO(henryjsolberg): support MultiIndex
        temp_name = bigframes.guid.generate_guid(prefix="temp_series_name_")
        key = key.rename(temp_name)
        keys_df = key.to_frame()
        keys_df = keys_df.set_index(temp_name, drop=True)
        return _perform_loc_list_join(series_or_dataframe, keys_df)
    elif isinstance(key, bigframes.core.indexes.Index):
        # TODO(henryjsolberg): support MultiIndex
        block = key._data._get_block()
        temp_labels = [
            label
            if label
            else bigframes.guid.generate_guid(prefix="temp_column_label_")
            for label in block.column_labels
        ]
        block = block.with_column_labels(temp_labels)
        block = block.drop_columns(temp_labels)
        keys_df = bigframes.DataFrame(block)
        return _perform_loc_list_join(series_or_dataframe, keys_df)
    elif pd.api.types.is_list_like(key):
        # TODO(henryjsolberg): support MultiIndex
        if len(key) == 0:
            return typing.cast(
                typing.Union[bigframes.DataFrame, bigframes.Series],
                series_or_dataframe.iloc[0:0],
            )
        keys_df = bigframes.DataFrame(
            {"old_index": key}, session=series_or_dataframe._get_block().expr._session
        )
        keys_df = keys_df.set_index("old_index", drop=True)
        return _perform_loc_list_join(series_or_dataframe, keys_df)
    elif isinstance(key, slice):
        raise NotImplementedError("loc does not yet support indexing with a slice")
    elif callable(key):
        raise NotImplementedError("loc does not yet support indexing with a callable")
    else:
        raise TypeError(
            "Invalid argument type. loc currently only supports indexing with a boolean bigframes Series or a list of index entries."
        )


def _perform_loc_list_join(
    series_or_dataframe: bigframes.Series | bigframes.DataFrame,
    keys_df: bigframes.DataFrame,
) -> bigframes.Series | bigframes.DataFrame:
    # right join based on the old index so that the matching rows from the user's
    # original dataframe will be duplicated and reordered appropriately
    original_index_name = series_or_dataframe.index.name
    if isinstance(series_or_dataframe, bigframes.Series):
        original_name = series_or_dataframe.name
        name = series_or_dataframe.name if series_or_dataframe.name is not None else "0"
        result = series_or_dataframe.to_frame().join(keys_df, how="right")[name]
        result = typing.cast(bigframes.Series, result)
        result = result.rename(original_name)
    else:
        result = series_or_dataframe.join(keys_df, how="right")
    result = result.rename_axis(original_index_name)
    return result


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
    elif pd.api.types.is_list_like(key):
        # TODO(henryjsolberg): support MultiIndex

        if len(key) == 0:
            return typing.cast(
                typing.Union[bigframes.DataFrame, bigframes.Series],
                series_or_dataframe.iloc[0:0],
            )
        df = series_or_dataframe
        if isinstance(series_or_dataframe, bigframes.Series):
            original_series_name = series_or_dataframe.name
            series_name = (
                original_series_name if original_series_name is not None else "0"
            )
            df = series_or_dataframe.to_frame()
        original_index_name = df.index.name
        temporary_index_name = bigframes.guid.generate_guid(prefix="temp_iloc_index_")
        df = df.rename_axis(temporary_index_name)

        # set to offset index and use regular loc, then restore index
        df = df.reset_index(drop=False)
        result = df.loc[key]
        result = result.set_index(temporary_index_name)
        result = result.rename_axis(original_index_name)

        if isinstance(series_or_dataframe, bigframes.Series):
            result = result[series_name]
            result = typing.cast(bigframes.Series, result)
            result = result.rename(original_series_name)

        return result

    elif isinstance(key, tuple):
        raise NotImplementedError(
            "iloc does not yet support indexing with a (row, column) tuple"
        )
    elif callable(key):
        raise NotImplementedError("iloc does not yet support indexing with a callable")
    else:
        raise TypeError("Invalid argument type.")
