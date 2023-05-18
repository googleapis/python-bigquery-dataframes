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

"""Block is a 2D data structure that supports data mutability and views.

These data structures are shared by DataFrame and Series. This allows views to
link in both directions (DataFrame to Series and vice versa) and prevents
circular dependencies.
"""

from __future__ import annotations

import itertools
import typing
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import geopandas as gpd  # type: ignore
import ibis.expr.schema as ibis_schema
import ibis.expr.types as ibis_types
import numpy
import pandas as pd
import pyarrow as pa  # type: ignore

import bigframes.aggregations as agg_ops
import bigframes.core as core
import bigframes.core.indexes as indexes
import bigframes.dtypes
import bigframes.guid as guid
import bigframes.operations as ops


class Block:
    """A mutable 2D data structure."""

    def __init__(
        self,
        expr: core.BigFramesExpr,
        index_columns: Iterable[str] = (),
    ):
        self._expr = expr
        self._index = indexes.ImplicitJoiner(self)
        self._index_columns = tuple(index_columns)
        self._sync_index()

    @property
    def index(self) -> Union[indexes.ImplicitJoiner, indexes.Index]:
        """Row identities for values in the Block."""
        return self._index

    @index.setter
    def index(self, value: indexes.ImplicitJoiner):
        # TODO(swast): We shouldn't allow changing the index, as that'll break
        # references to this block from existing Index objects.
        self._expr = value._expr
        if isinstance(value, indexes.Index):
            self._index_columns = (value._index_column,)
        else:
            self._index_columns = ()
        self._index = value

    @property
    def index_columns(self) -> Sequence[str]:
        """Column(s) to use as row labels."""
        return self._index_columns

    @index_columns.setter
    def index_columns(self, value: Iterable[str]):
        # TODO(swast): We shouldn't allow changing the index, as that'll break
        # references to this block from existing Index objects.
        self._index_columns = tuple(value)
        self._sync_index()

    @property
    def value_columns(self) -> Sequence[str]:
        """All value columns, mutually exclusive with index columns."""
        return [
            column
            for column in self._expr.column_names
            if column not in self.index_columns
        ]

    @property
    def expr(self) -> core.BigFramesExpr:
        """Expression representing all columns, including index columns."""
        return self._expr

    @expr.setter
    def expr(self, expr: core.BigFramesExpr):
        self._expr = expr
        self._sync_index()

    @property
    def dtypes(
        self,
    ) -> Sequence[Union[bigframes.dtypes.BigFramesDtype, numpy.dtype[typing.Any]]]:
        """Returns the dtypes as a Pandas Series object"""
        ibis_dtypes = [
            dtype
            for col, dtype in self.expr.to_ibis_expr(ordering_mode="unordered")
            .schema()
            .items()
            if col not in self.index_columns
        ]
        return [
            bigframes.dtypes.ibis_dtype_to_bigframes_dtype(ibis_dtype)
            for ibis_dtype in ibis_dtypes
        ]

    def reset_index(self) -> Block:
        """Reset the index of the block, promoting the old index to a value column.

        Arguments:
            name: this is the column id for the new value id derived from the old index

        Returns:
            A new Block because dropping index columns can break references
            from Index classes that point to this block.
        """
        block = self.copy()
        new_index_col_id = guid.generate_guid(prefix="index_")
        block.expr = self._expr.promote_offsets(new_index_col_id)

        # TODO(swast): Only remove a specified number of levels from a
        # MultiIndex.
        block.index_columns = [new_index_col_id]
        block._sync_index()
        block.index.name = None
        return block

    def _sync_index(self):
        """Update index to match latest expression and column(s).

        Index object contains a reference to the expression object so any changes to the block's expression requires building a new index object as well.
        """
        expr = self._expr
        columns = self._index_columns
        if len(columns) == 0:
            self._index = indexes.ImplicitJoiner(self, self._index.name)
        elif len(columns) == 1:
            index_column = columns[0]
            self._index = indexes.Index(self, index_column, name=self._index.name)
            # Rearrange so that index columns are first.
            if expr._columns and expr._columns[0].get_name() != index_column:
                expr_builder = expr.builder()
                index_columns = [
                    column
                    for column in expr_builder.columns
                    if column.get_name() == index_column
                ]
                value_columns = [
                    column
                    for column in expr_builder.columns
                    if column.get_name() != index_column
                ]
                expr_builder.columns = index_columns + value_columns
                # Avoid infinite loops by bypassing the property setter.
                self._expr = expr_builder.build()
        else:
            raise NotImplementedError("MultiIndex not supported.")

    def _to_dataframe(self, result, schema: ibis_schema.Schema) -> pd.DataFrame:
        """Convert BigQuery data to pandas DataFrame with specific dtypes."""
        df = result.to_dataframe(
            bool_dtype=pd.BooleanDtype(),
            int_dtype=pd.Int64Dtype(),
            float_dtype=pd.Float64Dtype(),
            string_dtype=pd.StringDtype(storage="pyarrow"),
            date_dtype=pd.ArrowDtype(pa.date32()),
            datetime_dtype=pd.ArrowDtype(pa.timestamp("us")),
            time_dtype=pd.ArrowDtype(pa.time64("us")),
            timestamp_dtype=pd.ArrowDtype(pa.timestamp("us", tz="UTC")),
        )

        # Convert Geography column from StringDType to GeometryDtype.
        for column_name, ibis_dtype in schema.items():
            if ibis_dtype.is_geospatial():
                df[column_name] = gpd.GeoSeries.from_wkt(
                    # https://github.com/geopandas/geopandas/issues/1879
                    df[column_name].replace({numpy.nan: None}),
                    # BigQuery geography type is based on the WGS84 reference ellipsoid.
                    crs="EPSG:4326",
                )
        return df

    def compute(
        self, value_keys: Optional[Iterable[str]] = None, max_results=None
    ) -> pd.DataFrame:
        """Run query and download results as a pandas DataFrame."""
        df, _ = self._compute_and_count(value_keys=value_keys, max_results=max_results)
        return df

    def _compute_and_count(
        self, value_keys: Optional[Iterable[str]] = None, max_results=None
    ) -> Tuple[pd.DataFrame, int]:
        """Run query and download results as a pandas DataFrame. Return the total number of results as well."""
        # TODO(swast): Allow for dry run and timeout.
        expr = self._expr

        value_column_names = value_keys or self.value_columns
        if value_keys is not None:
            index_columns = (
                expr.get_column(column_name) for column_name in self._index_columns
            )
            value_columns = (expr.get_column(column_name) for column_name in value_keys)
            expr = expr.projection(itertools.chain(index_columns, value_columns))

        results_iterator = expr.start_query().result(max_results=max_results)
        df = self._to_dataframe(
            results_iterator,
            expr.to_ibis_expr().schema(),
        )

        df = df.loc[:, [*self.index_columns, *value_column_names]]
        if self.index_columns:
            df = df.set_index(list(self.index_columns))
            # TODO(swast): Set names for all levels with MultiIndex.
            df.index.name = typing.cast(indexes.Index, self.index).name

        return df, results_iterator.total_rows

    def copy(self, value_columns: Optional[Iterable[ibis_types.Value]] = None) -> Block:
        """Create a copy of this Block, replacing value columns if desired."""
        # BigFramesExpr and Tuple are immutable, so just need a new wrapper.
        if value_columns is not None:
            expr = self._project_value_columns(value_columns)
        else:
            expr = self._expr

        block = Block(expr, self._index_columns)

        # TODO(swast): Support MultiIndex.
        block.index.name = self.index.name
        return block

    def _project_value_columns(
        self, value_columns: Iterable[ibis_types.Value]
    ) -> core.BigFramesExpr:
        """Return a related expression with new value columns."""
        columns = []
        index_columns = (
            self._expr.get_column(column_name) for column_name in self._index_columns
        )
        for column in itertools.chain(index_columns, value_columns):
            columns.append(column)
        return self._expr.projection(columns)

    def replace_value_columns(self, value_columns: Iterable[ibis_types.Value]):
        """Mutate all value columns in-place."""
        # TODO(swast): Deprecate this method, as it allows for operations
        # directly on Ibis values, which we want to discourage from higher
        # levels.
        # TODO(swast): Should we validate that we aren't dropping any columns
        # so that predicates are valid and column labels stay in sync in
        # DataFrame.
        self.expr = self._project_value_columns(value_columns)

    def get_value_col_exprs(
        self, column_names: Optional[Sequence[str]] = None
    ) -> List[ibis_types.Value]:
        """Retrive value column expressions."""
        column_names = self.value_columns if column_names is None else column_names
        return [self._expr.get_column(column_name) for column_name in column_names]

    def shape(self) -> typing.Tuple[int, int]:
        """Returns dimensions as (length, width) tuple."""
        impl_length, impl_width = self._expr.shape()
        return (impl_length, impl_width - len(self.index_columns))

    def apply_unary_op(self, column: str, op: ops.UnaryOp, output_name=None):
        self.expr = self._expr.project_unary_op(column, op, output_name)

    def project_binary_op(
        self,
        left_column_id: str,
        right_column_id: str,
        op: ops.BinaryOp,
        output_id: str,
    ):
        self.expr = self._expr.project_binary_op(
            left_column_id, right_column_id, op, output_id
        )

    def apply_window_op(
        self,
        column: str,
        op: agg_ops.WindowOp,
        window_spec: core.WindowSpec,
        output_name=None,
        *,
        skip_reproject_unsafe: bool = False,
    ):
        self.expr = self._expr.project_window_op(
            column,
            op,
            window_spec,
            output_name,
            skip_reproject_unsafe=skip_reproject_unsafe,
        )

    def filter(self, column_name: str):
        condition = typing.cast(
            ibis_types.BooleanValue, self._expr.get_column(column_name)
        )
        self.expr = self.expr.filter(condition)

    def aggregate(
        self,
        by_column_ids: typing.Sequence[str],
        aggregations: typing.Sequence[typing.Tuple[str, agg_ops.AggregateOp, str]],
        dropna: bool = True,
    ) -> Block:
        """
        Apply aggregations to the block. Callers responsible for setting index column(s) after.
        Arguments:
            by_column_id: column id of the aggregation key, this is preserved through the transform and used as index
            aggregations: input_column_id, operation, output_column_id tuples
            dropna: whether null keys should be dropped
        """
        # Note: 'by' columns will be moved to the front
        result_expr = self.expr.aggregate(by_column_ids, aggregations, dropna)
        return Block(result_expr)
