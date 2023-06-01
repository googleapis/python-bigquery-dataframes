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

# Type constraint for wherever column labels are used
Label = typing.Optional[str]


class Block:
    """A mutable 2D data structure."""

    def __init__(
        self,
        expr: core.BigFramesExpr,
        index_columns: Iterable[str] = (),
        column_labels: Optional[Sequence[Label]] = None,
        index_labels: Optional[Sequence[Label]] = None,
    ):
        """Construct a block object, will create default index if no index columns specified."""
        if index_labels and (len(index_labels) != len(list(index_columns))):
            raise ValueError(
                "'index_columns' and 'index_labels' must have equal length"
            )
        if len(list(index_columns)) == 0:
            new_index_col_id = guid.generate_guid(prefix="index_")
            expr = expr.promote_offsets(new_index_col_id)
            index_columns = [new_index_col_id]
        if len(list(index_columns)) > 1:
            raise NotImplementedError("MultiIndex not supported")
        self._index_columns = tuple(index_columns)
        index_labels = (
            tuple(index_labels)
            if index_labels
            else tuple([None for _ in index_columns])
        )
        self._index = indexes.Index(self, self._index_columns[0], name=index_labels[0])
        self._expr = self._normalize_expression(expr, self._index_columns)
        # TODO(tbergeron): Force callers to provide column labels
        self._column_labels = (
            tuple(column_labels) if column_labels else tuple(self.value_columns)
        )
        if len(self.value_columns) != len(self._column_labels):
            raise ValueError(
                "'index_columns' and 'index_labels' must have equal length"
            )

    @property
    def index(self) -> indexes.Index:
        """Row identities for values in the Block."""
        # TODO(285636739): Derive index value from block value
        return self._index

    @index.setter
    def index(self, value: indexes.Index):
        # TODO(swast): We shouldn't allow changing the index, as that'll break
        # references to this block from existing Index objects.
        self._expr = self._normalize_expression(value._expr, (value._index_column,))
        self._index_columns = (value._index_column,)
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
        self._sync_to_index()
        self._expr = self._normalize_expression(self._expr, self._index_columns)

    @property
    def value_columns(self) -> Sequence[str]:
        """All value columns, mutually exclusive with index columns."""
        return [
            column
            for column in self._expr.column_names
            if column not in self.index_columns
        ]

    @property
    def column_labels(self) -> List[Label]:
        return list(self._column_labels)

    @property
    def expr(self) -> core.BigFramesExpr:
        """Expression representing all columns, including index columns."""
        return self._expr

    @expr.setter
    def expr(self, expr: core.BigFramesExpr):
        # WARNING: Can corrupt block. Must make sure labels is updated to reflect changes.
        self._expr = self._normalize_expression(expr, self._index_columns)

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

    def reset_index(self, drop: bool = True) -> Block:
        """Reset the index of the block, promoting the old index to a value column.

        Arguments:
            name: this is the column id for the new value id derived from the old index

        Returns:
            A new Block because dropping index columns can break references
            from Index classes that point to this block.
        """
        block = self.copy()
        new_index_col_id = guid.generate_guid(prefix="index_")
        expr = self._expr.promote_offsets(new_index_col_id)
        if drop:
            # Even though the index might be part of the ordering, keep that
            # ordering expression as reset_index shouldn't change the row
            # order.
            expr = expr.drop_columns(self.index_columns)
            block = Block(
                expr,
                index_columns=[new_index_col_id],
                column_labels=self.column_labels,
                index_labels=[None],
            )
        else:
            # TODO(swast): Support MultiIndex
            index_label = self.index.name
            if index_label is None:
                if "index" not in self.column_labels:
                    index_label = "index"
                else:
                    index_label = "level_0"

            if index_label in self.column_labels:
                raise ValueError(f"cannot insert {index_label}, already exists")

            block = Block(
                expr,
                index_columns=[new_index_col_id],
                column_labels=[index_label, *self.column_labels],
                index_labels=[None],
            )
        return block

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

    def copy(
        self,
        value_columns: Optional[Iterable[ibis_types.Value]] = None,
        column_labels: Optional[Sequence[Label]] = None,
    ) -> Block:
        """Create a copy of this Block, replacing value columns if desired."""
        # BigFramesExpr and Tuple are immutable, so just need a new wrapper.
        if value_columns is not None:
            expr = self._project_value_columns(value_columns)
        else:
            expr = self._expr

        # TODO(swast): Support MultiIndex.
        return Block(
            expr,
            index_columns=self._index_columns,
            column_labels=column_labels
            if (column_labels is not None)
            else self._column_labels,
            index_labels=[self.index.name],
        )

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

    def replace_column_labels(self, value: List[Label]):
        if len(value) != len(self.value_columns):
            raise ValueError(
                f"The column labels size `{len(value)} ` should equal to the value"
                + f"columns size: {len(self.value_columns)}."
            )
        self._column_labels = tuple(value)

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
        # TODO(tbergeron): handle labels safely so callers don't need to
        self.expr = self._expr.project_unary_op(column, op, output_name)

    def apply_binary_op(
        self,
        left_column_id: str,
        right_column_id: str,
        op: ops.BinaryOp,
        output_id: str,
    ):
        # TODO(tbergeron): handle labels safely so callers don't need to
        self.expr = self._expr.project_binary_op(
            left_column_id, right_column_id, op, output_id
        )

    def apply_ternary_op(
        self,
        col_id_1: str,
        col_id_2: str,
        col_id_3: str,
        op: ops.TernaryOp,
        output_id: str,
    ):
        # TODO(tbergeron): handle labels safely so callers don't need to
        self.expr = self._expr.project_ternary_op(
            col_id_1, col_id_2, col_id_3, op, output_id
        )

    def apply_window_op(
        self,
        column: str,
        op: agg_ops.WindowOp,
        window_spec: core.WindowSpec,
        output_name=None,
        *,
        skip_null_groups: bool = False,
        skip_reproject_unsafe: bool = False,
    ):
        self.expr = self._expr.project_window_op(
            column,
            op,
            window_spec,
            output_name,
            skip_null_groups=skip_null_groups,
            skip_reproject_unsafe=skip_reproject_unsafe,
        )

    def assign_column(self, source_column_id: str, destination_column_id: str) -> Block:
        block = self.copy()
        block.expr = block.expr.assign(source_column_id, destination_column_id)
        if destination_column_id not in self.value_columns:
            block.replace_column_labels([*self.column_labels, None])
        return block

    def assign_constant(
        self,
        column_id: str,
        scalar_constant: typing.Any,
        label: typing.Optional[str] = None,
    ) -> Block:
        block = self.copy()
        block.expr = block.expr.assign_constant(column_id, scalar_constant)
        if column_id not in self.value_columns:
            block.replace_column_labels([*self.column_labels, label])
        elif label:
            block = block.assign_label(column_id, label)
        return block

    def assign_label(self, column_id: str, new_label: Label) -> Block:
        col_index = self.value_columns.index(column_id)
        new_labels = list(self.column_labels)
        new_labels[col_index] = new_label
        return self.copy(column_labels=new_labels)

    def filter(self, column_name: str):
        condition = typing.cast(
            ibis_types.BooleanValue, self._expr.get_column(column_name)
        )
        self.expr = self.expr.filter(condition)

    def aggregate_all_and_pivot(
        self,
        operation: agg_ops.AggregateOp,
        *,
        value_col_id: str = "values",
        dropna: bool = True,
    ) -> Block:
        aggregations = [
            (col_id, operation, col_id)
            for col_id, dtype in zip(self.value_columns, self.dtypes)
            if (dtype in bigframes.dtypes.NUMERIC_BIGFRAMES_TYPES)
        ]
        result_expr = self.expr.aggregate(
            aggregations, dropna=dropna
        ).transpose_single_row(
            labels=self.column_labels, index_col_id="index", value_col_id=value_col_id
        )
        return Block(result_expr, index_columns=["index"], column_labels=[None])

    def select_column(self, id: str) -> Block:
        return self.drop_columns(
            [col_id for col_id in self.value_columns if col_id != id]
        )

    def drop_columns(self, ids_to_drop: typing.Sequence[str]) -> Block:
        """Drops columns by id. Can drop index"""
        if set(ids_to_drop) & set(self.index_columns):
            raise ValueError(
                "Cannot directly drop index column. Use reset_index(drop=True)"
            )
        expr = self._expr.drop_columns(ids_to_drop)
        remaining_value_col_ids = [
            col_id for col_id in self.value_columns if (col_id not in ids_to_drop)
        ]
        labels = self._get_labels_for_columns(remaining_value_col_ids)
        return Block(expr, self.index_columns, labels, [self.index.name])

    def aggregate(
        self,
        by_column_ids: typing.Sequence[str],
        aggregations: typing.Sequence[typing.Tuple[str, agg_ops.AggregateOp, str]],
        *,
        as_index: bool = True,
        dropna: bool = True,
    ) -> Block:
        """
        Apply aggregations to the block. Callers responsible for setting index column(s) after.
        Arguments:
            by_column_id: column id of the aggregation key, this is preserved through the transform and used as index
            aggregations: input_column_id, operation, output_column_id tuples
            as_index: if True, grouping keys will be index columns in result, otherwise they will be non-index columns.
            dropna: whether null keys should be dropped
        """
        result_expr = self.expr.aggregate(aggregations, by_column_ids, dropna=dropna)

        aggregate_labels = self._get_labels_for_columns(
            [agg[0] for agg in aggregations]
        )
        if as_index:
            # TODO: Generalize to multi-index
            by_col_id = by_column_ids[0]
            if by_col_id in self.index_columns:
                # Groupby level 0 case, keep index name
                index_name = self.index.name
            else:
                index_name = self._get_labels_for_columns(by_column_ids)[0]
            return Block(
                result_expr,
                index_columns=by_column_ids,
                column_labels=aggregate_labels,
                index_labels=[index_name],
            )
        else:
            by_column_labels = self._get_labels_for_columns(by_column_ids)
            labels = (*by_column_labels, *aggregate_labels)
            return Block(result_expr, column_labels=labels)

    def _get_labels_for_columns(self, column_ids: typing.Sequence[str]):
        """Get column label for value columns, or index name for index columns"""
        lookup = {
            col_id: label
            for col_id, label in zip(self.value_columns, self._column_labels)
        }
        return [lookup.get(col_id, None) for col_id in column_ids]

    def _normalize_expression(
        self,
        expr: core.BigFramesExpr,
        index_columns: typing.Sequence[str],
        assert_value_size: typing.Optional[int] = None,
    ):
        """Normalizes expression by moving index columns to left."""
        value_columns = [
            col_id for col_id in expr.column_names.keys() if col_id not in index_columns
        ]
        if (assert_value_size is not None) and (
            len(value_columns) != assert_value_size
        ):
            raise ValueError("Unexpected number of value columns.")
        return expr.select_columns([*index_columns, *value_columns])

    def _sync_to_index(self):
        """Update index to match latest expression and column(s).

        Index object contains a reference to the expression object so any changes to the block's expression requires building a new index object as well.
        """
        columns = self._index_columns
        if len(columns) == 0:
            raise ValueError("Expect at least one index column.")
        elif len(columns) == 1:
            index_column = columns[0]
            self._index = indexes.Index(self, index_column, name=self.index.name)
        else:
            raise NotImplementedError("MultiIndex not supported.")
