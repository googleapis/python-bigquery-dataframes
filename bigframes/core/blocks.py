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

import functools
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
import bigframes.core.ordering as ordering
import bigframes.dtypes
import bigframes.guid as guid
import bigframes.operations as ops

# Type constraint for wherever column labels are used
Label = typing.Optional[str]


class BlockHolder(typing.Protocol):
    """Interface for mutable objects with state represented by a block value object."""

    def _set_block(self, block: Block):
        """Set the underlying block value of the object"""

    def _get_block(self) -> Block:
        """Get the underlying block value of the object"""


class Block:
    """A immutable 2D data structure."""

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
        self._index_labels = (
            tuple(index_labels)
            if index_labels
            else tuple([None for _ in index_columns])
        )
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
    def index(self) -> indexes.IndexValue:
        """Row identities for values in the Block."""
        return indexes.IndexValue(self)

    @property
    def index_columns(self) -> Sequence[str]:
        """Column(s) to use as row labels."""
        return self._index_columns

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

    @functools.cached_property
    def col_id_to_label(self) -> typing.Mapping[str, Label]:
        """Get column label for value columns, or index name for index columns"""
        return {
            col_id: label
            for col_id, label in zip(self.value_columns, self._column_labels)
        }

    @functools.cached_property
    def label_to_col_id(self) -> typing.Mapping[Label, typing.Sequence[str]]:
        """Get column label for value columns, or index name for index columns"""
        mapping: typing.Dict[Label, typing.Sequence[str]] = {}
        for id, label in self.col_id_to_label.items():
            mapping[label] = (*mapping.get(label, ()), id)
        return mapping

    def order_by(
        self,
        by: typing.Sequence[ordering.OrderingColumnReference],
        stable: bool = False,
    ) -> Block:
        return Block(
            self._expr.order_by(by, stable=stable),
            index_columns=self.index_columns,
            column_labels=self.column_labels,
            index_labels=[self.index.name],
        )

    def reversed(self) -> Block:
        return Block(
            self._expr.reversed(),
            index_columns=self.index_columns,
            column_labels=self.column_labels,
            index_labels=[self.index.name],
        )

    def reset_index(self, drop: bool = True) -> Block:
        """Reset the index of the block, promoting the old index to a value column.

        Arguments:
            name: this is the column id for the new value id derived from the old index

        Returns:
            A new Block because dropping index columns can break references
            from Index classes that point to this block.
        """
        block = self
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
            df.index.name = self.index.name

        return df, results_iterator.total_rows

    def with_column_labels(self, value: List[Label]) -> Block:
        if len(value) != len(self.value_columns):
            raise ValueError(
                f"The column labels size `{len(value)} ` should equal to the value"
                + f"columns size: {len(self.value_columns)}."
            )
        return Block(
            self._expr,
            index_columns=self.index_columns,
            column_labels=tuple(value),
            index_labels=[self.index.name],
        )

    def with_index_labels(self, value: List[Label]) -> Block:
        if len(value) != len(self.index_columns):
            raise ValueError(
                f"The index labels size `{len(value)} ` should equal to the index"
                + f"columns size: {len(self.value_columns)}."
            )
        return Block(
            self._expr,
            index_columns=self.index_columns,
            column_labels=self.column_labels,
            index_labels=tuple(value),
        )

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

    def apply_unary_op(
        self, column: str, op: ops.UnaryOp, result_label: Label = None
    ) -> typing.Tuple[Block, str]:
        """
        Apply a unary op to the block. Creates a new column to store the result.
        """
        # TODO(tbergeron): handle labels safely so callers don't need to
        result_id = guid.generate_guid()
        expr = self._expr.project_unary_op(column, op, result_id)
        block = Block(
            expr,
            index_columns=self.index_columns,
            column_labels=[*self.column_labels, result_label],
            index_labels=[self.index.name],
        )
        return (block, result_id)

    def apply_binary_op(
        self,
        left_column_id: str,
        right_column_id: str,
        op: ops.BinaryOp,
        result_label: Label = None,
    ) -> typing.Tuple[Block, str]:
        result_id = guid.generate_guid()
        expr = self._expr.project_binary_op(
            left_column_id, right_column_id, op, result_id
        )
        block = Block(
            expr,
            index_columns=self.index_columns,
            column_labels=[*self.column_labels, result_label],
            index_labels=[self.index.name],
        )
        return (block, result_id)

    def apply_ternary_op(
        self,
        col_id_1: str,
        col_id_2: str,
        col_id_3: str,
        op: ops.TernaryOp,
        result_label: Label = None,
    ) -> typing.Tuple[Block, str]:
        result_id = guid.generate_guid()
        expr = self._expr.project_ternary_op(
            col_id_1, col_id_2, col_id_3, op, result_id
        )
        block = Block(
            expr,
            index_columns=self.index_columns,
            column_labels=[*self.column_labels, result_label],
            index_labels=[self.index.name],
        )
        return (block, result_id)

    def multi_apply_window_op(
        self,
        columns: typing.Sequence[str],
        op: agg_ops.WindowOp,
        window_spec: core.WindowSpec,
        *,
        skip_null_groups: bool = False,
    ) -> Block:
        block = self
        for i, col_id in enumerate(columns):
            label = self.col_id_to_label[col_id]
            block, result_id = block.apply_window_op(
                col_id,
                op,
                window_spec=window_spec,
                skip_reproject_unsafe=(i + 1) < len(columns),
                result_label=label,
                skip_null_groups=skip_null_groups,
            )
            block = block.copy_values(result_id, col_id)
            block = block.drop_columns(result_id)
        return block

    def apply_window_op(
        self,
        column: str,
        op: agg_ops.WindowOp,
        window_spec: core.WindowSpec,
        *,
        result_label: Label = None,
        skip_null_groups: bool = False,
        skip_reproject_unsafe: bool = False,
    ) -> typing.Tuple[Block, str]:
        result_id = guid.generate_guid()
        expr = self._expr.project_window_op(
            column,
            op,
            window_spec,
            result_id,
            skip_null_groups=skip_null_groups,
            skip_reproject_unsafe=skip_reproject_unsafe,
        )
        block = Block(
            expr,
            index_columns=self.index_columns,
            column_labels=[*self.column_labels, result_label],
            index_labels=self._index_labels,
        )
        return (block, result_id)

    def copy_values(self, source_column_id: str, destination_column_id: str) -> Block:
        expr = self.expr.assign(source_column_id, destination_column_id)
        return Block(
            expr,
            index_columns=self.index_columns,
            column_labels=self.column_labels,
            index_labels=self._index_labels,
        )

    def create_constant(
        self,
        scalar_constant: typing.Any,
        label: Label = None,
    ) -> typing.Tuple[Block, str]:
        result_id = guid.generate_guid()
        expr = self.expr.assign_constant(result_id, scalar_constant)
        labels = [*self.column_labels, label]
        return (
            Block(
                expr,
                index_columns=self.index_columns,
                column_labels=labels,
                index_labels=self._index_labels,
            ),
            result_id,
        )

    def assign_label(self, column_id: str, new_label: Label) -> Block:
        col_index = self.value_columns.index(column_id)
        new_labels = list(self.column_labels)
        new_labels[col_index] = new_label
        return self.with_column_labels(new_labels)

    def filter(self, column_name: str):
        condition = typing.cast(
            ibis_types.BooleanValue, self._expr.get_column(column_name)
        )
        filtered_expr = self.expr.filter(condition)
        return Block(
            filtered_expr,
            index_columns=self.index_columns,
            column_labels=self.column_labels,
            index_labels=self._index_labels,
        )

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
        return self.select_columns([id])

    def select_columns(self, ids: typing.Sequence[str]) -> Block:
        expr = self._expr.select_columns([*self.index_columns, *ids])
        col_labels = self._get_labels_for_columns(ids)
        return Block(expr, self.index_columns, col_labels, [self.index.name])

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

    def rename(self, *, columns: typing.Mapping[Label, Label]):
        # TODO(tbergeron) Support function(Callable) as columns parameter.
        col_labels = [
            (columns.get(col_label, col_label)) for col_label in self.column_labels
        ]
        return self.with_column_labels(col_labels)

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
                index_name = self.col_id_to_label[by_col_id]
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
        lookup = self.col_id_to_label
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

    def slice(
        self: bigframes.core.blocks.Block,
        start: typing.Optional[int] = None,
        stop: typing.Optional[int] = None,
        step: typing.Optional[int] = None,
    ) -> bigframes.core.blocks.Block:
        sliced_expr = self.expr.slice(start=start, stop=stop, step=step)
        # since this is slice, return a copy even if unchanged
        block = Block(
            sliced_expr,
            index_columns=self.index_columns,
            column_labels=self.column_labels,
            index_labels=[self.index.name],
        )
        # TODO(swast): Support MultiIndex.
        return block

    def promote_offsets(self, label: Label = None) -> typing.Tuple[Block, str]:
        result_id = guid.generate_guid()
        return (
            Block(
                self._expr.promote_offsets(value_col_id=result_id),
                index_columns=self.index_columns,
                column_labels=[label, *self.column_labels],
                index_labels=self._index_labels,
            ),
            result_id,
        )

    def add_prefix(self, prefix: str, axis: str | int | None = None) -> Block:
        axis_number = _get_axis_number(axis)
        if axis_number == 0:
            expr = self._expr
            for index_col in self._index_columns:
                expr = expr.project_unary_op(index_col, ops.AsTypeOp("string"))
                prefix_op = ops.BinopPartialLeft(ops.add_op, prefix)
                expr = expr.project_unary_op(index_col, prefix_op)
            return Block(
                expr,
                index_columns=self.index_columns,
                column_labels=self.column_labels,
                index_labels=self._index_labels,
            )
        if axis_number == 1:
            expr = self._expr
            return Block(
                self._expr,
                index_columns=self.index_columns,
                column_labels=[f"{prefix}{label}" for label in self.column_labels],
                index_labels=self._index_labels,
            )

    def add_suffix(self, suffix: str, axis: str | int | None = None) -> Block:
        axis_number = _get_axis_number(axis)
        if axis_number == 0:
            expr = self._expr
            for index_col in self._index_columns:
                expr = expr.project_unary_op(index_col, ops.AsTypeOp("string"))
                prefix_op = ops.BinopPartialRight(ops.add_op, suffix)
                expr = expr.project_unary_op(index_col, prefix_op)
            return Block(
                expr,
                index_columns=self.index_columns,
                column_labels=self.column_labels,
                index_labels=self._index_labels,
            )
        if axis_number == 1:
            expr = self._expr
            return Block(
                self._expr,
                index_columns=self.index_columns,
                column_labels=[f"{label}{suffix}" for label in self.column_labels],
                index_labels=self._index_labels,
            )


def _get_axis_number(axis: str | int | None) -> typing.Literal[0, 1]:
    if axis in {0, "index", "rows", None}:
        return 0
    elif axis in {1, "columns"}:
        return 1
    else:
        raise ValueError(f"Not a valid axis: {axis}")
