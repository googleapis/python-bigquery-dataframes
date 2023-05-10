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

from dataclasses import dataclass
import functools
import math
import typing
from typing import Collection, Dict, Iterable, Optional, Sequence

from google.cloud import bigquery
import ibis
import ibis.expr.types as ibis_types

import bigframes.aggregations as agg_ops
from bigframes.core.ordering import (
    ExpressionOrdering,
    OrderingDirection,
    stringify_order_id,
)
import bigframes.guid
import bigframes.operations as ops

if typing.TYPE_CHECKING:
    from bigframes.session import Session


ORDER_ID_COLUMN = "bigframes_ordering_id"
PREDICATE_COLUMN = "bigframes_predicate"


@dataclass(frozen=True)
class WindowSpec:
    """
    Specifies a window over which aggregate and analytic function may be applied.
    grouping_keys: set of column ids to group on
    preceding: Number of preceding rows in the window
    following: Number of preceding rows in the window
    ordering: List of columns ids and ordering direction to override base ordering
    """

    grouping_keys: typing.Sequence[str] = tuple()
    ordering: typing.Sequence[typing.Tuple[str, OrderingDirection]] = tuple()
    preceding: typing.Optional[int] = None
    following: typing.Optional[int] = None


# TODO(swast): We might want to move this to it's own sub-module.
class BigFramesExpr:
    """Immutable BigFrames expression tree.
    Note: Usage of this class is considered to be private and subject to change
    at any time.
    This class is a wrapper around Ibis expressions. Its purpose is to defer
    Ibis projection operations to keep generated SQL small and correct when
    mixing and matching columns from different versions of a DataFrame.
    Args:
        session:
            A BigFrames session to allow more flexibility in running
            queries.
        table: An Ibis table expression.
        columns: Ibis value expressions that can be projected as columns.
        meta_columns: Ibis value expressions to store ordering.
        ordering: An ordering property of the data frame.
        predicates: A list of filters on the data frame.
    """

    def __init__(
        self,
        session: Session,
        table: ibis_types.Table,
        columns: Optional[Sequence[ibis_types.Value]] = None,
        meta_columns: Optional[Sequence[ibis_types.Value]] = None,
        ordering: Optional[ExpressionOrdering] = None,
        predicates: Optional[Collection[ibis_types.BooleanValue]] = None,
    ):
        self._session = session
        self._table = table
        self._predicates = tuple(predicates) if predicates is not None else ()
        # TODO: Validate ordering
        self._ordering = ordering or ExpressionOrdering()
        # Allow creating a DataFrame directly from an Ibis table expression.
        if columns is None:
            self._columns = tuple(
                table[key]
                for key in table.columns
                if ordering is None or key != ordering.ordering_id
            )
        else:
            # TODO(swast): Validate that each column references the same table (or
            # no table for literal values).
            self._columns = tuple(columns)

        # Meta columns store ordering, or other data that doesn't correspond to dataframe columns
        self._meta_columns = tuple(meta_columns) if meta_columns is not None else ()

        # To allow for more efficient lookup by column name, create a
        # dictionary mapping names to column values.
        self._column_names = {column.get_name(): column for column in self._columns}
        self._meta_column_names = {
            column.get_name(): column for column in self._meta_columns
        }

    @property
    def table(self) -> ibis_types.Table:
        return self._table

    @property
    def predicates(self) -> typing.Tuple[ibis_types.BooleanValue, ...]:
        return self._predicates

    @property
    def reduced_predicate(self) -> typing.Optional[ibis_types.BooleanValue]:
        """Returns the frame's predicates as an equivalent boolean value, useful where a single predicate value is preferred."""
        return (
            _reduce_predicate_list(self._predicates).name(PREDICATE_COLUMN)
            if self._predicates
            else None
        )

    @property
    def columns(self) -> typing.Tuple[ibis_types.Value, ...]:
        return self._columns

    @property
    def column_names(self) -> Dict[str, ibis_types.Value]:
        return self._column_names

    @property
    def meta_columns(self) -> typing.Tuple[ibis_types.Value, ...]:
        return self._meta_columns

    @property
    def ordering(self) -> Sequence[ibis_types.Value]:
        """Returns a sequence of ibis values which can be directly used to order a table expression. Has direction modifiers applied."""
        if not self._ordering:
            return []
        else:
            values = [
                self.get_any_column(ordering_value)
                for ordering_value in self._ordering.all_ordering_columns
            ]
            if self._ordering.is_ascending:
                # TODO(swast): When we assign literals / scalars, we might not
                # have a true Column. Do we need to check this before trying to
                # sort by such a column?
                return [
                    ibis.asc(typing.cast(ibis_types.Column, value)) for value in values
                ]
            else:
                # TODO(swast): When we assign literals / scalars, we might not
                # have a true Column. Do we need to check this before trying to
                # sort by such a column?
                return [
                    ibis.desc(typing.cast(ibis_types.Column, value)) for value in values
                ]

    def builder(self) -> BigFramesExprBuilder:
        """Creates a mutable builder for expressions."""
        # Since BigFramesExpr is intended to be immutable (immutability offers
        # potential opportunities for caching, though we might need to introduce
        # more node types for that to be useful), we create a builder class.
        return BigFramesExprBuilder(
            self._session,
            self._table,
            self._columns,
            self._meta_columns,
            ordering=self._ordering,
            predicates=self._predicates,
        )

    def insert_column(self, index: int, column: ibis_types.Value) -> BigFramesExpr:
        expr = self.builder()
        expr.columns.insert(index, column)
        return expr.build()

    def drop_columns(self, columns: Iterable[str]) -> BigFramesExpr:
        # Must generate offsets if we are dropping a column that ordering depends on
        expr = self
        for ordering_column in set(columns).intersection(
            self._ordering.ordering_value_columns
        ):
            expr = self._hide_column(ordering_column)

        expr_builder = expr.builder()
        remain_cols = [
            column for column in expr.columns if column.get_name() not in columns
        ]
        expr_builder.columns = remain_cols
        return expr_builder.build()

    def get_column(self, key: str) -> ibis_types.Value:
        """Gets the Ibis expression for a given column."""
        if key not in self._column_names.keys():
            raise ValueError(
                "Column name {} not in set of values: {}".format(
                    key, self._column_names.keys()
                )
            )
        return typing.cast(ibis_types.Value, self._column_names[key])

    def get_any_column(self, key: str) -> ibis_types.Value:
        """Gets the Ibis expression for a given column. Will also get hidden meta columns."""
        all_columns = {**self._column_names, **self._meta_column_names}
        if key not in all_columns.keys():
            raise ValueError(
                "Column name {} not in set of values: {}".format(
                    key, all_columns.keys()
                )
            )
        return typing.cast(ibis_types.Value, all_columns[key])

    def _get_meta_column(self, key: str) -> ibis_types.Value:
        """Gets the Ibis expression for a given metadata column."""
        if key not in self._meta_column_names.keys():
            raise ValueError(
                "Column name {} not in set of values: {}".format(
                    key, self._meta_column_names.keys()
                )
            )
        return self._meta_column_names[key]

    def apply_limit(self, max_results: int) -> BigFramesExpr:
        table = self.to_ibis_expr().limit(max_results)
        # Since we make a new table expression, the old column references now
        # point to the wrong table. Use the BigFramesExpr constructor to make
        # sure we have the correct references.
        return BigFramesExpr(self._session, table)

    def filter(self, predicate: ibis_types.BooleanValue) -> BigFramesExpr:
        """Filter the table on a given expression, the predicate must be a boolean series aligned with the table expression."""
        expr = self.builder()
        if expr.ordering:
            expr.ordering = expr.ordering.with_is_sequential(False)
        expr.predicates = [*self._predicates, predicate]
        return expr.build()

    def order_by(
        self, by: Sequence[str], ascending=True, na_last=True
    ) -> BigFramesExpr:
        # TODO(tbergeron): Always append fully ordered OID to end to guarantee total ordering.
        sort_col_ids = by
        nullity_meta_columns = []
        if (ascending and na_last) or (not ascending and not na_last):
            # In sql, nulls are the "lowest" value, so we need to adjust to make them act as "highest"
            nullity_meta_columns = [
                self.get_any_column(col).isnull().name(col + "_nullity") for col in by
            ]
            nullity_meta_column_names = [col.get_name() for col in nullity_meta_columns]
            sort_col_ids = [
                val
                for pair in zip(nullity_meta_column_names, sort_col_ids)
                for val in pair
            ]
        expr_builder = self.builder()
        expr_builder.ordering = self._ordering.with_ordering_columns(
            sort_col_ids, ascending
        )
        expr_builder.meta_columns = [*self.meta_columns, *nullity_meta_columns]
        return expr_builder.build()

    @property
    def offsets(self):
        if not self._ordering.is_sequential:
            raise ValueError(
                "Expression does not have offsets. Generate them first using project_offsets."
            )
        return self._get_meta_column(self._ordering.ordering_id)

    def project_offsets(self) -> BigFramesExpr:
        """Create a new expression that contains offsets. Should only be executed when offsets are needed for an operations. Has no effect on expression semantics."""
        if self._ordering.is_sequential:
            return self

        # TODO(tbergeron): Enforce total ordering
        table = self.to_ibis_expr(
            ordering_mode="offset_col", order_col_name=ORDER_ID_COLUMN
        )
        columns = [table[column_name] for column_name in self._column_names]
        ordering = ExpressionOrdering(
            ordering_id_column=ORDER_ID_COLUMN, is_sequential=True
        )
        return BigFramesExpr(
            self._session,
            table,
            columns=columns,
            meta_columns=[table[ORDER_ID_COLUMN]],
            ordering=ordering,
        )

    def _hide_column(self, column_id) -> BigFramesExpr:
        """Pushes columns to metadata columns list. Used to hide ordering columns that have been dropped or destructively mutated."""
        expr_builder = self.builder()
        # Need to rename column as caller might be creating a new row with the same name but different values.
        # Can avoid this if don't allow callers to determine ids and instead generate unique ones in this class.

        new_name = bigframes.guid.generate_guid(prefix="bigframes_meta_")
        expr_builder.meta_columns = [
            *self._meta_columns,
            self.get_column(column_id).name(new_name),
        ]

        ordering_columns = [
            col if col != column_id else new_name
            for col in self._ordering.ordering_value_columns
        ]

        expr_builder.ordering = self._ordering.with_ordering_columns(
            ordering_columns, self._ordering.is_ascending
        )
        return expr_builder.build()

    def projection(self, columns: Iterable[ibis_types.Value]) -> BigFramesExpr:
        """Creates a new expression based on this expression with new columns."""
        # TODO(swast): We might want to do validation here that columns derive
        # from the same table expression instead of (in addition to?) at
        # construction time.

        expr = self
        for ordering_column in set(self.column_names.keys()).intersection(
            self._ordering.ordering_value_columns
        ):
            # Need to hide ordering columns that are being dropped. Alternatively, could project offsets
            expr = expr._hide_column(ordering_column)
        builder = expr.builder()
        builder.columns = list(columns)
        new_expr = builder.build()
        return new_expr

    def shape(self) -> typing.Tuple[int, int]:
        """Returns dimensions as (length, width) tuple."""
        width = len(self.columns)
        length_query = self._session.bqclient.query(
            self.to_ibis_expr(ordering_mode="unordered").count().compile()
        )
        length = next(length_query.result())[0]

        return (length, width)

    def concat(self, other: typing.Sequence[BigFramesExpr]) -> BigFramesExpr:
        """Append together multiple BigFramesExpressions."""
        if len(other) == 0:
            return self
        tables = []
        prefix_base = 10
        prefix_size = math.ceil(math.log(len(other) + 1, prefix_base))

        # Must normalize all ids to the same encoding size
        max_encoding_size = max(
            [objects._ordering._ordering_encoding_size for objects in [self, *other]]
        )
        for i, expr in enumerate([self, *other]):
            ordering_prefix = str(i).zfill(prefix_size)
            table = expr.to_ibis_expr(
                ordering_mode="ordered_col", order_col_name=ORDER_ID_COLUMN
            )
            # Rename the value columns based on horizontal offset before applying union.
            table = table.select(
                [
                    col
                    if col != ORDER_ID_COLUMN
                    else (
                        ordering_prefix
                        + stringify_order_id(table[ORDER_ID_COLUMN], max_encoding_size)
                    ).name(ORDER_ID_COLUMN)
                    for col in table.columns
                ]
            )
            tables.append(table)

        combined_table = ibis.union(*tables)
        ordering = ExpressionOrdering(
            ordering_id_column=ORDER_ID_COLUMN,
            ordering_encoding_size=prefix_size + max_encoding_size,
        )
        return BigFramesExpr(
            self._session,
            combined_table,
            columns=[
                combined_table[col]
                for col in combined_table.columns
                if col != ORDER_ID_COLUMN
            ],
            meta_columns=[combined_table[ORDER_ID_COLUMN]],
            ordering=ordering,
        )

    def project_unary_op(
        self, column_name: str, op: ops.UnaryOp, output_name=None
    ) -> BigFramesExpr:
        """Creates a new expression based on this expression with unary operation applied to one column."""
        value = op._as_ibis(self.get_column(column_name)).name(
            output_name or column_name
        )
        return self._set_or_replace_by_id(output_name or column_name, value)

    def project_window_op(
        self,
        column_name: str,
        op: agg_ops.WindowOp,
        window_spec: WindowSpec,
        output_name=None,
    ) -> BigFramesExpr:
        """
        Creates a new expression based on this expression with unary operation applied to one column.
        column_name: the id of the input column present in the expression
        op: the windowable operator to apply to the input column
        window_spec: a specification of the window over which to apply the operator
        output_name: the id to assign to the output of the operator, by default will replace input col if distinct output id not provided
        """
        column = typing.cast(ibis_types.Column, self.get_column(column_name))
        window = self._ibis_window_from_spec(window_spec)

        cumulative_value = op._as_ibis(column, window)
        if op.skips_nulls:
            cumulative_value = (
                ibis.case().when(column.isnull(), ibis.NA).else_(cumulative_value).end()
            )
        result = self._set_or_replace_by_id(
            output_name or column_name, cumulative_value
        )
        # TODO(tbergeron): Should defer this until second window is applied to avoid unnecessarily creating new table expressions every analytic op.
        return result._reproject_to_table()

    def to_ibis_expr(
        self,
        ordering_mode: str = "order_by",
        order_col_name=ORDER_ID_COLUMN,
    ):
        """Creates an Ibis table expression representing the DataFrame.

        BigFrames expression are sorted, so three options are avaiable to reflect this in the ibis expression.
        The default is that the expression will be ordered by an order_by clause.
        "order_by": The output table will not have an ordering column, however there will be an order_by clause applied to the ouput.
        "offset_col": Zero-based offsets are generated as a column, this will not sort the rows however.
        "ordered_col": An ordered column is provided in output table, without guarantee that the values are sequential
        "expose_metadata": All columns projected in table expression, including hidden columns. Output is not otherwise ordered
        "unordered": No ordering information will be provided in output. Only value columns are projected.

        For offset or ordered column, order_col_name can be used to assign the output label for the ordering column.
        If none is specified, the default column name will be 'bigrames_ordering_id'

        Args:
            with_offsets: Output will include 0-based offsets as a column if set to True
            ordering_mode: One of "order_by", "ordered_col", or "offset_col"

        Returns:
            An ibis expression representing the data help by the BigFramesExpression.

        """

        assert ordering_mode in (
            "order_by",
            "ordered_col",
            "offset_col",
            "expose_metadata",
            "unordered",
        )

        table = self._table

        columns = list(self._columns)

        hidden_ordering_columns = [
            col
            for col in self._ordering.all_ordering_columns
            if col not in self._column_names.keys()
        ]

        if self.reduced_predicate is not None:
            columns.append(self.reduced_predicate)

        if ordering_mode in ("offset_col", "ordered_col"):
            # Generate offsets if current ordering id semantics are not sufficiently strict
            if (ordering_mode == "offset_col" and not self._ordering.is_sequential) or (
                ordering_mode == "ordered_col" and not self._ordering.order_id_defined
            ):
                window = ibis.window(order_by=self.ordering)
                if self._predicates:
                    window = window.group_by(self.reduced_predicate)
                columns.append(ibis.row_number().name(order_col_name).over(window))
            elif self._ordering.ordering_id:
                columns.append(
                    self._get_meta_column(self._ordering.ordering_id).name(
                        order_col_name
                    )
                )
            else:
                # Should not be possible.
                raise ValueError(
                    "Expression does not have ordering id and none was generated."
                )
        elif ordering_mode in ["order_by", "expose_metadata"]:
            columns.extend(
                [self._get_meta_column(name) for name in hidden_ordering_columns]
            )

        # Special case for empty tables, since we can't create an empty
        # projection.
        if not columns:
            return ibis.memtable([])

        table = table.select(columns)

        if self.reduced_predicate is not None:
            table = table.filter(table[PREDICATE_COLUMN])
            # Drop predicate as it is will be all TRUE after filtering
            table = table.drop(PREDICATE_COLUMN)

        if ordering_mode == "order_by":
            is_ascending = self._ordering.is_ascending
            # Some ordering columns are value columns, while other are used purely for ordering.
            # We drop the non-value columns after the ordering
            table = table.order_by(
                [
                    table[col_id] if is_ascending else ibis.desc(table[col_id])
                    for col_id in [*self._ordering.all_ordering_columns]
                ]
            )
            if not (ordering_mode == "expose_metadata"):
                table = table.drop(*hidden_ordering_columns)

        return table

    def start_query(
        self, job_config: Optional[bigquery.job.QueryJobConfig] = None
    ) -> bigquery.QueryJob:
        """Execute a query and return metadata about the results."""
        # TODO(swast): Cache the job ID so we can look it up again if they ask
        # for the results? We'd need a way to invalidate the cache if DataFrame
        # becomes mutable, though. Or move this method to the immutable
        # expression class.
        # TODO(swast): We might want to move this method to Session and/or
        # provide our own minimal metadata class. Tight coupling to the
        # BigQuery client library isn't ideal, especially if we want to support
        # a LocalSession for unit testing.
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        table = self.to_ibis_expr()
        sql = table.compile()
        if job_config is not None:
            return self._session.bqclient.query(sql, job_config=job_config)
        else:
            return self._session.bqclient.query(sql)

    def _reproject_to_table(self):
        """
        Internal operators that projects the internal representation into a
        new ibis table expression where each value column is a direct
        reference to a column in that table expression. Needed after
        some operations such as window operations that cannot be used
        recursively in projections.
        """
        table = self.to_ibis_expr(
            ordering_mode="expose_metadata", order_col_name=self._ordering.ordering_id
        )
        columns = [table[column_name] for column_name in self._column_names]
        meta_columns = [table[column_name] for column_name in self._meta_column_names]
        return BigFramesExpr(
            self._session,
            table,
            columns=columns,
            meta_columns=meta_columns,
            ordering=self._ordering,
        )

    def _ibis_window_from_spec(self, window_spec: WindowSpec):
        group_by: typing.List[ibis_types.Value] = (
            [
                typing.cast(ibis_types.Column, self.get_column(column))
                for column in window_spec.grouping_keys
            ]
            if window_spec.grouping_keys
            else []
        )
        if self.reduced_predicate is not None:
            group_by.append(self.reduced_predicate)

        if window_spec.ordering:
            order_overrides = [
                ibis.asc(typing.cast(ibis_types.Column, self.get_column(column)))
                if direction == OrderingDirection.ASC
                else ibis.desc(typing.cast(ibis_types.Column, self.get_column(column)))
                for column, direction in window_spec.ordering
            ]
            order_by = tuple([*order_overrides, *self.ordering])
        elif (window_spec.following is not None) or (window_spec.preceding is not None):
            # If window spec has following or preceding bounds, we need to apply an unambiguous ordering.
            order_by = tuple(self.ordering)
        else:
            # Unbound grouping window. Suitable for aggregations but not for analytic function application.
            order_by = None

        return ibis.window(
            preceding=window_spec.preceding,
            following=window_spec.following,
            order_by=order_by,
            group_by=group_by,
        )

    def _set_or_replace_by_id(self, id: str, value: ibis_types.Value):
        expr = self.builder()
        if id in self.column_names:
            expr.columns = list(
                [
                    value.name(id) if id == name else self.get_column(name)
                    for name in self.column_names
                ]
            )
        else:
            expr.columns = [*self.columns, value.name(id)]
        return expr.build()


class BigFramesExprBuilder:
    """Mutable expression class.
    Use BigFramesExpr.builder() to create from a BigFramesExpr object.
    """

    def __init__(
        self,
        session: Session,
        table: ibis_types.Table,
        columns: Collection[ibis_types.Value] = (),
        meta_columns: Collection[ibis_types.Value] = (),
        ordering: Optional[ExpressionOrdering] = None,
        predicates: Optional[Collection[ibis_types.BooleanValue]] = None,
    ):
        self.session = session
        self.table = table
        self.columns = list(columns)
        self.meta_columns = list(meta_columns)
        self.ordering = ordering
        self.predicates = list(predicates) if predicates is not None else None

    def build(self) -> BigFramesExpr:
        return BigFramesExpr(
            session=self.session,
            table=self.table,
            columns=self.columns,
            meta_columns=self.meta_columns,
            ordering=self.ordering,
            predicates=self.predicates,
        )


class BigFramesGroupByExpr:
    """Represents a grouping on a table. Does not currently support projection, filtering or sorting."""

    def __init__(self, expr: BigFramesExpr, by: typing.Any):
        self._session = expr._session
        self._expr = expr
        self._by = by

    def _to_ibis_expr(self):
        """Creates an Ibis table expression representing the DataFrame."""
        return self._expr.to_ibis_expr(ordering_mode="unordered")

    def aggregate(
        self, column_name: str, aggregate_op: agg_ops.AggregateOp
    ) -> BigFramesExpr:
        """Generate aggregate metrics, result preserve names of aggregated columns"""
        # TODO(tbergeron): generalize to multiple aggregations
        table = self._to_ibis_expr()
        result = table.group_by(self._by).aggregate(
            aggregate_op._as_ibis(table[column_name]).name(column_name)
        )
        return BigFramesExpr(self._session, result)


def _reduce_predicate_list(
    predicate_list: typing.Collection[ibis_types.BooleanValue],
) -> ibis_types.BooleanValue:
    """Converts a list of predicates BooleanValues into a single BooleanValue."""
    if len(predicate_list) == 0:
        raise ValueError("Cannot reduce empty list of predicates")
    if len(predicate_list) == 1:
        (item,) = predicate_list
        return item
    return functools.reduce(lambda acc, pred: acc.__and__(pred), predicate_list)
