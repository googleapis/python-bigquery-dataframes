from __future__ import annotations

import typing
from typing import Collection, Iterable, Optional, Sequence

from google.cloud import bigquery
import ibis
import ibis.expr.types as ibis_types
from ibis.expr.types.groupby import GroupedTable

if typing.TYPE_CHECKING:
    from bigframes.session import Session
ORDER_ID_COLUMN = "bigframes_ordering_id"


class ExpressionOrdering:
    """Immutable object that holds information about the ordering of rows in a BigFrames expression."""

    def __init__(
        self,
        ordering_value_columns: Optional[Sequence[ibis_types.Value]] = None,
        ordering_id_column: Optional[ibis_types.Value] = None,
        is_sequential: bool = False,
    ):
        # TODO(tbergeron): Allow flag to reverse ordering
        self._ordering_value_columns = (
            tuple(ordering_value_columns) if ordering_value_columns else ()
        )
        self._ordering_id_column = ordering_id_column
        self._is_sequential = is_sequential

    def with_is_sequential(self, is_sequential: bool):
        """Create a copy that is marked as non-sequential, this is useful when filtering, but not sorting, an expression."""
        return ExpressionOrdering(
            self._ordering_value_columns, self._ordering_id_column, is_sequential
        )

    @property
    def all_ordering_columns(self):
        return (
            list(self._ordering_value_columns)
            if self._ordering_id_column is None
            else [*self._ordering_value_columns, self._ordering_id_column]
        )

    @property
    def is_sequential(self) -> bool:
        return self._is_sequential

    @property
    def ordering_id(self) -> Optional[ibis_types.Value]:
        return self._ordering_id_column


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
        predicates: A list of filters on the data frame.
    """

    def __init__(
        self,
        session: Session,
        table: ibis_types.Table,
        columns: Optional[Sequence[ibis_types.Value]] = None,
        ordering: Optional[ExpressionOrdering] = None,
        predicates: Optional[Collection[ibis_types.BooleanValue]] = None,
    ):
        self._session = session
        self._table = table
        self._predicates = tuple(predicates) if predicates is not None else ()
        self._ordering = ordering
        # Allow creating a DataFrame directly from an Ibis table expression.
        if columns is None:
            self._columns = tuple(table[key] for key in table.columns)
        else:
            # TODO(swast): Validate that each column references the same table (or
            # no table for literal values).
            self._columns = tuple(columns)
        # To allow for more efficient lookup by column name, create a
        # dictionary mapping names to column values.
        self._column_names = {column.get_name(): column for column in self._columns}

    @property
    def table(self) -> ibis_types.Table:
        return self._table

    @property
    def predicates(self) -> typing.Tuple[ibis_types.BooleanValue, ...]:
        return self._predicates

    @property
    def column_names(self) -> dict[str, ibis_types.Value]:
        return self._column_names

    @property
    def ordering(self) -> Sequence[ibis_types.Value]:
        return self._ordering.all_ordering_columns if self._ordering else ()

    def builder(self) -> BigFramesExprBuilder:
        """Creates a mutable builder for expressions."""
        # Since BigFramesExpr is intended to be immutable (immutability offers
        # potential opportunities for caching, though we might need to introduce
        # more node types for that to be useful), we create a builder class.
        return BigFramesExprBuilder(
            self._session,
            self._table,
            self._columns,
            ordering=self._ordering,
            predicates=self._predicates,
        )

    def drop_columns(self, columns: Iterable[str]) -> BigFramesExpr:
        expr = self.builder()
        remain_cols = [
            column for column in expr.columns if column.get_name() not in columns
        ]
        expr.columns = remain_cols
        return expr.build()

    def get_column(self, key: str) -> ibis_types.Value:
        """Gets the Ibis expression for a given column."""
        return self._column_names[key]

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

    def order_by(self, by: Sequence[ibis_types.Value]) -> BigFramesExpr:
        # TODO(tbergeron): Always append fully ordered OID to end to guarantee total ordering.
        expr = self.builder()
        expr.ordering = ExpressionOrdering(
            ordering_value_columns=by,
        )
        return expr.build()

    @property
    def offsets(self):
        if not self._ordering.is_sequential:
            raise ValueError(
                "Expression does not have offsets. Generate them first using project_offsets."
            )
        return self._ordering.ordering_id

    def project_offsets(self) -> BigFramesExpr:
        """Create a new expression that contains offsets. Should only be executed when offsets are needed for an operations. Has no effect on expression semantics."""
        table = self._table
        # TODO(tbergeron): Enforce total ordering
        # TODO(tbergeron): Preserve original ordering id or row id in order to facilitate mutation.
        if not self._ordering:
            raise ValueError(
                "Offsets cannot be generated. Ordering not defined for this table."
            )
        if self._ordering.is_sequential:
            return self  # No-op as ordering column already has offsets
        table = table.order_by(list(self._ordering.all_ordering_columns))
        if len(self._predicates) > 0:
            table = table.filter(list(self._predicates))
        table = table.select(
            [
                *self._columns,
                ibis.row_number().name(ORDER_ID_COLUMN),
            ]
        )
        expr = self.builder()
        expr.table = table
        # Exclude ordering id in list of columns as it is not a value column.
        expr.columns = [table[column_name] for column_name in self._column_names]
        expr.ordering = ExpressionOrdering(
            ordering_id_column=table[ORDER_ID_COLUMN], is_sequential=True
        )
        return expr.build()

    def projection(self, columns: Iterable[ibis_types.Value]) -> BigFramesExpr:
        """Creates a new expression based on this expression with new columns."""
        # TODO(swast): We might want to do validation here that columns derive
        # from the same table expression instead of (in addition to?) at
        # construction time.
        expr = self.builder()
        expr.columns = list(columns)
        return expr.build()

    def to_ibis_expr(self):
        """Creates an Ibis table expression representing the DataFrame."""
        table = self._table
        if self._ordering:
            table = table.order_by(list(self._ordering.all_ordering_columns))
        if len(self._predicates) > 0:
            table = table.filter(list(self._predicates))
        if self._columns is not None:
            table = table.select(list(self._columns))
        return table

    def start_query(self) -> bigquery.QueryJob:
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
        return self._session.bqclient.query(sql)


class BigFramesExprBuilder:
    """Mutable expression class.
    Use BigFramesExpr.builder() to create from a BigFramesExpr object.
    """

    def __init__(
        self,
        session: Session,
        table: ibis_types.Table,
        columns: Collection[ibis_types.Value] = (),
        ordering: Optional[ExpressionOrdering] = None,
        predicates: Optional[Collection[ibis_types.BooleanValue]] = None,
    ):
        self.session = session
        self.table = table
        self.columns = list(columns)
        self.ordering = ordering
        self.predicates = list(predicates) if predicates is not None else None

    def build(self) -> BigFramesExpr:
        return BigFramesExpr(
            session=self.session,
            table=self.table,
            columns=self.columns,
            ordering=self.ordering,
            predicates=self.predicates,
        )


class BigFramesGroupByExpr:
    """Represents a grouping on a table. Does not currently support projection, filtering or sorting."""

    def __init__(self, expr: BigFramesExpr, by: typing.Any):
        self._session = expr._session
        self._expr = expr
        self._by = by

    def _to_ibis_expr(self) -> GroupedTable:
        """Creates an Ibis table expression representing the DataFrame."""
        return self._expr.to_ibis_expr().group_by(self._by)

    def aggregate(self, metrics: Collection[ibis_types.Scalar]) -> BigFramesExpr:
        table = self._to_ibis_expr().aggregate(metrics)
        # Since we make a new table expression, the old column references now
        # point to the wrong table. Use the BigFramesExpr constructor to make
        # sure we have the correct references.
        return BigFramesExpr(self._session, table)
