from __future__ import annotations

import typing
from typing import Collection, Optional

from google.cloud import bigquery
from ibis.expr.types import Table, Value

if typing.TYPE_CHECKING:
    from bigframes.session import Session


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
    """

    def __init__(
        self,
        session: Session,
        table: Table,
        columns: Optional[Collection[Value]] = None,
        limit: Optional[int] = None,
    ):
        self._session = session
        self._table = table
        self._limit = limit

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

    def builder(self) -> BigFramesExprBuilder:
        """Creates a mutable builder for expressions."""
        # Since BigFramesExpr is intended to be immutable (immutability offers
        # potential opportunities for caching, though we might need to introduce
        # more node types for that to be useful), we create a builder class.
        return BigFramesExprBuilder(
            self._session, self._table, self._columns, self._limit
        )

    def get_column(self, key: str) -> Value:
        """Gets the Ibis expression for a given column."""
        return self._column_names[key]

    def limit(self, max_results: int) -> BigFramesExpr:
        expr = self.builder()
        expr.limit = max_results if expr.limit is None else min(expr.limit, max_results)
        return expr.build()

    def projection(self, columns: Collection[Value]) -> BigFramesExpr:
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
        if self._columns is not None:
            table = self._table.select(self._columns)
        if self._limit is not None:
            table = table.limit(self._limit)
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
    """Mutable expression class."""

    def __init__(
        self,
        session: Session,
        table: Table,
        columns: Optional[Collection[Value]] = None,
        limit: Optional[int] = None,
    ):
        self.session = session
        self.table = table
        self.columns = list(columns) if columns is not None else None
        self.limit = limit

    def build(self) -> BigFramesExpr:
        return BigFramesExpr(self.session, self.table, self.columns, self.limit)
