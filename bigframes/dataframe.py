"""DataFrame is a two dimensional data structure."""

from typing import Collection, Iterable, Optional, Union

from ibis.expr.types import Column, Table
import pandas

import bigframes.series


class DataFrame:
    """A 2D data structure, representing data and deferred computation.

    .. warning::
        This constructor is **private**. Use a public method such as
        ``Engine.read_gbq`` to construct a DataFrame.
    """

    def __init__(
        self,
        table: Table,
        columns: Optional[Collection[Column]] = None,
    ):
        super().__init__()

        # Allow creating a DataFrame directly from an Ibis table expression.
        if columns is None:
            columns = [table.get_column(key) for key in table.columns]

        self._table = table
        # TODO(swast): Validate that each column references the same table.
        self._columns = columns
        self._column_names = {column.get_name(): column for column in columns}

    def __getitem__(self, key: Union[str, Iterable[str]]) -> bigframes.series.Series:
        """Gets the specified column(s) from the DataFrame."""
        # NOTE: This implements the operations described in
        # https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html

        if isinstance(key, str):
            column = self._column_names[key]
            return bigframes.series.Series(self._table, column)

        # TODO(swast): Allow for filtering of rows by a boolean Series, returning a
        # filtered version of this DataFrame instead of a Series.

        # Select a subset of columns or re-order columns.
        # In Ibis after you apply a projection, any column objects from the
        # table before the projection can't be combined with column objects
        # from the table after the projection. This is because the table after
        # a projection is considered a totally separate table expression.
        #
        # This is unexpected behavior for a pandas user, who expects their old
        # Series objects to still work with the new / mutated DataFrame. We
        # avoid applying a projection in Ibis until it's absolutely necessary
        # to provide pandas-like semantics.
        return DataFrame(
            self._table,
            [self._column_names[column_name] for column_name in key],
        )

    def _to_ibis_expr(self):
        table = self._table
        if self._columns is not None:
            table = self._table.select(self._columns)
        return table

    def compute(self) -> pandas.DataFrame:
        """Executes deferred operations and downloads the results."""
        table = self._to_ibis_expr()
        return table.execute()

    def head(self, max_results: Optional[int] = 5) -> pandas.DataFrame:
        """Executes deferred operations and downloads a specific number of rows."""
        table = self._to_ibis_expr()
        return table.execute(limit=max_results)
