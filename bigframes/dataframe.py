"""DataFrame is a two dimensional data structure."""

from __future__ import annotations

from typing import Iterable, Union

import pandas

import bigframes.core
import bigframes.series


class DataFrame:
    """A 2D data structure, representing data and deferred computation.

    .. warning::
        This constructor is **private**. Use a public method such as
        ``Session.read_gbq`` to construct a DataFrame.
    """

    def __init__(
        self,
        expr: bigframes.core.BigFramesExpr,
    ):
        # TODO(swast): To support mutable cells (such as with inplace=True),
        # we might want to store columns as a collection of Series instead.
        self._expr = expr

    def __getitem__(
        self, key: Union[str, Iterable[str]]
    ) -> Union[bigframes.series.Series, "DataFrame"]:
        """Gets the specified column(s) from the DataFrame."""
        # NOTE: This implements the operations described in
        # https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html

        if isinstance(key, str):
            column = self._expr.get_column(key)
            return bigframes.series.Series(self._expr, column)

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
        expr = self._expr.projection(
            [self._expr.get_column(column_name) for column_name in key]
        )
        return DataFrame(expr)

    def __repr__(self) -> str:
        """Converts a DataFrame to a string."""
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        job = self._expr.start_query().result(max_results=10)
        rows = job.total_rows
        columns = len(job.schema)
        preview = job.to_dataframe()

        # TODO(swast): Print the SQL too?
        lines = [repr(preview)]
        if rows > len(preview.index):
            lines.append("...")

        lines.append("")
        lines.append(f"[{rows} rows x {columns} columns]")
        return "\n".join(lines)

    def compute(self) -> pandas.DataFrame:
        """Executes deferred operations and downloads the results."""
        job = self._expr.start_query()
        return job.result().to_dataframe()

    def head(self, n: int = 5) -> DataFrame:
        """Limits DataFrame to a specific number of rows."""
        expr = self._expr.limit(n)
        return DataFrame(expr)
