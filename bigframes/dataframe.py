"""DataFrame is a two dimensional data structure."""

from typing import Iterable, Optional, Union

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

    def compute(self) -> pandas.DataFrame:
        """Executes deferred operations and downloads the results."""
        job = self._expr.start_query()
        return job.result().to_dataframe()

    def head(self, max_results: Optional[int] = 5) -> pandas.DataFrame:
        """Limits DataFrame to a specific number of rows."""
        # TOOD(swast): This will be deferred once more opportunistic style
        # execution is implemented.
        job = self._expr.start_query()
        # TODO(swast): Type annotations to be corrected here:
        # https://github.com/googleapis/python-bigquery/pull/1487
        rows = job.result(max_results=max_results)  # type: ignore
        return rows.to_dataframe()
