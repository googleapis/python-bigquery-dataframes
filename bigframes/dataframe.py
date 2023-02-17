"""DataFrame is a two dimensional data structure."""

from __future__ import annotations

import typing
from typing import Iterable, Mapping, Optional, Union

import ibis
import ibis.expr.types as ibis_types
import pandas

import bigframes.core
import bigframes.core.indexes.implicitjoiner
import bigframes.core.indexes.index
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
        *,
        index: Optional[bigframes.core.indexes.implicitjoiner.ImplicitJoiner] = None,
    ):
        if index is None:
            index = bigframes.core.indexes.implicitjoiner.ImplicitJoiner(expr)

        # TODO(swast): To support mutable cells (such as with inplace=True),
        # we might want to store columns as a collection of Series instead.
        self._expr = expr
        self._index = index

    @property
    def index(self) -> bigframes.core.indexes.implicitjoiner.ImplicitJoiner:
        return self._index

    def __getitem__(
        self, key: Union[str, Iterable[str]]
    ) -> Union[bigframes.series.Series, "DataFrame"]:
        """Gets the specified column(s) from the DataFrame."""
        # NOTE: This implements the operations described in
        # https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html

        if isinstance(key, str):
            column = self._expr.get_column(key)
            return bigframes.series.Series(self._expr, column, index=self._index)

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
        # TODO(swast): Do we need to apply an implicit join when doing a
        # projection?

        # TODO(swast): Probably need to include index column(s) here, too.
        projection_keys = list(key)

        if isinstance(self._index, bigframes.core.indexes.index.Index):
            index_name = self._index._index_column
            # TODO(swast): Should we disallow selecting the index column like
            # any other column?
            if index_name not in projection_keys:
                projection_keys.append(index_name)

        expr = self._expr.projection(
            [self._expr.get_column(column_name) for column_name in projection_keys]
        )
        index = self._index.copy()
        index._expr = expr
        return DataFrame(expr, index=index)

    def __getattr__(self, key: str):
        if key not in self._expr.column_names.keys():
            raise AttributeError(key)
        return self.__getitem__(key)

    def __repr__(self) -> str:
        """Converts a DataFrame to a string."""
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        job = self._expr.start_query().result(max_results=10)
        rows = job.total_rows
        columns = len(job.schema)
        # TODO(swast): Need to set index if we have an index column(s)
        preview = job.to_dataframe()

        # TODO(swast): Print the SQL too?
        # Grab all but the final two lines so we can replace the row count with
        # the actual row count from the query.
        lines = repr(preview).split("\n")[:-2]
        if rows > len(preview.index):
            lines.append("...")

        lines.append("")
        lines.append(f"[{rows} rows x {columns} columns]")
        return "\n".join(lines)

    def compute(self) -> pandas.DataFrame:
        """Executes deferred operations and downloads the results."""
        # TODO(swast): Use Ibis for now for consistency with Series, but
        # ideally we'd do our own thing via the BQ client library where we can
        # more easily override the output dtypes to use nullable dtypes and
        # avoid lossy conversions.
        df = self._expr.to_ibis_expr().execute()
        # TODO(swast): We need something that encapsulates data + index
        # column(s) so that Series and DataFrame don't need to duplicate
        # Index logic.
        if isinstance(self._index, bigframes.core.indexes.index.Index):
            df = df.set_index(self._index._index_column)
            df.index.name = self._index.name
        return df

    def head(self, n: int = 5) -> DataFrame:
        """Limits DataFrame to a specific number of rows."""
        expr = self._expr.apply_limit(n)
        index = self._index.copy()
        index._expr = expr
        return DataFrame(expr, index=index)

    def drop(self, columns: Union[str, Iterable[str]]) -> DataFrame:
        """Drop specified column(s)."""
        if isinstance(columns, str):
            columns = [columns]

        # TODO(swast): Validate that we aren't trying to drop the index column.
        expr_builder = self._expr.builder()
        remain_cols = [
            column
            for column in expr_builder.columns
            if column.get_name() not in columns
        ]
        expr_builder.columns = remain_cols
        expr = expr_builder.build()
        index = self._index.copy()
        index._expr = expr
        return DataFrame(expr, index=index)

    def rename(self, columns: Mapping[str, str]) -> DataFrame:
        """Alter column labels."""
        # TODO(garrettwu) Support function(Callable) as columns parameter.
        expr_builder = self._expr.builder()

        for i, col in enumerate(expr_builder.columns):
            if col.get_name() in columns.keys():
                expr_builder.columns[i] = col.name(columns[col.get_name()])

        expr = expr_builder.build()
        index = self._index.copy()
        index._expr = expr
        return DataFrame(expr, index=index)

    def assign(self, **kwargs) -> DataFrame:
        """Assign new columns to a DataFrame.

        Returns a new object with all original columns in addition to new ones. Existing columns that are re-assigned will be overwritten.
        """
        # TODO(garrettwu) Support list-like values. Requires ordering.
        # TODO(garrettwu) Support callable values.

        expr_builder = self._expr.builder()
        existing_col_pos_map = {
            col.get_name(): i for i, col in enumerate(expr_builder.columns)
        }
        for k, v in kwargs.items():
            expr_builder.table = expr_builder.table.mutate(**{k: v})
            if k in existing_col_pos_map.keys():
                expr_builder.columns[existing_col_pos_map[k]] = expr_builder.table[k]
            else:
                expr_builder.columns.append(expr_builder.table[k])

        expr = expr_builder.build()
        index = self._index.copy()
        index._expr = expr
        return DataFrame(expr, index=index)

    def set_index(self, key: str) -> DataFrame:
        expr = self._expr
        index_expr = typing.cast(ibis_types.Column, expr.get_column(key))

        if not self._expr.ordering:
            expr = expr.order_by([ibis.asc(index_expr)])

        # TODO(swast): Somehow need to separate index column from data columns.
        index = bigframes.core.indexes.index.Index(expr, key)
        return DataFrame(expr, index=index)
