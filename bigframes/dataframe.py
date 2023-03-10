"""DataFrame is a two dimensional data structure."""

from __future__ import annotations

import typing
from typing import Iterable, Mapping, Optional, Union

import ibis
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types
import pandas

import bigframes.core
import bigframes.core.blocks as blocks
import bigframes.core.indexes.implicitjoiner
import bigframes.core.indexes.index
import bigframes.dtypes
import bigframes.series


class DataFrame:
    """A 2D data structure, representing data and deferred computation.

    .. warning::
        This constructor is **private**. Use a public method such as
        ``Session.read_gbq`` to construct a DataFrame.
    """

    def __init__(
        self,
        block: blocks.Block,
    ):
        self._block = block

    @property
    def index(
        self,
    ) -> Union[
        bigframes.core.indexes.implicitjoiner.ImplicitJoiner,
        bigframes.core.indexes.index.Index,
    ]:
        return self._block.index

    @property
    def dtypes(self) -> pandas.Series:
        """Returns the dtypes as a Pandas Series object"""
        schema_elements = self._block.expr.to_ibis_expr().schema().items()
        column_names, ibis_dtypes = zip(*schema_elements)
        bigframes_dtypes = [
            bigframes.dtypes.ibis_dtype_to_bigframes_dtype(ibis_dtype)
            for ibis_dtype in ibis_dtypes
        ]
        return pandas.Series(data=bigframes_dtypes, index=column_names)

    @property
    def sql(self) -> str:
        """Compiles this dataframe's expression tree to SQL"""
        return self._block.expr.to_ibis_expr().compile()

    def __getitem__(
        self, key: Union[str, Iterable[str], bigframes.series.Series]
    ) -> Union[bigframes.series.Series, "DataFrame"]:
        """Gets the specified column(s) from the DataFrame."""
        # NOTE: This implements the operations described in
        # https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html

        if isinstance(key, str):
            # Check that the column exists.
            # TODO(swast): Make sure we can't select Index columns this way.
            _ = self._block.expr.get_column(key)
            # Return a view so that mutations on the Series can affect this DataFrame.
            # TODO(swast): Copy if "copy-on-write" semantics are enabled.
            return bigframes.series.Series(self._block, key)

        if isinstance(key, bigframes.series.Series):
            if key._to_ibis_expr().type() == ibis_dtypes.bool:
                # TODO: enforce stricter alignment
                combined_index, (
                    _,
                    get_column_right,
                ) = self._block.index.join(key.index, how="left")
                right = get_column_right(key._value_column)

                block = self._block.copy()
                block.index = combined_index
                filtered_expr = block.expr.filter((right == ibis.literal(True)))
                block.expr = filtered_expr
                return DataFrame(block)
            else:
                raise ValueError(
                    "Only boolean series currently supported for indexing."
                )

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

        # TODO(swast): Should we disallow selecting the index column like
        # any other column?
        block = self._block.copy()
        block.replace_value_columns(
            [self._block.expr.get_column(column_name) for column_name in key]
        )
        return DataFrame(block)

    def __getattr__(self, key: str):
        if key not in self._block.expr.column_names:
            raise AttributeError(key)
        return self.__getitem__(key)

    def __repr__(self) -> str:
        """Converts a DataFrame to a string."""
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        job = self._block.expr.start_query().result(max_results=10)
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
        return self._block.compute()

    def head(self, n: int = 5) -> DataFrame:
        """Limits DataFrame to a specific number of rows."""
        block = self._block.copy()
        block.expr = self._block.expr.apply_limit(n)
        return DataFrame(block)

    def drop(self, columns: Union[str, Iterable[str]]) -> DataFrame:
        """Drop specified column(s)."""
        if isinstance(columns, str):
            columns = [columns]

        # TODO(swast): Validate that we aren't trying to drop the index column.
        block = self._block.copy()
        block.expr = self._block.expr.drop_columns(columns)
        return DataFrame(block)

    def rename(self, columns: Mapping[str, str]) -> DataFrame:
        """Alter column labels."""
        # TODO(garrettwu) Support function(Callable) as columns parameter.
        expr_builder = self._block.expr.builder()

        for i, col in enumerate(expr_builder.columns):
            if col.get_name() in columns:
                expr_builder.columns[i] = col.name(columns[col.get_name()])

        block = self._block.copy()
        block.expr = expr_builder.build()
        return DataFrame(block)

    def assign(self, **kwargs) -> DataFrame:
        """Assign new columns to a DataFrame.

        Returns a new object with all original columns in addition to new ones. Existing columns that are re-assigned will be overwritten.
        """
        # TODO(garrettwu) Support list-like values. Requires ordering.
        # TODO(garrettwu) Support callable values.

        expr_builder = self._block.expr.builder()
        existing_col_pos_map = {
            col.get_name(): i for i, col in enumerate(expr_builder.columns)
        }
        for k, v in kwargs.items():
            if type(v) == bigframes.series.Series:
                v = v._value
            expr_builder.table = expr_builder.table.mutate(**{k: v})
            if k in existing_col_pos_map:
                expr_builder.columns[existing_col_pos_map[k]] = expr_builder.table[k]
            else:
                expr_builder.columns.append(expr_builder.table[k])

        block = self._block.copy()
        block.expr = expr_builder.build()
        return DataFrame(block)

    def reset_index(self, *, drop: bool = False) -> DataFrame:
        """Reset the index of the DataFrame, and use the default one instead."""
        original_index_columns = self._block.index_columns
        block = self._block.copy()
        # TODO(swast): Only remove a specified number of levels from a
        # MultiIndex.
        block.index_columns = ()

        if drop:
            # Even though the index might be part of the ordering, keep that
            # ordering expression as reset_index shouldn't change the row
            # order.
            block.expr = block.expr.drop_columns(original_index_columns)

        return DataFrame(block)

    def set_index(self, key: str) -> DataFrame:
        """Set the DataFrame index using existing columns."""
        expr = self._block.expr
        prev_index_columns = self._block.index_columns
        index_expr = typing.cast(ibis_types.Column, expr.get_column(key))

        if not expr.ordering:
            expr = expr.order_by([ibis.asc(index_expr)])

        block = self._block.copy()
        block.index = bigframes.core.indexes.index.Index(expr, key)
        block.expr = block.expr.drop_columns(prev_index_columns)
        return DataFrame(block)

    def dropna(self) -> DataFrame:
        """Remove rows with missing values."""
        # Can't dropna on original table expr, it will raise "not same table expression" on selection.
        table = self._block.expr.to_ibis_expr()
        table = table.dropna()

        expr = bigframes.core.BigFramesExpr(self._block.expr._session, table)
        block = self._block.copy()
        block.expr = expr
        return DataFrame(block)

    def merge(
        self,
        right: DataFrame,
        how: str = "inner",  # TODO(garrettwu): Currently can take inner, outer, left and right. To support cross joins
        # TODO(garrettwu): Support "on" list of columns and None. Currently a single column must be provided
        on: Optional[str] = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> DataFrame:
        """Merge DataFrame objects with a database-style join."""
        if not on:
            raise ValueError("Must specify a column to join on.")

        # Drop index column in joins. Consistent with Pandas.
        left = self.reset_index(drop=True)
        right = right.reset_index(drop=True)

        left_table = left._block.expr.to_ibis_expr()
        right_table = right._block.expr.to_ibis_expr()

        joined_table = left_table.join(
            right_table, left_table[on] == right_table[on], how, suffixes=suffixes
        )
        block = blocks.Block(
            bigframes.core.BigFramesExpr(left._block.expr._session, joined_table)
        )
        joined_frame = DataFrame(block)

        # Ibis emits redundant columns for outer joins. See:
        # https://ibis-project.org/ibis-for-pandas-users/#merging-tables
        left_on_name = on + suffixes[0]
        right_on_name = on + suffixes[1]
        if (
            left_on_name in joined_frame._block.expr.column_names
            and right_on_name in joined_frame._block.expr.column_names
        ):
            joined_frame = joined_frame.drop(right_on_name)
            joined_frame = joined_frame.rename({left_on_name: on})

        return joined_frame
