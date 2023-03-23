"""DataFrame is a two dimensional data structure."""

from __future__ import annotations

import operator
import re
import typing
from typing import Iterable, Mapping, Optional, Union

import google.cloud.bigquery as bigquery
import ibis
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types
import pandas as pd

import bigframes.core
import bigframes.core.blocks as blocks
import bigframes.core.indexes.implicitjoiner
import bigframes.core.indexes.index
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.series

_INDEX_COLUMN_NAME = "bigframes_index_{}"


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
    def dtypes(self) -> pd.Series:
        """Returns the dtypes as a Pandas Series object"""
        schema_elements = [
            el
            for el in self._block.expr.to_ibis_expr().schema().items()
            if el[0] not in self._block.index_columns
        ]
        column_names, ibis_dtypes = zip(*schema_elements)
        bigframes_dtypes = [
            bigframes.dtypes.ibis_dtype_to_bigframes_dtype(ibis_dtype)
            for ibis_dtype in ibis_dtypes
        ]
        return pd.Series(data=bigframes_dtypes, index=column_names)

    @property
    def columns(self) -> pd.Index:
        """Returns the column labels of the dataframe"""
        return self.dtypes.index

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
            # Copy to emulate "Copy-on-Write" semantics.
            block = self._block.copy()
            # Since we're working with a "copy", we can drop the other value
            # columns. They aren't needed on the Series.
            block.expr = block.expr.drop_columns(
                [column for column in self.columns if column != key]
            )
            return bigframes.series.Series(block, key, name=key)

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
        # Grab all but the final 2 lines if those are the shape of the DF. So we can replace the row count with
        # the actual row count from the query.
        lines = repr(preview).split("\n")
        pattern = re.compile("\\[[0-9]+ rows x [0-9]+ columns\\]")
        if pattern.match(lines[-1]):
            lines = lines[:-2]

        if rows > len(preview.index):
            lines.append("...")

        lines.append("")
        lines.append(f"[{rows} rows x {columns} columns]")
        return "\n".join(lines)

    def _apply_scalar_bi_op(
        self, other: float | int, op, reverse: bool = False
    ) -> DataFrame:
        scalar = bigframes.dtypes.literal_to_ibis_scalar(other)
        block = self._block.copy()
        expr_builder = block.expr.builder()
        for i, column in enumerate(expr_builder.columns):
            if not column.get_name() in block.index_columns:
                value = op(scalar, column) if reverse else op(column, scalar)
                expr_builder.columns[i] = value.name(column.get_name())
        block.expr = expr_builder.build()
        return DataFrame(block)

    def add(self, other: float | int) -> DataFrame:
        return self._apply_scalar_bi_op(other, operator.add)

    __radd__ = __add__ = radd = add

    def sub(self, other: float | int) -> DataFrame:
        return self._apply_scalar_bi_op(other, operator.sub)

    __sub__ = sub

    def rsub(self, other: float | int) -> DataFrame:
        return self._apply_scalar_bi_op(other, operator.sub, reverse=True)

    __rsub__ = rsub

    def mul(self, other: float | int) -> DataFrame:
        return self._apply_scalar_bi_op(other, operator.mul)

    __rmul__ = __mul__ = rmul = mul

    def truediv(self, other: float | int) -> DataFrame:
        return self._apply_scalar_bi_op(other, operator.truediv)

    div = __truediv__ = truediv

    def rtruediv(self, other: float | int) -> DataFrame:
        return self._apply_scalar_bi_op(other, operator.truediv, reverse=True)

    __rtruediv__ = rtruediv

    def floordiv(self, other: float | int) -> DataFrame:
        return self._apply_scalar_bi_op(other, operator.floordiv)

    __floordiv__ = floordiv

    def rfloordiv(self, other: float | int) -> DataFrame:
        return self._apply_scalar_bi_op(other, operator.floordiv, reverse=True)

    __rfloordiv__ = rfloordiv

    def compute(self) -> pd.DataFrame:
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

        cur = self
        for k, v in kwargs.items():
            cur = cur._assign_single_item(k, v)

        return cur

    def _assign_single_item(self, k, v) -> DataFrame:
        if isinstance(v, bigframes.series.Series):
            return self._assign_series_join_on_index(k, v)
        else:
            return self._assign_scalar(k, v)

    def _assign_scalar(self, k, v) -> DataFrame:
        v = bigframes.dtypes.literal_to_ibis_scalar(v)
        block = self._block.copy()
        expr_builder = block.expr.builder()
        existing_col_pos_map = {
            col.get_name(): i for i, col in enumerate(expr_builder.columns)
        }

        v = v.name(k)

        if k in existing_col_pos_map:
            expr_builder.columns[existing_col_pos_map[k]] = v
        else:
            expr_builder.columns.append(v)

        block.expr = expr_builder.build()
        return DataFrame(block)

    def _assign_series_join_on_index(self, k, v: bigframes.series.Series) -> DataFrame:
        joined_index, (get_column_left, get_column_right) = self.index.join(
            v.index, how="left"
        )
        expr_builder = joined_index._expr.builder()

        # Restore original column names
        columns = []
        for column in self._block.expr.columns:
            column_name = column.get_name()
            # If it is a replacement, then replace with column from right
            if column_name == k:
                columns.append(get_column_right(v._value.get_name()).name(column_name))
            else:
                columns.append(get_column_left(column_name).name(column_name))

        # Assign a new column
        if k not in self._block.expr.column_names:
            columns.append(get_column_right(v._value.get_name()).name(k))

        expr_builder.columns = columns
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

    def set_index(self, key: str, *, drop=True) -> DataFrame:
        """Set the DataFrame index using existing columns."""
        expr = self._block.expr
        prev_index_columns = self._block.index_columns
        index_expr = typing.cast(ibis_types.Column, expr.get_column(key))

        if not expr.ordering:
            expr = expr.order_by([index_expr])

        expr = expr.drop_columns(prev_index_columns)

        index_column_name = key
        if not drop:
            index_column_name = _INDEX_COLUMN_NAME.format(0)
            index_expr = index_expr.name(index_column_name)
            expr = expr.insert_column(0, index_expr)

        block = self._block.copy()
        block.expr = expr
        block.index = bigframes.core.indexes.index.Index(
            expr, index_column=index_column_name, name=key
        )
        return DataFrame(block)

    def dropna(self) -> DataFrame:
        """Remove rows with missing values."""
        predicates = [
            column.notnull()
            for column in self._block.expr.columns
            if column.get_name() in self._block.value_columns
        ]
        block = self._block.copy()
        for predicate in predicates:
            block.expr = block.expr.filter(predicate)
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

        # Drop index column in joins. Consistent with pandas.
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

    def abs(self) -> DataFrame:
        return self._apply_to_rows(ops.abs_op)

    def isnull(self) -> DataFrame:
        return self._apply_to_rows(ops.isnull_op)

    isna = isnull

    def notnull(self) -> DataFrame:
        return self._apply_to_rows(ops.notnull_op)

    notna = notnull

    def to_parquet(self, path):
        """Write DataFrame to parquet file(s) on GCS."""
        # TODO(swast): Support index=True argument.
        # TODO(swast): Can we support partition columns argument?
        if not path.startswith("gs://"):
            raise NotImplementedError(
                "Only Google Cloud Storage (gs://...) paths are supported."
            )
        expr = self._block.expr
        bqclient = expr._session.bqclient
        value_columns = (expr.get_column(column_name) for column_name in self.columns)
        expr = expr.projection(value_columns)
        query_job: bigquery.QueryJob = expr.start_query()
        query_job.result()  # Wait for query to finish.
        query_job.reload()  # Download latest job metadata.
        # TODO(swast): Some warning that wildcard is recommended for large
        # query results? See:
        # https://cloud.google.com/bigquery/docs/exporting-data#limit_the_exported_file_size
        destination_table = query_job.destination
        extract_job = bqclient.extract_table(destination_table, destination_uris=[path])
        extract_job.result()

    def _apply_to_rows(self, operation):
        block = self._block
        columns = block._expr.columns
        new_columns = [
            operation(column).name(column.get_name())
            for column in columns
            if column.get_name() not in block.index_columns
        ]
        new_block = block.copy(new_columns)
        return DataFrame(new_block)
