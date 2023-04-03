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

"""DataFrame is a two dimensional data structure."""

from __future__ import annotations

import operator
import re
import typing
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import google.cloud.bigquery as bigquery
import ibis
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types
import pandas as pd

import bigframes.core
import bigframes.core.blocks as blocks
import bigframes.core.indexes as indexes
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.series


class DataFrame:
    """A 2D data structure, representing data and deferred computation.

    .. warning::
        This constructor is **private**. Use a public method such as
        ``Session.read_gbq`` to construct a DataFrame.
    """

    def __init__(self, block: blocks.Block, columns: Optional[Sequence[str]] = None):
        self._block = block
        # One on one match between BF column names and real value column names in BQ SQL.
        self._col_names = list(columns) if columns else list(self._block.value_columns)

    def _copy(
        self, columns: Optional[Tuple[Sequence[ibis_types.Value], Sequence[str]]] = None
    ) -> DataFrame:
        if not columns:
            return DataFrame(self._block.copy())

        value_cols, col_names = columns
        if len(value_cols) != len(col_names):
            raise ValueError(
                f"Column sizes not equal. Value columns size: {len(value_cols)}, column names size: {len(col_names)}"
            )

        block = self._block.copy(value_cols)
        return DataFrame(block, col_names)

    def _find_indices(
        self, columns: Union[str, Sequence[str]], tolerance: bool = False
    ) -> Sequence[int]:
        """Find corresponding indices in df._column_names for column name(s). Order is kept the same as input names order.

        Args:
            columns: column name(s)
            tolerance: True to pass through columns not found. False to raise ValueError.
        """
        columns = [columns] if isinstance(columns, str) else columns

        # Dict of {col_name -> [indices]}
        col_indices_dict: Dict[str, List[int]] = {}
        for i, col_name in enumerate(self._col_names):
            col_indices_dict[col_name] = col_indices_dict.get(col_name, [])
            col_indices_dict[col_name].append(i)

        indices = []
        for n in columns:
            if n not in col_indices_dict:
                if not tolerance:
                    raise ValueError(f"Column name {n} doesn't exist")
            else:
                indices += col_indices_dict[n]
        return indices

    def _sql_names(
        self, columns: Union[str, Sequence[str]], tolerance: bool = False
    ) -> Sequence[str]:
        """Retrieve sql name (column name in BQ schema) of column(s)."""
        indices = self._find_indices(columns, tolerance)

        return [self._block.value_columns[i] for i in indices]

    @property
    def index(
        self,
    ) -> Union[indexes.ImplicitJoiner, indexes.Index,]:
        return self._block.index

    @property
    def dtypes(self) -> pd.Series:
        """Returns the dtypes as a Pandas Series object"""
        ibis_dtypes = [
            dtype
            for col, dtype in self._block.expr.to_ibis_expr(ordering_mode="unordered")
            .schema()
            .items()
            if col not in self._block.index_columns
        ]
        bigframes_dtypes = [
            bigframes.dtypes.ibis_dtype_to_bigframes_dtype(ibis_dtype)
            for ibis_dtype in ibis_dtypes
        ]
        return pd.Series(data=bigframes_dtypes, index=self._col_names)

    @property
    def columns(self) -> pd.Index:
        """Returns the column labels of the dataframe"""
        return self.dtypes.index

    @property
    def shape(self) -> Tuple[int, int]:
        """Return a tuple representing the dimensionality of the DataFrame."""
        job = self._block.expr.start_query().result()
        rows = job.total_rows
        cols = len(self.columns)
        return (rows, cols)

    @property
    def size(self) -> int:
        rows, cols = self.shape
        return rows * cols

    @property
    def sql(self) -> str:
        """Compiles this dataframe's expression tree to SQL"""
        # Has to be unordered as it is impossible to order the sql without
        # including metadata columns in selection with ibis.
        return self._block.expr.to_ibis_expr(ordering_mode="unordered").compile()

    def __getitem__(
        self, key: Union[str, Sequence[str], bigframes.series.Series]
    ) -> Union[bigframes.series.Series, "DataFrame"]:
        """Gets the specified column(s) from the DataFrame."""
        # NOTE: This implements the operations described in
        # https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html

        if isinstance(key, bigframes.series.Series):
            return self._getitem_bool_series(key)

        sql_names = self._sql_names(key)
        # Only input is a str and only find one column, returns a Series
        if isinstance(key, str) and len(sql_names) == 1:
            sql_name = sql_names[0]
            # Check that the column exists.
            # TODO(swast): Make sure we can't select Index columns this way.
            index_exprs = [
                self._block.expr.get_column(index_key)
                for index_key in self._block.index_columns
            ]
            series_expr = self._block.expr.get_column(sql_name)
            # Copy to emulate "Copy-on-Write" semantics.
            block = self._block.copy()
            # Since we're working with a "copy", we can drop the other value
            # columns. They aren't needed on the Series.
            block.expr = block.expr.projection(index_exprs + [series_expr])
            block.index.name = self._block.index.name
            # key can only be str or [str] with 1 item when sql_names only has 1 item
            series_name = key if isinstance(key, str) else key[0]
            return bigframes.series.Series(block, sql_name, name=series_name)

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

        # Select a number of columns as DF.
        value_cols = self._block.get_value_col_exprs(sql_names)
        col_names = []
        for item_name in key:
            for col_name in self._col_names:
                if item_name == col_name:
                    col_names.append(col_name)

        return self._copy((value_cols, col_names))

    # Bool Series selects rows
    def _getitem_bool_series(self, key: bigframes.series.Series) -> DataFrame:
        if not key._to_ibis_expr().type() == ibis_dtypes.bool:
            raise ValueError("Only boolean series currently supported for indexing.")
            # TODO: enforce stricter alignment
        combined_index, (
            get_column_left,
            get_column_right,
        ) = self._block.index.join(key.index, how="left")
        right = get_column_right(key._value_column)

        aligned_expr = combined_index._expr
        filtered_expr = aligned_expr.filter((right == ibis.literal(True)))

        column_names = self._block.expr.column_names.keys()
        expression = filtered_expr.projection(
            [get_column_left(left_col).name(left_col) for left_col in column_names]
        )

        block = blocks.Block(expression)
        block.index = (
            indexes.Index(expression, self.index._index_column, self.index.name)
            if isinstance(self.index, indexes.Index)
            else indexes.ImplicitJoiner(expression, self.index.name)
        )
        return DataFrame(block, self._col_names)

    def __getattr__(self, key: str):
        if key not in self._col_names:
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
        preview = preview.set_axis(self._col_names, axis=1)

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
        value_cols = []
        for value_col in self._block.get_value_col_exprs():
            value_cols.append(
                (op(scalar, value_col) if reverse else op(value_col, scalar)).name(
                    value_col.get_name()
                )
            )
        return self._copy((value_cols, self._col_names))

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
        df = self._block.compute()
        return df.set_axis(self._col_names, axis=1)

    def head(self, n: int = 5) -> DataFrame:
        """Limits DataFrame to a specific number of rows."""
        df = self._copy()
        df._block.expr = self._block.expr.apply_limit(n)
        return df

    def drop(self, columns: Union[str, Iterable[str]]) -> DataFrame:
        """Drop specified column(s)."""
        if isinstance(columns, str):
            columns = [columns]
        columns = list(columns)

        df = self._copy()
        df._block.expr = self._block.expr.drop_columns(self._sql_names(columns))
        df._col_names = [
            col_name for col_name in self._col_names if col_name not in columns
        ]
        return df

    def rename(self, columns: Mapping[str, str]) -> DataFrame:
        """Alter column labels."""
        # TODO(garrettwu) Support function(Callable) as columns parameter.
        col_names = [(columns.get(col_name, col_name)) for col_name in self._col_names]
        return self._copy((self._block.get_value_col_exprs(), col_names))

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

    def _assign_single_item(
        self, k: str, v: Union[bigframes.series.Series, int, float]
    ) -> DataFrame:
        if isinstance(v, bigframes.series.Series):
            return self._assign_series_join_on_index(k, v)
        else:
            return self._assign_scalar(k, v)

    def _assign_scalar(self, k: str, v: Union[int, float]) -> DataFrame:
        scalar = bigframes.dtypes.literal_to_ibis_scalar(v)
        scalar = scalar.name(k)

        value_cols = self._block.get_value_col_exprs()
        col_names = list(self._col_names)

        sql_names = self._sql_names(k, tolerance=True)
        if sql_names:
            for i, value_col in enumerate(value_cols):
                if value_col.get_name() in sql_names:
                    value_cols[i] = scalar
        else:
            # TODO(garrettwu): Make sure sql name won't conflict.
            value_cols.append(scalar)
            col_names.append(k)

        return self._copy((value_cols, col_names))

    def _assign_series_join_on_index(self, k, v: bigframes.series.Series) -> DataFrame:
        joined_index, (get_column_left, get_column_right) = self.index.join(
            v.index, how="left"
        )

        sql_names = self._sql_names(k, tolerance=True)
        col_names = list(self._col_names)

        # Restore original column names
        value_cols = []
        for column_name in self._block.value_columns:
            # If it is a replacement, then replace with column from right
            if column_name in sql_names:
                value_cols.append(
                    get_column_right(v._value.get_name()).name(column_name)
                )
            elif column_name:
                value_cols.append(get_column_left(column_name).name(column_name))

        # Assign a new column
        if not sql_names:
            # TODO(garrettwu): make sure sql name doesn't already exist.
            value_cols.append(get_column_right(v._value.get_name()).name(k))
            col_names.append(k)

        block = blocks.Block(joined_index._expr)
        block.index = joined_index
        block.replace_value_columns(value_cols)
        return DataFrame(block, col_names)

    def reset_index(self, *, drop: bool = False) -> DataFrame:
        """Reset the index of the DataFrame, and use the default one instead."""
        original_index_columns = self._block.index_columns
        block = self._block.copy()
        # TODO(swast): Only remove a specified number of levels from a
        # MultiIndex.
        # TODO(swast): Create new sequential index and materialize.
        block.index_columns = ()
        col_names = list(self._col_names)

        if drop:
            # Even though the index might be part of the ordering, keep that
            # ordering expression as reset_index shouldn't change the row
            # order.
            block.expr = block.expr.drop_columns(original_index_columns)
        else:
            col_names = list(original_index_columns) + col_names

        return DataFrame(block, col_names)

    def set_index(self, key: str, *, drop: bool = True) -> DataFrame:
        """Set the DataFrame index using existing columns."""
        expr = self._block.expr
        prev_index_columns = self._block.index_columns
        index_expr = typing.cast(ibis_types.Column, expr.get_column(key))

        # TODO(swast): Don't override ordering once all DataFrames/Series have
        # an ordering.
        if not expr.ordering or (
            len(expr.ordering)
            and expr.ordering[0].get_name() == bigframes.core.ORDER_ID_COLUMN
        ):
            expr = expr.order_by([key])

        expr = expr.drop_columns(prev_index_columns)

        col_names = list(self._col_names)
        index_column_name = key
        if not drop:
            index_column_name = indexes.INDEX_COLUMN_NAME.format(0)
            index_expr = index_expr.name(index_column_name)
            expr = expr.insert_column(0, index_expr)
        else:
            col_names.remove(key)

        block = self._block.copy()
        block.expr = expr
        block.index_columns = self._sql_names(key)
        block.index = bigframes.core.indexes.index.Index(
            expr, index_column=index_column_name, name=key
        )
        return DataFrame(block, col_names)

    def sort_index(self) -> DataFrame:
        """Sort the DataFrame by index labels."""
        index_columns = self._block.index_columns
        expr = self._block.expr.order_by(index_columns)
        block = self._block.copy()
        block.expr = expr
        return DataFrame(block)

    def dropna(self) -> DataFrame:
        """Remove rows with missing values."""
        predicates = [
            column.notnull()
            for column in self._block.expr.columns
            if column.get_name() in self._block.value_columns
        ]
        df = self._copy()
        for predicate in predicates:
            df._block.expr = df._block.expr.filter(predicate)
        return df

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

        left_on_sql = self._sql_names(on)
        # 0 elements alreasy throws an exception
        if len(left_on_sql) > 1:
            raise ValueError(f"The column label {on} is not unique.")
        left_on_sql = left_on_sql[0]

        right_on_sql = right._sql_names(on)
        if len(right_on_sql) > 1:
            raise ValueError(f"The column label {on} is not unique.")
        right_on_sql = right_on_sql[0]

        # Drop index column in joins. Consistent with pandas.
        left = self.reset_index(drop=True)
        right = right.reset_index(drop=True)

        # TODO(tbergeron): Merge logic with index join and applying deterministic post-join order
        left_table = left._block.expr.to_ibis_expr(ordering_mode="unordered")
        right_table = right._block.expr.to_ibis_expr(ordering_mode="unordered")

        joined_table = left_table.join(
            right_table,
            left_table[left_on_sql] == right_table[right_on_sql],
            how,
            suffixes=suffixes,
        )
        block = blocks.Block(
            bigframes.core.BigFramesExpr(left._block.expr._session, joined_table)
        )
        joined_frame = DataFrame(block)

        # Ibis emits redundant columns for outer joins. See:
        # https://ibis-project.org/ibis-for-pandas-users/#merging-tables
        left_on_name = left_on_sql + suffixes[0]
        right_on_name = right_on_sql + suffixes[1]
        if (
            left_on_name in joined_frame._block.expr.column_names
            and right_on_name in joined_frame._block.expr.column_names
        ):
            joined_frame = joined_frame.drop(right_on_name)
            joined_frame = joined_frame.rename({left_on_name: on})

        joined_frame._col_names = self._get_merged_col_names(right, on, suffixes)

        return joined_frame

    def _get_merged_col_names(
        self, right: DataFrame, on: str, suffixes: tuple[str, str] = ("_x", "_y")
    ) -> List[str]:
        left_col_names = [
            (
                col_name + suffixes[0]
                if col_name in right._col_names and col_name != on
                else col_name
            )
            for col_name in self._col_names
        ]
        right_col_names = [
            (
                col_name + suffixes[1]
                if col_name in self._col_names and col_name != on
                else col_name
            )
            for col_name in right._col_names
        ]
        right_col_names.remove(on)
        return left_col_names + right_col_names

    def join(self, other: DataFrame, how: str) -> DataFrame:
        """Join columns of another dataframe"""

        if not self.columns.intersection(other.columns).empty:
            raise NotImplementedError("Deduping column names is not implemented")

        left = self
        right = other
        combined_index, (get_column_left, get_column_right) = left.index.join(
            right.index, how=how
        )

        block = blocks.Block(combined_index._expr)
        block.index = combined_index

        index_columns = []
        if isinstance(combined_index, indexes.Index):
            index_columns = [
                # TODO(swast): Support MultiIndex.
                combined_index._expr.get_any_column(combined_index._index_column)
            ]

        expr_bldr = block.expr.builder()
        expr_bldr.columns = (
            index_columns
            + [
                # TODO(swast): Support suffix if there are duplicates.
                get_column_left(col_name).name(col_name)
                for col_name in left.columns
            ]
            + [
                # TODO(swast): Support suffix if there are duplicates.
                get_column_right(col_name).name(col_name)
                for col_name in right.columns
            ]
        )
        # TODO(swast): Maintain some ordering post-join.
        block.expr = expr_bldr.build()
        return DataFrame(block, self._col_names + other._col_names)

    def abs(self) -> DataFrame:
        return self._apply_to_rows(ops.abs_op)

    def isnull(self) -> DataFrame:
        return self._apply_to_rows(ops.isnull_op)

    isna = isnull

    def notnull(self) -> DataFrame:
        return self._apply_to_rows(ops.notnull_op)

    notna = notnull

    def to_pandas(self) -> pd.DataFrame:
        """Writes DataFrame to Pandas DataFrame."""
        # TODO(chelsealin): Support block parameters.
        # TODO(chelsealin): Add to_pandas_batches() API.
        return self.compute()

    def to_csv(self, paths: str) -> None:
        """Writes DataFrame to comma-separated values (csv) file(s) on GCS.

        Args:
            paths: a destination URIs of GCS files(s) to store the extracted dataframe in format of
                ``gs://<bucket_name>/<object_name_or_glob>``.
                If the data size is more than 1GB, you must use a wildcard to export the data into
                multiple files and the size of the files varies.

        Returns:
            None.
        """
        # TODO(swast): Support index=True argument.
        # TODO(swast): Can we support partition columns argument?
        # TODO(chelsealin): Support local file paths.
        # TODO(swast): Some warning that wildcard is recommended for large
        # query results? See:
        # https://cloud.google.com/bigquery/docs/exporting-data#limit_the_exported_file_size
        if not paths.startswith("gs://"):
            raise NotImplementedError(
                "Only Google Cloud Storage (gs://...) paths are supported."
            )

        source_table = self._get_destination_table()
        job_config = bigquery.ExtractJobConfig(
            destination_format=bigquery.DestinationFormat.CSV
        )
        extract_job = self._block.expr._session.bqclient.extract_table(
            source_table, destination_uris=[paths], job_config=job_config
        )
        extract_job.result()  # Wait for extract job to finish

    def to_gbq(
        self,
        destination_table: str,
        *,
        if_exists: Optional[Literal["fail", "replace", "append"]] = "fail",
    ) -> None:
        """Writes the BigFrames DataFrame as a BigQuery table.

        Args:
            destination_table:
                Name of table to be written, in the form `dataset.tablename` or
                `project.dataset.tablename`.

            if_exists:
                Behavior when the destination table exists. Value can be one of:
                - `fail`: raise google.api_core.exceptions.Conflict.
                - `replace`: If table exists, drop it, recreate it, and insert data.
                - `append`: If table exists, insert data. Create if it does not exist.

        Returns:
            None.
        """

        if "." not in destination_table:
            raise ValueError(
                "Invalid Table Name. Should be of the form 'datasetId.tableId' or "
                "'projectId.datasetId.tableId'"
            )

        dispositions = {
            "fail": bigquery.WriteDisposition.WRITE_EMPTY,
            "replace": bigquery.WriteDisposition.WRITE_TRUNCATE,
            "append": bigquery.WriteDisposition.WRITE_APPEND,
        }
        if if_exists not in dispositions:
            raise ValueError("'{0}' is not valid for if_exists".format(if_exists))

        job_config = bigquery.QueryJobConfig(
            write_disposition=dispositions[if_exists],
            destination=bigquery.table.TableReference.from_string(
                destination_table,
                default_project=self._block.expr._session.bqclient.project,
            ),
        )

        self._get_destination_table(job_config=job_config)

    def to_parquet(self, paths: str) -> None:
        """Writes DataFrame to parquet file(s) on GCS.

        Args:
            paths: a destination URIs of GCS files(s) to store the extracted dataframe in format of
                ``gs://<bucket_name>/<object_name_or_glob>``.
                If the data size is more than 1GB, you must use a wildcard to export the data into
                multiple files and the size of the files varies.

        Returns:
            None.
        """
        # TODO(swast): Support index=True argument.
        # TODO(swast): Can we support partition columns argument?
        # TODO(chelsealin): Support local file paths.
        # TODO(swast): Some warning that wildcard is recommended for large
        # query results? See:
        # https://cloud.google.com/bigquery/docs/exporting-data#limit_the_exported_file_size
        if not paths.startswith("gs://"):
            raise NotImplementedError(
                "Only Google Cloud Storage (gs://...) paths are supported."
            )

        source_table = self._get_destination_table()
        job_config = bigquery.ExtractJobConfig(
            destination_format=bigquery.DestinationFormat.PARQUET
        )
        extract_job = self._block.expr._session.bqclient.extract_table(
            source_table, destination_uris=[paths], job_config=job_config
        )
        extract_job.result()  # Wait for extract job to finish

    def _apply_to_rows(self, operation) -> DataFrame:
        value_cols = [
            operation(value_col).name(value_col.get_name())
            for value_col in self._block.get_value_col_exprs()
        ]
        return self._copy((value_cols, self._col_names))

    def _get_destination_table(
        self, job_config: Optional[bigquery.job.QueryJobConfig] = None
    ):
        """Execute a query job presenting the DataFrames and returns the destination table."""
        expr = self._block.expr
        value_columns = (expr.get_column(column_name) for column_name in self.columns)
        expr = expr.projection(value_columns)
        query_job: bigquery.QueryJob = expr.start_query(job_config)
        query_job.result()  # Wait for query to finish.
        query_job.reload()  # Download latest job metadata.
        return query_job.destination
