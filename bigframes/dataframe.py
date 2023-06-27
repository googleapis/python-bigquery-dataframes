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

import random
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
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types
import numpy
import pandas as pd
import typing_extensions

import bigframes.core
import bigframes.core.blocks as blocks
import bigframes.core.groupby as groupby
import bigframes.core.guid
import bigframes.core.indexers as indexers
import bigframes.core.indexes as indexes
import bigframes.core.joins as joins
import bigframes.core.ordering as order
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops
import bigframes.series
import bigframes.series as bf_series
import bigframes.session
import third_party.bigframes_vendored.pandas.core.frame as vendored_pandas_frame
import third_party.bigframes_vendored.pandas.io.common as vendored_pandas_io_common
import third_party.bigframes_vendored.pandas.pandas._typing as vendored_pandas_typing

# BigQuery has 1 MB query size limit, 5000 items shouldn't take more than 10% of this depending on data type.
# TODO(tbergeron): Convert to bytes-based limit
MAX_INLINE_DF_SIZE = 5000


# Inherits from pandas DataFrame so that we can use the same docstrings.
class DataFrame(vendored_pandas_frame.DataFrame):
    __doc__ = vendored_pandas_frame.DataFrame.__doc__

    def __init__(
        self,
        data=None,
        index: vendored_pandas_typing.Axes | None = None,
        columns: vendored_pandas_typing.Axes | None = None,
        dtype: typing.Optional[
            bigframes.dtypes.BigFramesDtypeString | bigframes.dtypes.BigFramesDtype
        ] = None,
        copy: typing.Optional[bool] = None,
        *,
        session: typing.Optional[bigframes.session.Session] = None,
    ):
        if copy is not None and not copy:
            raise ValueError("DataFrame constructor only supports copy=True")

        # Check to see if constructing from BigFrames objects before falling back to pandas constructor
        block = None
        if isinstance(data, blocks.Block):
            block = data

        elif isinstance(data, DataFrame):
            block = data._get_block()

        # Dict of Series
        elif (
            _is_dict_like(data)
            and len(data) >= 1
            and any(isinstance(data[key], bf_series.Series) for key in data.keys())
        ):
            if not all(isinstance(data[key], bf_series.Series) for key in data.keys()):
                # TODO(tbergeron): Support local list/series data by converting to memtable.
                raise NotImplementedError(
                    "Cannot mix BigFrames Series with other types."
                )
            keys = list(data.keys())
            first_label, first_series = keys[0], data[keys[0]]
            block = (
                typing.cast(bf_series.Series, first_series)
                ._get_block()
                .with_column_labels([first_label])
            )

            for key in keys[1:]:
                other = typing.cast(bf_series.Series, data[key])
                other_block = other._block.with_column_labels([key])
                # Pandas will keep original sorting if all indices are aligned.
                # We cannot detect this easily however, and so always sort on index
                result_index, _ = block.index.join(  # type:ignore
                    other_block.index, how="outer", sort=True
                )
                block = result_index._block

        if block:
            if index:
                raise NotImplementedError(
                    "DataFrame 'index' constructor parameter not supported when passing BigFrames objects"
                )
            if columns:
                block = block.select_columns(list(columns))  # type:ignore
            if dtype:
                block = block.multi_apply_unary_op(
                    block.value_columns, ops.AsTypeOp(dtype)
                )
            self._block = block

        else:
            import bigframes.pandas

            pd_dataframe = pd.DataFrame(
                data=data,
                index=index,  # type:ignore
                columns=columns,  # type:ignore
                dtype=dtype,  # type:ignore
            )
            if pd_dataframe.size < MAX_INLINE_DF_SIZE:
                self._block = blocks.block_from_local(
                    pd_dataframe, session or bigframes.pandas.get_global_session()
                )
            elif session:
                self._block = session.read_pandas(pd_dataframe)._get_block()
            else:
                self._block = bigframes.pandas.read_pandas(pd_dataframe)._get_block()

    def __dir__(self):
        return dir(type(self)) + self._block.column_labels

    def _ipython_key_completions_(self) -> List[str]:
        return list([label for label in self._block.column_labels if label])

    def _find_indices(
        self,
        columns: Union[blocks.Label, Sequence[blocks.Label]],
        tolerance: bool = False,
    ) -> Sequence[int]:
        """Find corresponding indices in df._block.column_labels for column name(s).
        Order is kept the same as input names order.

        Args:
            columns: column name(s)
            tolerance: True to pass through columns not found. False to raise
                ValueError.
        """
        columns = columns if _is_list_like(columns) else [columns]  # type:ignore

        # Dict of {col_label -> [indices]}
        col_indices_dict: Dict[blocks.Label, List[int]] = {}
        for i, col_label in enumerate(self._block.column_labels):
            col_indices_dict[col_label] = col_indices_dict.get(col_label, [])
            col_indices_dict[col_label].append(i)

        indices = []
        for n in columns:
            if n not in col_indices_dict:
                if not tolerance:
                    raise ValueError(f"Column name {n} doesn't exist")
            else:
                indices += col_indices_dict[n]
        return indices

    def _sql_names(
        self,
        columns: Union[blocks.Label, Sequence[blocks.Label]],
        tolerance: bool = False,
    ) -> Sequence[str]:
        """Retrieve sql name (column name in BQ schema) of column(s)."""
        indices = self._find_indices(columns, tolerance)

        return [self._block.value_columns[i] for i in indices]

    @property
    def index(
        self,
    ) -> indexes.Index:
        return indexes.Index(self)

    @property
    def loc(self) -> indexers._LocIndexer:
        return indexers._LocIndexer(self)

    @property
    def iloc(self) -> indexers._iLocIndexer:
        return indexers._iLocIndexer(self)

    @property
    def dtypes(self) -> pd.Series:
        return pd.Series(data=self._block.dtypes, index=self._block.column_labels)

    @property
    def columns(self) -> pd.Index:
        return self.dtypes.index

    @property
    def shape(self) -> Tuple[int, int]:
        block_length, _ = self._block.expr.shape()
        return (block_length, len(self.columns))

    @property
    def size(self) -> int:
        rows, cols = self.shape
        return rows * cols

    @property
    def ndim(self) -> int:
        return 2

    @property
    def empty(self) -> bool:
        # TODO(swast): Should also return true if there are columns but no
        # rows.
        return not bool(self._block.value_columns)

    @property
    def values(self) -> numpy.ndarray:
        return self.to_numpy()

    def astype(
        self,
        dtype: Union[
            bigframes.dtypes.BigFramesDtypeString, bigframes.dtypes.BigFramesDtype
        ],
    ) -> DataFrame:
        return self._apply_to_rows(ops.AsTypeOp(dtype))

    def to_sql_query(
        self, always_include_index: bool
    ) -> Tuple[str, List[Tuple[str, bool]]]:
        """Compiles this dataframe's expression tree to SQL, optionally
        including unnamed index columns

        Args:
            always_include_index: whether to include unnamed index columns.
            If False, only named indexes are included.

        Returns: a tuple of (sql_string, index_column_list). Each entry in the
            index column list is a tuple of (column_name, named). If named is
            is false, then the column name exists only in SQL"""
        # Has to be unordered as it is impossible to order the sql without
        # including metadata columns in selection with ibis.
        ibis_expr = self._block.expr.to_ibis_expr(ordering_mode="unordered")
        column_labels = self._block.column_labels

        # TODO(swast): Need to have a better way of controlling when to include
        # the index or not.
        index_has_name = self.index.name is not None
        if index_has_name:
            column_labels = column_labels + [typing.cast(str, self.index.name)]
        elif always_include_index:
            # In this mode include the index even if it is a nameless generated
            # column like 'bigframes_index_0'
            column_labels = column_labels + [self._block.index_columns[0]]

        column_labels_deduped = typing.cast(
            List[str],
            vendored_pandas_io_common.dedup_names(
                column_labels, is_potential_multiindex=False
            ),
        )
        column_ids = self._block.value_columns
        substitutions = {}
        for column_id, column_label in zip(column_ids, column_labels_deduped):
            # TODO(swast): Do we need to further escape this, or can we rely on
            # the BigQuery unicode column name feature?
            substitutions[column_id] = column_label

        index_cols: List[Tuple[str, bool]] = []
        if index_has_name or always_include_index:
            if len(self._block.index_columns) != 1:
                raise NotImplementedError("Only exactly 1 index column is supported")

            substitutions[self._block.index_columns[0]] = column_labels_deduped[-1]
            index_cols = [(column_labels_deduped[-1], index_has_name)]
        else:
            ibis_expr = ibis_expr.drop(self._block.index_columns[0])

        ibis_expr = ibis_expr.relabel(substitutions)
        return typing.cast(str, ibis_expr.compile()), index_cols

    @property
    def sql(self) -> str:
        """Compiles this dataframe's expression tree to SQL"""
        sql, _ = self.to_sql_query(always_include_index=False)
        return sql

    def __getitem__(
        self, key: Union[blocks.Label, Sequence[blocks.Label], bigframes.series.Series]
    ) -> Union[bigframes.series.Series, "DataFrame"]:
        """Gets the specified column(s) from the DataFrame."""
        # NOTE: This implements the operations described in
        # https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html

        if isinstance(key, bigframes.series.Series):
            return self._getitem_bool_series(key)

        sql_names = self._sql_names(key)
        # Only input is a str and only find one column, returns a Series
        if isinstance(key, str) and len(sql_names) == 1:
            return bigframes.series.Series(self._block.select_column(sql_names[0]))

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
        # TODO(swast): Do we need to apply implicit join when doing a
        # projection?

        # Select a number of columns as DF.
        key = key if _is_list_like(key) else [key]  # type:ignore

        selected_ids: Tuple[str, ...] = ()
        for label in key:
            col_ids = self._block.label_to_col_id.get(label, [])
            selected_ids = (*selected_ids, *col_ids)

        return DataFrame(self._block.select_columns(selected_ids))

    # Bool Series selects rows
    def _getitem_bool_series(self, key: bigframes.series.Series) -> DataFrame:
        if not key._to_ibis_expr().type() == ibis_dtypes.bool:
            raise ValueError("Only boolean series currently supported for indexing.")
            # TODO: enforce stricter alignment
        combined_index, (
            get_column_left,
            get_column_right,
        ) = self._block.index.join(key._block.index, how="left")
        block = combined_index._block
        filter_col_id = get_column_right(key._value_column)
        block = block.filter(filter_col_id)
        block = block.drop_columns([filter_col_id])
        return DataFrame(block)

    def __getattr__(self, key: str):
        if key in self._block.column_labels:
            return self.__getitem__(key)
        elif hasattr(pd.DataFrame(), key):
            raise NotImplementedError(
                "BigFrames has not yet implemented an equivalent to pandas.DataFrame.{key} . Please check https://github.com/googleapis/bigframes/issues for existing feature requests, or file your own. You may also send feedback to the bigframes team at bigframes-feedback@google.com. You may include information about your use case, as well as code snippets or other feedback.".format(
                    key=key
                )
            )
        else:
            raise AttributeError(key)

    def __repr__(self) -> str:
        """Converts a DataFrame to a string. Calls compute. Only represents the first
        25 results."""
        max_results = 25
        pandas_df, row_count = self._retrieve_repr_request_results(max_results)
        column_count = len(pandas_df.columns)

        # Modify the end of the string to reflect count.
        repr_string = repr(pandas_df)
        lines = repr_string.split("\n")
        pattern = re.compile("\\[[0-9]+ rows x [0-9]+ columns\\]")
        if pattern.match(lines[-1]):
            lines = lines[:-2]

        if row_count > len(lines) - 1:
            lines.append("...")

        lines.append("")
        lines.append(f"[{row_count} rows x {column_count} columns]")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """
        Returns an html string primarily for use by notebooks for displaying
        a representation of the DataFrame. Displays 20 rows by default since
        many notebooks are not configured for large tables.
        """
        max_results = 20
        pandas_df, row_count = self._retrieve_repr_request_results(max_results)
        column_count = len(pandas_df.columns)
        # _repr_html_ stub is missing so mypy thinks it's a Series. Ignore mypy.
        html_string = pandas_df._repr_html_()  # type:ignore
        html_string += f"[{row_count} rows x {column_count} columns in total]"
        return html_string

    def _retrieve_repr_request_results(
        self, max_results: int
    ) -> Tuple[pd.DataFrame, int]:
        """
        Retrieves a pandas dataframe containing only max_results many rows for use
        with printing methods.

        Returns a tuple of the dataframe and the overall number of rows of the query.
        """
        head_df = self.head(n=max_results)
        computed_df = head_df._block.compute(max_results=max_results)
        count = self.shape[0]
        formatted_df = computed_df.set_axis(self._block.column_labels, axis=1)
        # we reset the axis and substitute the bf index name for the default
        formatted_df.index.name = self.index.name
        return formatted_df, count

    def _apply_binop(
        self,
        other: float | int | bigframes.Series,
        op,
        axis: str | int = "columns",
    ):
        if isinstance(other, (float, int)):
            return self._apply_scalar_binop(other, op)
        elif isinstance(other, bigframes.Series):
            return self._apply_series_binop(other, op, axis=axis)
        raise NotImplementedError(
            f"binary operation is not implemented on the second operand of type {type(other).__name__}."
        )

    def _apply_scalar_binop(self, other: float | int, op: ops.BinaryOp) -> DataFrame:
        block, constant_col_id = self._block.create_constant(other)
        for column_id, label in zip(
            self._block.value_columns, self._block.column_labels
        ):
            block, _ = block.apply_binary_op(
                column_id,
                constant_col_id,
                op,
                result_label=label,
            )
            block = block.drop_columns([column_id])
        block = block.drop_columns([constant_col_id])
        return DataFrame(block)

    def _apply_series_binop(
        self,
        other: bigframes.Series,
        op: ops.BinaryOp,
        axis: str | int = "columns",
    ) -> DataFrame:
        if axis not in ("columns", "index", 0, 1):
            raise ValueError(f"Invalid input: axis {axis}.")

        if axis in ("columns", 1):
            raise NotImplementedError("Row Series operations haven't been supported.")

        joined_index, (get_column_left, get_column_right) = self._block.index.join(
            other._block.index, how="outer"
        )

        series_column_id = other._value.get_name()
        series_col = get_column_right(series_column_id)
        block = joined_index._block
        for column_id, label in zip(
            self._block.value_columns, self._block.column_labels
        ):
            block, _ = block.apply_binary_op(
                get_column_left(column_id),
                series_col,
                op,
                result_label=label,
            )
            block = block.drop_columns([get_column_left(column_id)])

        block = block.drop_columns([series_col])
        block = block.with_index_labels([self.index.name])
        return DataFrame(block)

    def le(self, other: typing.Any, axis: str | int = "columns") -> DataFrame:
        return self._apply_binop(other, ops.le_op, axis=axis)

    def lt(self, other: typing.Any, axis: str | int = "columns") -> DataFrame:
        return self._apply_binop(other, ops.lt_op, axis=axis)

    def ge(self, other: typing.Any, axis: str | int = "columns") -> DataFrame:
        return self._apply_binop(other, ops.ge_op, axis=axis)

    def gt(self, other: typing.Any, axis: str | int = "columns") -> DataFrame:
        return self._apply_binop(other, ops.gt_op, axis=axis)

    __lt__ = lt

    __le__ = le

    __gt__ = gt

    __ge__ = ge

    def add(
        self, other: float | int | bigframes.Series, axis: str | int = "columns"
    ) -> DataFrame:
        # TODO(swast): Support fill_value parameter.
        # TODO(swast): Support level parameter with MultiIndex.
        return self._apply_binop(other, ops.add_op, axis=axis)

    __radd__ = __add__ = radd = add

    def sub(
        self, other: float | int | bigframes.Series, axis: str | int = "columns"
    ) -> DataFrame:
        return self._apply_binop(other, ops.sub_op, axis=axis)

    __sub__ = subtract = sub

    def rsub(
        self, other: float | int | bigframes.Series, axis: str | int = "columns"
    ) -> DataFrame:
        return self._apply_binop(other, ops.reverse(ops.sub_op), axis=axis)

    __rsub__ = rsub

    def mul(
        self, other: float | int | bigframes.Series, axis: str | int = "columns"
    ) -> DataFrame:
        return self._apply_binop(other, ops.mul_op, axis=axis)

    __rmul__ = __mul__ = rmul = multiply = mul

    def truediv(
        self, other: float | int | bigframes.Series, axis: str | int = "columns"
    ) -> DataFrame:
        return self._apply_binop(other, ops.div_op, axis=axis)

    div = divide = __truediv__ = truediv

    def rtruediv(
        self, other: float | int | bigframes.Series, axis: str | int = "columns"
    ) -> DataFrame:
        return self._apply_binop(other, ops.reverse(ops.div_op), axis=axis)

    __rtruediv__ = rdiv = rtruediv

    def floordiv(
        self, other: float | int | bigframes.Series, axis: str | int = "columns"
    ) -> DataFrame:
        return self._apply_binop(other, ops.floordiv_op, axis=axis)

    __floordiv__ = floordiv

    def rfloordiv(
        self, other: float | int | bigframes.Series, axis: str | int = "columns"
    ) -> DataFrame:
        return self._apply_binop(other, ops.reverse(ops.floordiv_op), axis=axis)

    __rfloordiv__ = rfloordiv

    def mod(self, other: int | bigframes.Series, axis: str | int = "columns") -> DataFrame:  # type: ignore
        return self._apply_binop(other, ops.mod_op, axis=axis)

    def rmod(self, other: int | bigframes.Series, axis: str | int = "columns") -> DataFrame:  # type: ignore
        return self._apply_binop(other, ops.reverse(ops.mod_op), axis=axis)

    __mod__ = mod

    __rmod__ = rmod

    def compute(self) -> pd.DataFrame:
        """Executes deferred operations and downloads the results."""
        df = self._block.compute()
        return df.set_axis(self._block.column_labels, axis=1)

    def copy(self) -> DataFrame:
        return DataFrame(self._block)

    def head(self, n: int = 5) -> DataFrame:
        return typing.cast(DataFrame, self.iloc[:n])

    def drop(self, *, columns: Union[str, Iterable[str]]) -> DataFrame:
        if not _is_list_like(columns):
            columns = [columns]  # type:ignore
        columns = list(columns)

        block = self._block.drop_columns(self._sql_names(columns))
        return DataFrame(block)

    def rename(self, *, columns: Mapping[blocks.Label, blocks.Label]) -> DataFrame:
        block = self._block.rename(columns=columns)
        return DataFrame(block)

    def rename_axis(self, mapper: Optional[str], **kwargs) -> DataFrame:
        if len(kwargs) != 0:
            raise NotImplementedError(
                "rename_axis does not currently support any keyword arguments."
            )
        # limited implementation: the new index name is simply the 'mapper' parameter
        block = self._block.with_index_labels([mapper])
        return DataFrame(block)

    def assign(self, **kwargs) -> DataFrame:
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

    def _assign_scalar(self, label: str, value: Union[int, float]) -> DataFrame:
        # TODO(swast): Make sure that k is the ID / SQL name, not a label,
        # which could be invalid SQL.
        col_ids = self._sql_names(label, tolerance=True)

        block, constant_col_id = self._block.create_constant(value, label)
        for col_id in col_ids:
            block = block.copy_values(constant_col_id, col_id)

        if len(col_ids) > 0:
            block = block.drop_columns([constant_col_id])

        return DataFrame(block)

    def _assign_series_join_on_index(
        self, label: str, series: bigframes.series.Series
    ) -> DataFrame:
        joined_index, (get_column_left, get_column_right) = self._block.index.join(
            series._block.index, how="left"
        )

        column_ids = [
            get_column_left(col_id) for col_id in self._sql_names(label, tolerance=True)
        ]
        block = joined_index._block
        source_column = get_column_right(series._value_column)

        # Replace each column matching the label
        for column_id in column_ids:
            block = block.copy_values(source_column, column_id).assign_label(
                column_id, label
            )

        if not column_ids:
            # Append case, so new column needs appropriate label
            block = block.assign_label(source_column, label)
        else:
            # Update case, remove after copying into columns
            block = block.drop_columns([source_column])

        return DataFrame(block.with_index_labels([self.index.name]))

    def reset_index(self, *, drop: bool = False) -> DataFrame:
        original_index_ids = self._block.index_columns
        if len(original_index_ids) != 1:
            raise NotImplementedError("reset_index() doesn't yet support MultiIndex.")
        block = self._block.reset_index(drop)
        return DataFrame(block)

    def set_index(self, keys: str, *, drop: bool = True) -> DataFrame:
        expr = self._block.expr
        prev_index_columns = self._block.index_columns

        matching_col_ids = self._sql_names(keys, tolerance=False)

        if len(matching_col_ids) > 1:
            raise ValueError("Index data must be 1-dimensional")

        key = matching_col_ids[0]

        index_expr = typing.cast(ibis_types.Column, expr.get_column(key))

        expr = expr.drop_columns(prev_index_columns)

        column_labels = list(self._block.column_labels)
        index_columns = self._sql_names(keys)
        if not drop:
            index_column_id = indexes.INDEX_COLUMN_ID.format(0)
            index_expr = index_expr.name(index_column_id)
            expr = expr.insert_column(0, index_expr)
            index_columns = [index_column_id]
        else:
            column_labels.remove(keys)

        block = blocks.Block(
            expr,
            index_columns=index_columns,
            column_labels=column_labels,
            index_labels=[keys],
        )
        return DataFrame(block)

    def sort_index(self) -> DataFrame:
        index_columns = self._block.index_columns
        ordering = [order.OrderingColumnReference(column) for column in index_columns]
        return DataFrame(self._block.order_by(ordering))

    def sort_values(
        self,
        by: str | typing.Sequence[str],
        *,
        ascending: bool | typing.Sequence[bool] = True,
        na_position: typing.Literal["first", "last"] = "last",
    ) -> DataFrame:
        if na_position not in {"first", "last"}:
            raise ValueError("Param na_position must be one of 'first' or 'last'")

        sort_labels = (by,) if isinstance(by, str) else tuple(by)
        sort_column_ids = self._sql_names(sort_labels)

        len_by = len(sort_labels)
        if not isinstance(ascending, bool):
            if len(ascending) != len_by:
                raise ValueError("Length of 'ascending' must equal length of 'by'")
            sort_directions = ascending
        else:
            sort_directions = (ascending,) * len_by

        ordering = []
        for i in range(len(sort_labels)):
            column_id = sort_column_ids[i]
            direction = (
                order.OrderingDirection.ASC
                if sort_directions[i]
                else order.OrderingDirection.DESC
            )
            na_last = na_position == "last"
            ordering.append(
                order.OrderingColumnReference(
                    column_id, direction=direction, na_last=na_last
                )
            )

        return DataFrame(self._block.order_by(ordering))

    def add_prefix(self, prefix: str, axis: int | str | None = None) -> DataFrame:
        return DataFrame(self._get_block().add_prefix(prefix, axis))

    def add_suffix(self, suffix: str, axis: int | str | None = None) -> DataFrame:
        return DataFrame(self._get_block().add_suffix(suffix, axis))

    def dropna(self) -> DataFrame:
        block = self._block
        for column in self._block.value_columns:
            block, result_id = block.apply_unary_op(column, ops.notnull_op)
            block = block.filter(result_id)
            block = block.drop_columns([result_id])

        return DataFrame(block)

    def sum(self, *, numeric_only: bool = False) -> bigframes.Series:
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        block = self._block.aggregate_all_and_pivot(agg_ops.sum_op)
        return bigframes.Series(block.select_column("values"))

    def mean(self, *, numeric_only: bool = False) -> bigframes.Series:
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        block = self._block.aggregate_all_and_pivot(agg_ops.mean_op)
        return bigframes.Series(block.select_column("values"))

    def std(self, *, numeric_only: bool = False) -> bigframes.Series:
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        block = self._block.aggregate_all_and_pivot(agg_ops.std_op)
        return bigframes.Series(block.select_column("values"))

    def var(self, *, numeric_only: bool = False) -> bigframes.Series:
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        block = self._block.aggregate_all_and_pivot(agg_ops.var_op)
        return bigframes.Series(block.select_column("values"))

    def min(self, *, numeric_only: bool = False) -> bigframes.Series:
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        block = self._block.aggregate_all_and_pivot(agg_ops.min_op)
        return bigframes.Series(block.select_column("values"))

    def max(self, *, numeric_only: bool = False) -> bigframes.Series:
        if not numeric_only:
            raise NotImplementedError("Operation only supports 'numeric_only'=True")
        block = self._block.aggregate_all_and_pivot(agg_ops.max_op)
        return bigframes.Series(block.select_column("values"))

    def merge(
        self,
        right: DataFrame,
        how: Literal[
            "inner",
            "left",
            "outer",
            "right",
        ] = "inner",
        # TODO(garrettwu): Currently can take inner, outer, left and right. To support
        # cross joins
        # TODO(garrettwu): Support "on" list of columns and None. Currently a single
        # column must be provided
        on: Optional[str] = None,
        *,
        sort: bool = False,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> DataFrame:
        if not on:
            raise ValueError("Must specify a column to join on.")

        left = self
        left_on_sql = self._sql_names(on)
        # 0 elements alreasy throws an exception
        if len(left_on_sql) > 1:
            raise ValueError(f"The column label {on} is not unique.")
        left_on_sql = left_on_sql[0]

        right_on_sql = right._sql_names(on)
        if len(right_on_sql) > 1:
            raise ValueError(f"The column label {on} is not unique.")
        right_on_sql = right_on_sql[0]

        (
            joined_expr,
            join_key_id,
            (get_column_left, get_column_right),
        ) = joins.join_by_column(
            left._block.expr,
            left_on_sql,
            right._block.expr,
            right_on_sql,
            how=how,
            sort=sort,
        )
        # TODO(swast): Add suffixes to the column labels instead of reusing the
        # column IDs as the new labels.
        # Drop the index column(s) to be consistent with pandas.
        left_columns = [
            join_key_id if col_id == left_on_sql else get_column_left(col_id)
            for col_id in left._block.value_columns
        ]
        right_columns = [
            get_column_right(col_id)
            for col_id in right._block.value_columns
            if col_id != right_on_sql
        ]
        expr = joined_expr.select_columns([*left_columns, *right_columns])
        labels = self._get_merged_col_labels(right, on=on, suffixes=suffixes)

        # Constructs default index
        block = blocks.Block(expr, column_labels=labels)
        return DataFrame(block)

    def _get_merged_col_labels(
        self, right: DataFrame, on: str, suffixes: tuple[str, str] = ("_x", "_y")
    ) -> List[blocks.Label]:
        left_col_labels = [
            (
                ("{name}" + suffixes[0]).format(name=col_label)
                if col_label in right._block.column_labels and col_label != on
                else col_label
            )
            for col_label in self._block.column_labels
        ]
        right_col_labels = [
            (
                ("{name}" + suffixes[1]).format(name=col_label)
                if col_label in self._block.column_labels and col_label != on
                else col_label
            )
            for col_label in right._block.column_labels
            if col_label != on
        ]
        return left_col_labels + right_col_labels

    def join(self, other: DataFrame, *, how: str = "left") -> DataFrame:
        if not self.columns.intersection(other.columns).empty:
            raise NotImplementedError("Deduping column names is not implemented")

        left = self
        right = other
        combined_index, (get_column_left, get_column_right) = left._block.index.join(
            right._block.index, how=how
        )
        return DataFrame(combined_index._block)

    def groupby(
        self,
        by: typing.Union[blocks.Label, typing.Sequence[blocks.Label]],
        *,
        as_index: bool = True,
        dropna: bool = True,
    ) -> groupby.DataFrameGroupBy:
        if as_index and not isinstance(by, str):
            raise ValueError(
                "Set as_index=False if grouping by list of values. Mutli-index not"
                + "yet supported"
            )
        by_col_ids = self._sql_names(by)
        return groupby.DataFrameGroupBy(
            self._block,
            by_col_ids,
            dropna=dropna,
            as_index=as_index,
        )

    def abs(self) -> DataFrame:
        return self._apply_to_rows(ops.abs_op)

    def isna(self) -> DataFrame:
        return self._apply_to_rows(ops.isnull_op)

    isnull = isna

    def notna(self) -> DataFrame:
        return self._apply_to_rows(ops.notnull_op)

    notnull = notna

    def cumsum(self):
        is_numeric_types = [
            (dtype in bigframes.dtypes.NUMERIC_BIGFRAMES_TYPES)
            for _, dtype in self.dtypes.items()
        ]
        if not all(is_numeric_types):
            raise ValueError("All values must be numeric to apply cumsum.")
        return self._apply_window_op(
            agg_ops.sum_op,
            bigframes.core.WindowSpec(following=0),
        )

    def cumprod(self) -> DataFrame:
        is_numeric_types = [
            (dtype in bigframes.dtypes.NUMERIC_BIGFRAMES_TYPES)
            for _, dtype in self.dtypes.items()
        ]
        if not all(is_numeric_types):
            raise ValueError("All values must be numeric to apply cumsum.")
        return self._apply_window_op(
            agg_ops.product_op,
            bigframes.core.WindowSpec(following=0),
        )

    def cummin(self) -> DataFrame:
        return self._apply_window_op(
            agg_ops.min_op,
            bigframes.core.WindowSpec(following=0),
        )

    def cummax(self) -> DataFrame:
        return self._apply_window_op(
            agg_ops.max_op,
            bigframes.core.WindowSpec(following=0),
        )

    def shift(self, periods: int = 1) -> DataFrame:
        window = bigframes.core.WindowSpec(
            preceding=periods if periods > 0 else None,
            following=-periods if periods < 0 else None,
        )
        return self._apply_window_op(agg_ops.ShiftOp(periods), window)

    def _apply_window_op(
        self,
        op: agg_ops.WindowOp,
        window_spec: bigframes.core.WindowSpec,
    ):
        block = self._block.multi_apply_window_op(
            self._block.value_columns,
            op,
            window_spec=window_spec,
        )
        return DataFrame(block)

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        *,
        random_state: Optional[int] = None,
    ) -> DataFrame:
        if n is not None and frac is not None:
            raise ValueError("Only one of 'n' or 'frac' parameter can be specified.")

        ns = (n,) if n is not None else ()
        fracs = (frac,) if frac is not None else ()

        return self._split(ns=ns, fracs=fracs, random_state=random_state)[0]

    def _split(
        self,
        ns: Iterable[int] = (),
        fracs: Iterable[float] = (),
        *,
        random_state: Optional[int] = None,
    ) -> List[DataFrame]:
        """Internal function to support splitting DF to multiple parts along index axis.

        At most one of ns and fracs can be passed in. If neither, default to ns = (1,).
        Return a list of sampled DataFrames.
        """
        if ns and fracs:
            raise ValueError("Only one of 'ns' or 'fracs' parameter must be specified.")

        block = self._block
        if not ns and not fracs:
            ns = (1,)

        if ns:
            sample_sizes = ns
        else:
            total_rows = block.shape()[0]
            # Round to nearest integer. "round half to even" rule applies.
            # At least to be 1.
            sample_sizes = [round(frac * total_rows) or 1 for frac in fracs]

        # Set random_state if it is not provided
        if random_state is None:
            random_state = random.randint(-(2**30), 2**30)

        # Create a new column with random_state value.
        block, random_state_col = block.create_constant(random_state)

        # Create an ordering col and a new sum col which is ordering+random_state.
        block, ordering_col = block.promote_offsets()
        block, sum_col = block.apply_binary_op(
            ordering_col, random_state_col, ops.add_op
        )

        # Apply hash method to sum col and order by it.
        block, string_sum_col = block.apply_unary_op(
            sum_col, ops.AsTypeOp("string[pyarrow]")
        )
        block, hash_string_sum_col = block.apply_unary_op(string_sum_col, ops.hash_op)
        block = block.order_by([order.OrderingColumnReference(hash_string_sum_col)])

        drop_cols = [
            random_state_col,
            ordering_col,
            sum_col,
            string_sum_col,
            hash_string_sum_col,
        ]
        block = block.drop_columns(drop_cols)
        df = DataFrame(block)

        intervals = []
        cur = 0
        for sample_size in sample_sizes:
            intervals.append((cur, cur + sample_size))
            cur += sample_size

        # DF.iloc[slice] always returns DF.
        return [
            typing.cast(DataFrame, df.iloc[lower:upper]) for lower, upper in intervals
        ]

    def to_pandas(self) -> pd.DataFrame:
        """Writes DataFrame to Pandas DataFrame."""
        # TODO(chelsealin): Support block parameters.
        # TODO(chelsealin): Add to_pandas_batches() API.
        return self.compute()

    def to_csv(self, path_or_buf: str, *, index: bool = True) -> None:
        # TODO(swast): Can we support partition columns argument?
        # TODO(chelsealin): Support local file paths.
        # TODO(swast): Some warning that wildcard is recommended for large
        # query results? See:
        # https://cloud.google.com/bigquery/docs/exporting-data#limit_the_exported_file_size
        if not path_or_buf.startswith("gs://"):
            raise NotImplementedError(
                "Only Google Cloud Storage (gs://...) paths are supported."
            )

        source_table = self._execute_query(index=index)
        job_config = bigquery.ExtractJobConfig(
            destination_format=bigquery.DestinationFormat.CSV
        )
        extract_job = self._block.expr._session.bqclient.extract_table(
            source_table, destination_uris=[path_or_buf], job_config=job_config
        )
        extract_job.result()  # Wait for extract job to finish

    def to_json(
        self,
        path_or_buf: str,
        orient: Literal[
            "split", "records", "index", "columns", "values", "table"
        ] = "columns",
        *,
        lines: bool = False,
        index: bool = True,
    ) -> None:
        # TODO(swast): Can we support partition columns argument?
        # TODO(chelsealin): Support local file paths.
        # TODO(swast): Some warning that wildcard is recommended for large
        # query results? See:
        # https://cloud.google.com/bigquery/docs/exporting-data#limit_the_exported_file_size
        if not path_or_buf.startswith("gs://"):
            raise NotImplementedError(
                "Only Google Cloud Storage (gs://...) paths are supported."
            )

        if lines is True and orient != "records":
            raise ValueError(
                "'lines' keyword is only valid when 'orient' is 'records'."
            )

        # TODO(ashleyxu) Support lines=False for small tables with arrays and TO_JSON_STRING.
        # See: https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions#to_json_string
        if lines is False:
            raise NotImplementedError(
                "Only newline delimited JSON format is supported."
            )

        source_table = self._execute_query(index=index)
        job_config = bigquery.ExtractJobConfig(
            destination_format=bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
        )
        extract_job = self._block.expr._session.bqclient.extract_table(
            source_table, destination_uris=[path_or_buf], job_config=job_config
        )
        extract_job.result()  # Wait for extract job to finish

    def to_gbq(
        self,
        destination_table: str,
        *,
        if_exists: Optional[Literal["fail", "replace", "append"]] = "fail",
        index: bool = True,
    ) -> None:
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

        self._execute_query(index=index, job_config=job_config)

    def to_numpy(
        self, dtype=None, copy=False, na_value=None, **kwargs
    ) -> numpy.ndarray:
        return self.compute().to_numpy(dtype, copy, na_value, **kwargs)

    __array__ = to_numpy

    def to_parquet(self, path: str, *, index: bool = True) -> None:
        # TODO(swast): Can we support partition columns argument?
        # TODO(chelsealin): Support local file paths.
        # TODO(swast): Some warning that wildcard is recommended for large
        # query results? See:
        # https://cloud.google.com/bigquery/docs/exporting-data#limit_the_exported_file_size
        if not path.startswith("gs://"):
            raise NotImplementedError(
                "Only Google Cloud Storage (gs://...) paths are supported."
            )

        source_table = self._execute_query(index=index)
        job_config = bigquery.ExtractJobConfig(
            destination_format=bigquery.DestinationFormat.PARQUET
        )
        extract_job = self._block.expr._session.bqclient.extract_table(
            source_table, destination_uris=[path], job_config=job_config
        )
        extract_job.result()  # Wait for extract job to finish

    def _apply_to_rows(self, operation: ops.UnaryOp):
        block = self._block.multi_apply_unary_op(self._block.value_columns, operation)
        return DataFrame(block)

    def _execute_query(
        self, index: bool, job_config: Optional[bigquery.job.QueryJobConfig] = None
    ):
        """Executes a query job presenting this dataframe and returns the destination
        table."""
        expr = self._block.expr
        session = expr._session
        columns = list(self._block.value_columns)
        column_labels = list(self._block.column_labels)
        # This code drops unnamed indexes to keep consistent with the behavior of
        # most pandas write APIs. The exception is `pandas.to_csv`, which keeps
        # unnamed indexes as `Unnamed: 0`.
        # TODO(chelsealin): check if works for multiple indexes.
        if index and self.index.name is not None:
            columns.extend(self._block.index_columns)
            # TODO(swast): support MultiIndex
            column_labels.append(self.index.name)
        # TODO(chelsealin): normalize the file formats if we needs, such as arbitrary
        # unicode for column labels.
        value_columns = (expr.get_column(column_name) for column_name in columns)
        expr = expr.projection(value_columns)

        # Make columns in SQL reflect _labels_ not _ids_. Note: This may use
        # the arbitrary unicode column labels feature in BigQuery, which is
        # currently (June 2023) in preview.
        # TODO(swast): Handle duplicate and NULL labels.
        ibis_expr = expr.to_ibis_expr()
        renamed_columns = [
            ibis_expr[col_id].name(col_label)
            for col_id, col_label in zip(columns, column_labels)
        ]
        ibis_expr = ibis_expr.select(*renamed_columns)
        sql = session.ibis_client.compile(ibis_expr)  # type: ignore
        query_job: bigquery.QueryJob = session.bqclient.query(
            sql, job_config=job_config  # type: ignore
        )
        query_job.result()  # Wait for query to finish.
        query_job.reload()  # Download latest job metadata.
        return query_job.destination

    def map(self, func, na_action: Optional[str] = None) -> DataFrame:
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if na_action not in {None, "ignore"}:
            raise ValueError(f"na_action={na_action} not supported")

        # TODO(shobs): Support **kwargs

        return self._apply_to_rows(
            ops.RemoteFunctionOp(func, apply_on_null=(na_action is None))
        )

    applymap = map

    def _slice(
        self,
        start: typing.Optional[int] = None,
        stop: typing.Optional[int] = None,
        step: typing.Optional[int] = None,
    ) -> bigframes.DataFrame:
        block = self._block.slice(start=start, stop=stop, step=step)
        return bigframes.DataFrame(block)

    def _set_block(self, block: blocks.Block):
        self._block = block

    def _get_block(self) -> blocks.Block:
        return self._block


def _is_list_like(obj: typing.Any) -> typing_extensions.TypeGuard[typing.Sequence]:
    return pd.api.types.is_list_like(obj)


def _is_dict_like(obj: typing.Any) -> typing_extensions.TypeGuard[typing.Mapping]:
    return pd.api.types.is_dict_like(obj)
