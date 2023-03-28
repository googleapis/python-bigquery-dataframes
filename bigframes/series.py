"""Series is a 1 dimensional data structure."""

from __future__ import annotations

import typing
from typing import Optional

import ibis
import ibis.common.exceptions
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types
import pandas
import pandas.core.dtypes.common
import typing_extensions

import bigframes.aggregations as agg_ops
import bigframes.core
import bigframes.core.blocks as blocks
import bigframes.core.indexes.implicitjoiner
import bigframes.core.indexes.index
import bigframes.dtypes
import bigframes.indexers
import bigframes.operations as ops
import bigframes.scalar
import bigframes.view_windows


class Series:
    """A 1D data structure, representing data and deferred computation.

    .. warning::
        This constructor is **private**. Use a public method such as
        ``DataFrame[column_name]`` to construct a Series.
    """

    def __init__(
        self,
        block: blocks.Block,
        value_column: str,
        *,
        name: Optional[str] = None,
    ):
        self._block = block
        self._value_column = value_column
        self._name = name

    @property
    def _value(self) -> ibis_types.Value:
        """Private property to get Ibis expression for the value column."""
        return self._viewed_block.expr.get_column(self._value_column)

    @property
    def dtype(self) -> bigframes.dtypes.BigFramesDtype:
        """Returns the dtype of the Series"""
        return bigframes.dtypes.ibis_dtype_to_bigframes_dtype(
            typing.cast(bigframes.dtypes.IbisDtype, self._value.type())
        )

    @property
    def index(self) -> bigframes.core.indexes.implicitjoiner.ImplicitJoiner:
        return self._viewed_block.index

    @property
    def loc(self) -> bigframes.indexers.LocSeriesIndexer:
        """Set items by index label.

        No get or slice support currently supported.
        """
        return bigframes.indexers.LocSeriesIndexer(self)

    @property
    def name(self) -> Optional[str]:
        # TODO(swast): Introduce a level of indirection over Ibis to allow for
        # more accurate pandas behavior (such as allowing for unnamed or
        # non-uniquely named objects) without breaking SQL.
        return self._name

    @property
    def _viewed_block(self) -> blocks.Block:
        """Gets a copy of block after any views have been applied. Mutations to this copy do not affect any existing series/dataframes."""
        return self._block.copy()

    def __repr__(self) -> str:
        """Converts a Series to a string."""
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        # TODO(swast): Avoid downloading the whole series by using job
        # metadata, like we do with DataFrame.
        preview = self.compute()
        return repr(preview)

    def _to_ibis_expr(self):
        """Creates an Ibis table expression representing the Series."""
        expr = self._viewed_block.expr.projection([self._value])
        ibis_expr = expr.to_ibis_expr()[self._value_column]
        if self._name:
            return ibis_expr.name(self._name)
        return ibis_expr

    def compute(self) -> pandas.Series:
        """Executes deferred operations and downloads the results."""
        df = self._viewed_block.compute((self._value_column,))
        series = df[self._value_column]
        series.name = self._name
        return series

    def cumsum(self) -> Series:
        window = ibis.cumulative_window(
            order_by=self._block.expr.ordering,
            group_by=self._block.expr.reduced_predicate,
        )

        def cumsum_op(x: ibis_types.Value):
            # Emulate pandas by return NA for every value after first NA
            # Might be worth diverging here to make cumsum more useful
            x_numeric = typing.cast(ibis_types.NumericColumn, x)
            have_seen_na = (
                typing.cast(ibis_types.BooleanColumn, x_numeric.isnull())
                .any()
                .over(window)
            )
            cumsum = x_numeric.sum().over(window)
            pandas_style_cumsum = (
                ibis.case().when(have_seen_na, ibis.NA).else_(cumsum).end()
            )
            return pandas_style_cumsum

        return self._apply_unary_op(cumsum_op)

    def cummax(self) -> Series:
        window = ibis.cumulative_window(
            order_by=self._block.expr.ordering,
            group_by=self._block.expr.reduced_predicate,
        )

        def cummax_op(x: ibis_types.Value):
            cummax = typing.cast(ibis_types.NumericColumn, x).max().over(window)
            pandas_style_cummax = (
                ibis.case().when(x.isnull(), ibis.NA).else_(cummax).end()
            )
            return pandas_style_cummax

        return self._apply_unary_op(cummax_op)

    def cummin(self) -> Series:
        window = ibis.cumulative_window(
            order_by=self._block.expr.ordering,
            group_by=self._block.expr.reduced_predicate,
        )

        def cummin_op(x: ibis_types.Value):
            cummin = typing.cast(ibis_types.NumericColumn, x).min().over(window)
            pandas_style_cummin = (
                ibis.case().when(x.isnull(), ibis.NA).else_(cummin).end()
            )
            return pandas_style_cummin

        return self._apply_unary_op(cummin_op)

    def fillna(self, value) -> "Series":
        """Fills NULL values."""

        def fillna_op(x: ibis_types.Value):
            return x.fillna(value)

        return self._apply_unary_op(fillna_op)

    def head(self, n: int = 5) -> Series:
        """Limits Series to a specific number of rows."""
        return ViewSeries(
            self._block,
            self._value_column,
            bigframes.view_windows.SliceViewWindow(0, n),
            name=self._name,
        )

    def len(self) -> "Series":
        """Compute the length of each string."""

        return self._apply_unary_op(ops.len_op)

    def isnull(self) -> "Series":
        """Returns a boolean same-sized object indicating if the values are NULL/missing."""

        return self._apply_unary_op(ops.isnull_op)

    def notnull(self) -> "Series":
        """Returns a boolean same-sized object indicating if the values are not NULL/missing."""

        return self._apply_unary_op(ops.notnull_op)

    notna = notnull

    def lower(self) -> "Series":
        """Convert strings in the Series to lowercase."""

        def lower_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x).lower()

        return self._apply_unary_op(lower_op)

    def upper(self) -> "Series":
        """Convert strings in the Series to uppercase."""

        def upper_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x).upper()

        return self._apply_unary_op(upper_op)

    def strip(self) -> "Series":
        """Removes whitespace characters from the beginning and end of each string in the Series."""

        def strip_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x).strip()

        return self._apply_unary_op(strip_op)

    def __add__(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.add_op)

    __radd__ = __add__

    def __sub__(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.sub_op)

    def __rsub__(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.reverse(ops.sub_op))

    def __mul__(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.mul_op)

    __rmul__ = __mul__

    def __truediv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.div_op, ibis_dtypes.float)

    def __rtruediv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.reverse(ops.div_op), ibis_dtypes.float)

    def __floordiv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.floordiv_op, ibis_dtypes.int64)

    def __rfloordiv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(
            other, ops.reverse(ops.floordiv_op), ibis_dtypes.int64
        )

    def __lt__(self, other) -> Series:  # type: ignore
        return self._apply_binary_op(other, ops.lt_op, ibis_dtypes.bool)

    def __le__(self, other) -> Series:  # type: ignore
        return self._apply_binary_op(other, ops.le_op, ibis_dtypes.bool)

    def __gt__(self, other) -> Series:  # type: ignore
        return self._apply_binary_op(other, ops.gt_op, ibis_dtypes.bool)

    def __ge__(self, other) -> Series:  # type: ignore
        return self._apply_binary_op(other, ops.ge_op, ibis_dtypes.bool)

    def __mod__(self, other) -> Series:  # type: ignore
        return self._apply_binary_op(other, ops.mod_op, ibis_dtypes.int64)

    def __rmod__(self, other) -> Series:  # type: ignore
        return self._apply_binary_op(other, ops.reverse(ops.mod_op), ibis_dtypes.int64)

    def abs(self) -> "Series":
        """Calculate absolute value of numbers in the Series."""
        return self._apply_unary_op(ops.abs_op)

    def reverse(self) -> "Series":
        """Reverse strings in the Series."""
        return self._apply_unary_op(ops.reverse_op)

    def round(self, decimals=0) -> "Series":
        """Round each value in a Series to the given number of decimals."""

        def round_op(x: ibis_types.Value):
            return typing.cast(ibis_types.NumericValue, x).round(digits=decimals)

        return self._apply_unary_op(round_op)

    def all(self) -> bigframes.scalar.Scalar:
        """Returns true if and only if all elements are True. Nulls are ignored"""
        return self._apply_aggregation(agg_ops.all_op)

    def any(self) -> bigframes.scalar.Scalar:
        """Returns true if and only if at least one element is True. Nulls are ignored"""
        return self._apply_aggregation(agg_ops.any_op)

    def count(self) -> bigframes.scalar.Scalar:
        """Counts the number of values in the series. Ignores null/nan."""
        return self._apply_aggregation(agg_ops.count_op)

    def max(self) -> bigframes.scalar.Scalar:
        """Return the maximum values over the requested axis."""
        return self._apply_aggregation(agg_ops.max_op)

    def min(self) -> bigframes.scalar.Scalar:
        """Return the maximum values over the requested axis."""
        return self._apply_aggregation(agg_ops.min_op)

    def mean(self) -> bigframes.scalar.Scalar:
        """Finds the mean of the numeric values in the series. Ignores null/nan.

        Note: pandas and BigFrames may not perform floating point operations in
        exactly the same order. Expect some floating point wobble. When
        comparing computed results, use a method such as :func:`math.isclose`
        or :func:`numpy.isclose` to account for this.

        Returns:
            A BigFrames Scalar so that this can be used in other expressions.
            To get the numeric result call
            :func:`~bigframes.scalar.Scalar.compute()`.
        """
        return self._apply_aggregation(agg_ops.mean_op)

    def sum(self) -> bigframes.scalar.Scalar:
        """Sums the numeric values in the series. Ignores null/nan.

        Note: pandas and BigFrames may not perform floating point operations in
        exactly the same order. Expect some floating point wobble. When
        comparing computed results, use a method such as :func:`math.isclose`
        or :func:`numpy.isclose` to account for this.

        Returns:
            A BigFrames Scalar so that this can be used in other expressions.
            To get the numeric result call
            :func:`~bigframes.scalar.Scalar.compute()`.
        """
        return self._apply_aggregation(agg_ops.sum_op)

    def slice(self, start=None, stop=None) -> "Series":
        """Slice substrings from each element in the Series."""

        def slice_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x)[start:stop]

        return self._apply_unary_op(slice_op)

    def __eq__(self, other: object) -> Series:  # type: ignore
        """Element-wise equals between the series and another series or literal."""
        return self.eq(other)

    def __ne__(self, other: object) -> Series:  # type: ignore
        """Element-wise not-equals between the series and another series or literal."""
        return self.ne(other)

    def __invert__(self) -> Series:
        """Element-wise logical negation. Does not handle null or nan values."""

        return self._apply_unary_op(ops.invert_op)

    def eq(self, other: object) -> Series:
        """
        Element-wise equals between the series and another series or literal.
        None and NaN are not equal to themselves.
        This is inconsitent with Pandas eq behavior with None: https://github.com/pandas-dev/pandas/issues/20442
        """
        # TODO: enforce stricter alignment
        (left, right, index) = self._align(other)
        block = self._viewed_block
        block.index = index
        block.replace_value_columns(
            [
                (left == right).name(self._value_column),
            ]
        )
        name = self._name
        if isinstance(other, Series) and other.name != self.name:
            name = None
        return Series(
            block,
            self._value_column,
            name=name,
        )

    def ne(self, other: object) -> Series:
        """
        Element-wise not-equals between the series and another series or literal.
        None and NaN are not equal to themselves.
        This is inconsitent with Pandas eq behavior with None: https://github.com/pandas-dev/pandas/issues/20442
        """
        # TODO: enforce stricter alignment
        (left, right, index) = self._align(other)
        block = self._viewed_block
        block.index = index
        block.replace_value_columns(
            [
                (left != right).name(self._value_column),
            ]
        )
        name = self._name
        if isinstance(other, Series) and other.name != self.name:
            name = None
        return Series(
            block,
            self._value_column,
            name=name,
        )

    def __getitem__(self, indexer: Series):
        """Get items using boolean series indexer."""
        # TODO: enforce stricter alignment, should fail if indexer is missing any keys.
        (left, right, index) = self._align(indexer, "left")
        block = self._viewed_block
        block.index = index
        block.replace_value_columns(
            [
                left,
            ]
        )
        filtered_expr = block.expr.filter((right == ibis.literal(True)))
        block.expr = filtered_expr
        return Series(block, self._value_column, name=self._name)

    def _align(self, other: typing.Any, how="outer") -> tuple[ibis_types.Value, ibis_types.Value, bigframes.core.indexes.implicitjoiner.ImplicitJoiner]:  # type: ignore
        """Aligns the series value with other scalar or series object. Returns new left value, right value and joined tabled expression."""
        # TODO: Support deferred scalar
        if isinstance(other, Series):
            combined_index, (
                get_column_left,
                get_column_right,
            ) = self.index.join(other.index, how=how)
            left_value = get_column_left(self._value_column)
            right_value = get_column_right(other._value.get_name())
            return (left_value, right_value, combined_index)
        elif isinstance(other, bigframes.scalar.Scalar):
            # TODO(tbereron): support deferred scalars.
            raise ValueError("Deferred scalar not yet supported for binary operations.")
        as_literal_value = _interpret_as_ibis_literal(other)
        if as_literal_value is not None:
            combined_index = self.index
            left_value = self._value
            right_value = as_literal_value
            return (left_value, right_value, combined_index)
        else:
            return NotImplemented

    def _apply_aggregation(
        self,
        op: typing.Callable[[ibis_types.Column], ibis_types.Scalar],
    ) -> bigframes.scalar.Scalar:
        return bigframes.scalar.Scalar(op(self[self.notnull()]._to_ibis_expr()))

    def _apply_unary_op(
        self,
        op: typing.Callable[[ibis_types.Value], ibis_types.Value],
    ) -> Series:
        """Applies a binary operator to the series and other."""
        block = self._viewed_block
        block.replace_value_columns([op(self._value).name(self._value_column)])
        return Series(
            block,
            self._value_column,
            name=self._name,
        )

    def _apply_binary_op(
        self,
        other: typing.Any,
        op: typing.Callable[[ibis_types.Value, ibis_types.Value], ibis_types.Value],
        expected_dtype: typing.Optional[ibis_dtypes.DataType] = None,
    ) -> Series:
        """Applies a binary operator to the series and other."""
        (left, right, index) = self._align(other)
        block = self._viewed_block
        block.index = index
        if not isinstance(right, ibis_types.NullScalar):
            result_expr = op(left, right).name(self._value_column)
        else:
            # Cannot do sql op with null literal, so just replace expression directly with default
            # value
            output_dtype = expected_dtype if expected_dtype else left.type()
            default_value = ibis_types.null().cast(output_dtype)
            result_expr = default_value.name(self._value_column)
        block.replace_value_columns([result_expr])

        name = self._name
        if isinstance(other, Series) and other.name != self.name:
            name = None

        return Series(
            block,
            self._value_column,
            name=name,
        )

    def find(self, sub, start=None, end=None) -> "Series":
        """Return the position of the first occurence of substring."""

        def find_op(x: ibis_types.Value):
            return typing.cast(ibis_types.StringValue, x).find(sub, start, end)

        return self._apply_unary_op(find_op)

    def groupby(
        self,
        by: typing.Optional[Series] = None,
        axis=None,
        level: typing.Optional[
            int | str | typing.Sequence[int] | typing.Sequence[str]
        ] = None,
        as_index=True,
        *,
        dropna: bool = True,
    ):
        """Group the series by a given list of column labels. Only supports grouping by values from another aligned Series."""
        if (by is not None) and (level is not None):
            raise ValueError("Do not specify both 'by' and 'level'")
        if not as_index:
            raise ValueError("as_index=False only valid with DataFrame")
        if axis:
            raise ValueError("No axis named {} for object type Series".format(level))
        if by is not None:
            return self._groupby_series(by, dropna)
        if level is not None:
            return self._groupby_level(level, dropna)
        else:
            raise TypeError("You have to supply one of 'by' and 'level'")

    def _groupby_level(
        self,
        level: int | str | typing.Sequence[int] | typing.Sequence[str],
        dropna: bool = True,
    ):
        # TODO(tbergeron): Add multi-index groupby when that feature is implemented.
        if isinstance(level, int) and (level > 0 or level < -1):
            raise ValueError("level > 0 or level < -1 only valid with MultiIndex")
        if isinstance(level, str) and level != self.index.name:
            raise ValueError("level name {} is not the name of the index".format(level))
        if _is_list_like(level):
            if len(level) > 1:
                raise ValueError("multiple levels only valid with MultiIndex")
            if len(level) == 0:
                raise ValueError("No group keys passed!")
            return self._groupby_level(level[0], dropna)
        if level and not self._block.index_columns:
            raise ValueError(
                "groupby level requires and explicit index on the dataframe"
            )
        block = self._viewed_block
        # If all validations passed, must be grouping on the single-level index
        group_key = self._block.index_columns[0]
        key = block._expr.get_column(group_key)
        value = self._value
        if dropna:
            filtered_expr = block.expr.filter((key.notnull()))
            block.expr = filtered_expr
        return SeriesGroupyBy(block, value.get_name(), key.get_name())

    def _groupby_series(
        self,
        by: Series,
        dropna: bool = True,
    ):
        block = self._viewed_block
        if dropna:
            by = by[by.notna()]
        (value, key, index) = self[self.notna()]._align(
            by, "inner" if dropna else "left"
        )
        block = self._viewed_block
        block.index = index
        block.replace_value_columns([key, value])
        return SeriesGroupyBy(block, value.get_name(), key.get_name())

    def apply(self, func) -> Series:
        """Returns a series with a user defined function applied."""
        # TODO(shobs, b/274645634): Support convert_dtype, args, **kwargs
        return self._apply_unary_op(func)


class ViewSeries(Series):
    """
    A series representing a view of underlying data. May filter and/or reorder underlying data.

    Mutations on cells in the view series propogate back to the parent series.
    """

    def __init__(
        self,
        block: blocks.Block,
        value_column: str,
        view_window: bigframes.view_windows.SliceViewWindow,
        **kwargs,
    ):
        super().__init__(block, value_column, **kwargs)
        self._view_window = view_window

    @property
    def _viewed_block(self) -> blocks.Block:
        """Gets a copy of block after any views have been applied. Mutations to this copy do not affect any existing series/dataframes."""
        return self._view_window.apply_window(self._block)

    def head(self, n: int = 5) -> Series:
        # TODO(tbergeron): Implement stacked view windows.
        raise NotImplementedError("Recursively applying windows not yet supported.")


class SeriesGroupyBy:
    """Represents a deferred series with a grouping expression."""

    def __init__(self, block: blocks.Block, value_column: str, by: str):
        # TODO(tbergeron): Support more group-by expression types
        # TODO(tbergeron): Implement as a view
        self._block = block
        self._value_column = value_column
        self._by = by

    @property
    def value(self):
        return self._block.expr.get_column(self._value_column)

    def all(self) -> Series:
        """Returns true if and only if all elements are True. Nulls are ignored"""
        return self._aggregate(agg_ops.all_op(self.value).name(self._value_column))

    def any(self) -> Series:
        """Returns true if and only if at least one element is True. Nulls are ignored"""
        return self._aggregate(agg_ops.any_op(self.value).name(self._value_column))

    def count(self) -> Series:
        """Counts the number of elements in each group. Ignores null/nan."""
        return self._aggregate(agg_ops.count_op(self.value).name(self._value_column))

    def sum(self) -> Series:
        """Sums the numeric values for each group in the series. Ignores null/nan."""
        return self._aggregate(agg_ops.sum_op(self.value).name(self._value_column))

    def mean(self) -> Series:
        """Finds the mean of the numeric values for each group in the series. Ignores null/nan."""
        return self._aggregate(agg_ops.mean_op(self.value).name(self._value_column))

    def _aggregate(self, metric: ibis_types.Scalar) -> Series:
        group_expr = bigframes.core.BigFramesGroupByExpr(self._block.expr, self._by)
        block = blocks.Block(group_expr.aggregate((metric,)), (self._by,))
        return Series(block, metric.get_name(), name=metric.get_name())


def _interpret_as_ibis_literal(value: typing.Any) -> typing.Optional[ibis_types.Value]:
    if isinstance(value, Series) or isinstance(value, pandas.Series):
        return None
    if pandas.isna(value):
        # TODO(tbergeron): Ensure correct handling of NaN - maybe not map to Null
        return ibis_types.null()
    try:
        return ibis_types.literal(value)
    except ibis.common.exceptions.IbisTypeError:
        # Value cannot be converted into literal.
        return None


def _is_list_like(obj: typing.Any) -> typing_extensions.TypeGuard[typing.Sequence]:
    return pandas.core.dtypes.common.is_list_like(obj)
