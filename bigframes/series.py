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
import bigframes.operations.base
import bigframes.operations.strings as strings
import bigframes.scalar


class Series(bigframes.operations.base.SeriesMethods):
    """A 1D data structure, representing data and deferred computation.

    .. warning::
        This constructor is **private**. Use a public method such as
        ``DataFrame[column_name]`` to construct a Series.
    """

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
    def iloc(self) -> bigframes.indexers.IlocSeriesIndexer:
        """Get items by slice.

        No set operations currently supported.
        """
        return bigframes.indexers.IlocSeriesIndexer(self)

    @property
    def name(self) -> Optional[str]:
        # TODO(swast): Introduce a level of indirection over Ibis to allow for
        # more accurate pandas behavior (such as allowing for unnamed or
        # non-uniquely named objects) without breaking SQL.
        return self._name

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

    def between(self, left, right, inclusive="both"):
        if inclusive not in ["both", "neither", "left", "right"]:
            raise ValueError(
                "Must set 'inclusive' to one of 'both', 'neither', 'left', or 'right'"
            )
        left_op = ops.ge_op if (inclusive in ["left", "both"]) else ops.gt_op
        right_op = ops.le_op if (inclusive in ["right", "both"]) else ops.lt_op
        return self._apply_binary_op(left, left_op).__and__(
            self._apply_binary_op(right, right_op)
        )

    def cumsum(self) -> Series:
        return self._apply_cumulative_aggregate(agg_ops.sum_op)

    def cummax(self) -> Series:
        return self._apply_cumulative_aggregate(agg_ops.max_op)

    def cummin(self) -> Series:
        return self._apply_cumulative_aggregate(agg_ops.min_op)

    def fillna(self, value) -> "Series":
        """Fills NULL values."""

        def fillna_op(x: ibis_types.Value):
            return x.fillna(value)

        return self._apply_unary_op(fillna_op)

    def head(self, n: int = 5) -> Series:
        """Limits Series to a specific number of rows."""
        return typing.cast(Series, self.iloc[0:n])

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

    def __and__(self, other: bool | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.and_op, short_nulls=False)

    __rand__ = __and__

    def __or__(self, other: bool | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.or_op, short_nulls=False)

    __ror__ = __or__

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

    def __matmul__(self, other: Series):
        return (self * other).sum()

    dot = __matmul__

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

    def prod(self) -> bigframes.scalar.Scalar:
        """Finds the product of the numeric values for each group in the series. Ignores null/nan.

        Note: pandas and BigFrames may not perform floating point operations in
        exactly the same order. Expect some floating point wobble. When
        comparing computed results, use a method such as :func:`math.isclose`
        or :func:`numpy.isclose` to account for this.

        Returns:
            A BigFrames Scalar so that this can be used in other expressions.
            To get the numeric result call
            :func:`~bigframes.scalar.Scalar.compute()`.
        """
        return self._apply_aggregation(agg_ops.product_op)

    product = prod

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

    def where(self, cond, other=None):
        return self._apply_ternary_op(cond, other or pandas.NA, ops.where_op)

    def clip(self, lower, upper):
        return self._apply_ternary_op(lower, upper, ops.clip_op)

    def __getitem__(self, indexer: Series):
        """Get items using boolean series indexer."""
        # TODO: enforce stricter alignment, should fail if indexer is missing any keys.
        (left, right, index) = self._align(indexer, "left")
        block = self._viewed_block
        block.index = index
        block.replace_value_columns(
            [
                left.name(self._value_column),
            ]
        )
        filtered_expr = block.expr.filter((right == ibis.literal(True)))
        block.expr = filtered_expr
        return Series(block, self._value_column, name=self._name)

    def _align(self, other: typing.Any, how="outer") -> tuple[ibis_types.Value, ibis_types.Value, bigframes.core.indexes.implicitjoiner.ImplicitJoiner]:  # type: ignore
        """Aligns the series value with other scalar or series object. Returns new left value, right value and joined tabled expression."""
        values, index = self._align_n(
            [
                other,
            ],
            how,
        )
        return (values[0], values[1], index)

    def _align3(self, other1: typing.Any, other2: typing.Any, how="left") -> tuple[ibis_types.Value, ibis_types.Value, ibis_types.Value, bigframes.core.indexes.implicitjoiner.ImplicitJoiner]:  # type: ignore
        """Aligns the series value with 2 other scalars or series objects. Returns new values and joined tabled expression."""
        values, index = self._align_n([other1, other2], how)
        return (values[0], values[1], values[2], index)

    def _align_n(
        self, others: typing.Sequence[typing.Any], how="outer"
    ) -> tuple[
        typing.Sequence[ibis_types.Value],
        bigframes.core.indexes.implicitjoiner.ImplicitJoiner,
    ]:
        values = [self._value]
        index = self.index
        for other in others:
            as_literal = _interpret_as_ibis_literal(other)
            if isinstance(other, Series):
                combined_index, (
                    get_column_left,
                    get_column_right,
                ) = index.join(other.index, how=how)
                values = [
                    *[get_column_left(value.get_name()) for value in values],
                    get_column_right(other._value_column),
                ]
                index = combined_index
            elif isinstance(other, bigframes.scalar.Scalar):
                # TODO(tbereron): support deferred scalars.
                raise ValueError(
                    "Deferred scalar not yet supported for binary operations."
                )
            elif as_literal is not None:
                values = [*values, as_literal]
            else:
                raise NotImplementedError(f"Unsupported operand of type {type(other)}")
        return (values, index)

    def _apply_aggregation(
        self,
        op: typing.Callable[[ibis_types.Column], ibis_types.Value],
    ) -> bigframes.scalar.Scalar:
        aggregation_result = typing.cast(
            ibis_types.Scalar, op(self[self.notnull()]._to_ibis_expr())
        )
        return bigframes.scalar.Scalar(aggregation_result)

    def _apply_cumulative_aggregate(self, op: agg_ops.AggregateOp):
        expr = self._block.expr
        # TODO: Push this down into expression objects
        window = ibis.cumulative_window(
            order_by=expr.ordering,
            group_by=expr.reduced_predicate,
        )

        def cumulative_op(x: ibis_types.Column):
            cumop = op.with_window(window)(x)
            # Pandas cumulative ops will result in NA if input row was NA, instead of current cumulative value
            pandas_style_cumop = (
                ibis.case().when(x.isnull(), ibis.NA).else_(cumop).end()
            )
            return pandas_style_cumop

        block = blocks.Block(expr, self._block.index_columns)
        block.index.name = self.index.name
        block.replace_value_columns(
            [
                cumulative_op(
                    # TODO(swast): When we assign literals / scalars, we might not
                    # have a true Column. Do we need to check this before trying to
                    # aggregate such a column?
                    typing.cast(ibis_types.Column, expr.get_column(self._value_column))
                ).name(self._value_column)
            ]
        )
        return Series(
            block,
            self._value_column,
            name=self._value_column,
        )

    def _apply_binary_op(
        self,
        other: typing.Any,
        op: ops.BinaryOp,
        expected_dtype: typing.Optional[ibis_dtypes.DataType] = None,
        short_nulls=True,
    ) -> Series:
        """Applies a binary operator to the series and other."""
        (left, right, index) = self._align(other)

        block = blocks.Block(index._expr)
        block.index = index
        if isinstance(right, ibis_types.NullScalar) and short_nulls:
            # Cannot do sql op with null literal, so just replace expression directly with default
            # value
            output_dtype = expected_dtype if expected_dtype else left.type()
            default_value = ibis_types.null().cast(output_dtype)
            result_expr = default_value.name(self._value_column)
        else:
            result_expr = op(left, right).name(self._value_column)
        block.replace_value_columns([result_expr])

        name = self._name
        if isinstance(other, Series) and other.name != self.name:
            name = None

        return Series(
            block,
            self._value_column,
            name=name,
        )

    def _apply_ternary_op(
        self,
        other: typing.Any,
        other2: typing.Any,
        op: ops.TernaryOp,
    ) -> Series:
        """Applies a binary operator to the series and other."""
        (x, y, z, index) = self._align3(other, other2)

        block = blocks.Block(index._expr)
        block.index = index

        result_expr = op(x, y, z).name(self._value_column)
        block.replace_value_columns([result_expr])
        return Series(
            block,
            self._value_column,
            name=self.name,
        )

    def value_counts(self):
        counts = self.groupby(self).count()
        block = counts._block
        block.expr = block.expr.order_by([counts._value_column], ascending=False)
        return Series(block, counts._value_column, name="count")

    def sort_values(self, axis=0, ascending=True, na_position="last") -> Series:
        """Sort series by values in ascending or descending order."""
        if na_position not in ["first", "last"]:
            raise ValueError("Param na_position must be one of 'first' or 'last'")
        block = self._viewed_block
        block.expr = block.expr.order_by(
            [self._value_column], ascending=ascending, na_last=(na_position == "last")
        )
        return Series(
            block,
            self._value_column,
            name=self.name,
        )

    def sort_index(self, axis=0, *, ascending=True, na_position="last") -> Series:
        """Sort series by index labels in ascending or descending order."""
        # TODO(tbergeron): Support level parameter once multi-index introduced.
        if na_position not in ["first", "last"]:
            raise ValueError("Param na_position must be one of 'first' or 'last'")
        block = self._viewed_block
        block.expr = block.expr.order_by(
            block.index_columns, ascending=ascending, na_last=(na_position == "last")
        )
        return Series(
            block,
            self._value_column,
            name=self.name,
        )

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
        return SeriesGroupyBy(
            block,
            value.get_name(),
            key.get_name(),
            value_name=self.name,
            key_name=self.index.name,
            dropna=dropna,
        )

    def _groupby_series(
        self,
        by: Series,
        dropna: bool = True,
    ):
        block = self._viewed_block
        (value, key, index) = self._align(by, "inner" if dropna else "left")
        block = self._viewed_block
        block.index = index
        return SeriesGroupyBy(
            block,
            value.get_name(),
            key.get_name(),
            value_name=self.name,
            key_name=by.name,
            dropna=dropna,
        )

    def apply(self, func) -> Series:
        """Returns a series with a user defined function applied."""
        # TODO(shobs, b/274645634): Support convert_dtype, args, **kwargs
        return self._apply_unary_op(func)

    def mask(self, cond, other=None) -> Series:
        """Replace values in a series where the condition is true."""
        if not isinstance(cond, Series):
            raise TypeError(
                f"Only bigframes series condition is supported, received {type(cond).__name__}"
            )
        return self.where(~cond, other)

    def to_frame(self) -> bigframes.DataFrame:
        """Convert Series to DataFrame."""
        block = self._viewed_block

        # To be consistent with Pandas, it assigns 0 as the column name if missing. 0 is the first element of RangeIndex.
        col_names = [self.name] if self.name else ["0"]
        return bigframes.DataFrame(block.index, col_names)

    # Keep this at the bottom of the Series class to avoid
    # confusing type checker by overriding str
    @property
    def str(self) -> strings.StringMethods:
        return strings.StringMethods(self._block, self._value_column, name=self._name)


class SeriesGroupyBy:
    """Represents a deferred series with a grouping expression."""

    def __init__(
        self,
        block: blocks.Block,
        value_column: str,
        by: str,
        value_name: typing.Optional[str] = None,
        key_name: typing.Optional[str] = None,
        dropna=True,
    ):
        # TODO(tbergeron): Support more group-by expression types
        self._block = block
        self._value_column = value_column
        self._by = by
        self._value_name = value_name
        self._key_name = key_name
        self._dropna = dropna  # Applies to aggregations but not windowing

    @property
    def value(self):
        return self._block.expr.get_column(self._value_column)

    def all(self) -> Series:
        """Returns true if and only if all elements are True. Nulls are ignored"""
        return self._aggregate(agg_ops.all_op)

    def any(self) -> Series:
        """Returns true if and only if at least one element is True. Nulls are ignored"""
        return self._aggregate(agg_ops.any_op)

    def count(self) -> Series:
        """Counts the number of elements in each group. Ignores null/nan."""
        return self._aggregate(agg_ops.count_op)

    def sum(self) -> Series:
        """Sums the numeric values for each group in the series. Ignores null/nan."""
        return self._aggregate(agg_ops.sum_op)

    def mean(self) -> Series:
        """Finds the mean of the numeric values for each group in the series. Ignores null/nan."""
        return self._aggregate(agg_ops.mean_op)

    def prod(self) -> Series:
        """Finds the mean of the numeric values for each group in the series. Ignores null/nan."""
        return self._aggregate(agg_ops.product_op)

    def cumsum(self) -> Series:
        return self._cumulative_aggregate(agg_ops.sum_op)

    def cumprod(self) -> Series:
        return self._cumulative_aggregate(agg_ops.product_op)

    def cummax(self) -> Series:
        return self._cumulative_aggregate(agg_ops.max_op)

    def cummin(self) -> Series:
        return self._cumulative_aggregate(agg_ops.min_op)

    def cumcount(self) -> Series:
        return self._cumulative_aggregate(
            agg_ops.rank_op, ignore_na=False, discard_name=True
        )

    def _aggregate(self, aggregate_op: typing.Any) -> Series:
        group_expr = bigframes.core.BigFramesGroupByExpr(self._block.expr, self._by)
        result_expr = group_expr.aggregate(self._value_column, aggregate_op)
        if self._dropna:
            result_expr = result_expr.filter(
                ops.notnull_op(result_expr.get_column(self._by))
            )
        block = blocks.Block(result_expr, index_columns=[self._by])
        if self._key_name:
            block.index.name = self._key_name
        return Series(block, self._value_column, name=self._value_name)

    def _cumulative_aggregate(
        self,
        op: agg_ops.AggregateOp,
        ignore_na=True,
        discard_name=False,
    ):
        expr = self._block.expr
        if expr.reduced_predicate is not None:
            expr = expr.filter(expr.reduced_predicate)
        window = ibis.cumulative_window(
            order_by=expr.ordering, group_by=expr.get_column(self._by)
        )

        def cumulative_op(x: ibis_types.Column):
            cumop = op.with_window(window)(x)
            if ignore_na:
                cumop = ibis.case().when(x.isnull(), ibis.NA).else_(cumop).end()
            return cumop

        block = blocks.Block(expr, self._block.index_columns)
        block.index.name = self._block.index.name
        block.replace_value_columns(
            [
                cumulative_op(
                    # TODO(swast): When we assign literals / scalars, we might not
                    # have a true Column. Do we need to check this before trying to
                    # aggregate such a column?
                    typing.cast(ibis_types.Column, expr.get_column(self._value_column))
                ).name(self._value_column)
            ]
        )
        return Series(
            block,
            self._value_column,
            name=self._value_name if not discard_name else None,
        )


def _interpret_as_ibis_literal(value: typing.Any) -> typing.Optional[ibis_types.Value]:
    if (
        isinstance(value, Series)
        or isinstance(value, bigframes.scalar.Scalar)
        or isinstance(value, pandas.Series)
    ):
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
