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
from typing import Optional, Union

import ibis
import ibis.common.exceptions
import ibis.expr.types as ibis_types
import numpy
import pandas
import pandas.core.dtypes.common
import typing_extensions

import bigframes.aggregations as agg_ops
import bigframes.core
from bigframes.core import WindowSpec
import bigframes.core.blocks as blocks
import bigframes.core.indexes.implicitjoiner
import bigframes.core.indexes.index
from bigframes.core.ordering import OrderingColumnReference, OrderingDirection
import bigframes.core.window
import bigframes.dtypes
from bigframes.guid import generate_guid
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
    def dtype(self):
        """Returns the dtype of the Series"""
        return self._block.dtypes[0]

    @property
    def dtypes(self):
        """Returns the dtype of the Series"""
        return self._block.dtypes[0]

    @property
    def index(self) -> bigframes.core.indexes.implicitjoiner.ImplicitJoiner:
        """The index of the series."""
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
        """The name of the series."""
        return self._name

    @property
    def shape(self) -> typing.Tuple[int]:
        """Returns the dimensions of the series as a tuple"""
        return (self._block.shape()[0],)

    @property
    def size(self) -> int:
        """Returns the number of elements in the series"""
        return self._block.shape()[0]

    @property
    def empty(self) -> bool:
        """True if and only if the series has no items"""
        return self._block.shape()[0] == 0

    def copy(self) -> Series:
        """Creates a deep copy of the series."""
        return Series(self._block.copy(), self._value_column, name=self.name)

    def reset_index(
        self,
        *,
        name: typing.Optional[str] = None,
        drop: bool = False,
    ):
        """
        Create a new dataframe/series with the index reset.

        Arguments:
            name: column label for original series values if drop=False
            drop: if true, original index is discarded, else it becomes a column

        Returns:
            Series if drop=True else Dataframe
        """
        block = self._block.reset_index()

        if drop:
            return Series(block, self._value_column, name=self.name)
        else:
            old_index_col_id = self._block.index_columns[0]
            # Need to generate some non-null labels until dataframe decouples column labels from internal column id
            # This may also cause collisions as we are overriding unique internal ids
            former_index_label = self.index.name or generate_guid("index_")
            series_value_label = name or self.name or generate_guid("value_")
            # TODO(b/282041134) Remove deprecate_rename_column once label/id separation in dataframe
            block.expr = block._expr.deprecated_rename_column(
                self._value_column, series_value_label
            ).deprecated_rename_column(old_index_col_id, former_index_label)
            block.replace_column_labels([former_index_label, series_value_label])
            return bigframes.DataFrame(block.index)

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

    def astype(
        self,
        dtype: Union[
            bigframes.dtypes.BigFramesDtypeString, bigframes.dtypes.BigFramesDtype
        ],
    ) -> Series:
        """Casts the series to another compatible dtype."""
        ibis_dtype = bigframes.dtypes.bigframes_dtype_to_ibis_dtype(dtype)
        return self._apply_unary_op(bigframes.operations.AsTypeOp(ibis_dtype))

    def compute(self) -> pandas.Series:
        """Executes deferred operations and downloads the results."""
        df = self._viewed_block.compute((self._value_column,))
        series = df[self._value_column]
        series.name = self._name
        return series

    def drop(self, labels: str | typing.Sequence[str]):
        """Drops rows with the given label(s). Note: will never raise KeyError even if labels are not present."""
        block = self._block.copy()
        if isinstance(block._index, bigframes.core.indexes.Index):
            index_column = block.expr.get_column(block._index._index_column)
        else:
            raise ValueError("Cannot drop labels without explicit index.")
        if _is_list_like(labels):
            condition = ~index_column.isin(labels)
        else:
            label_value = _interpret_as_ibis_literal(labels)
            if label_value is None:
                raise ValueError(f"Could not interpret label value(s): {labels}")
            condition = index_column.__ne__(label_value)
        block.expr = self._block.expr.filter(condition)
        return Series(block, self._value_column, name=self.name)

    def between(self, left, right, inclusive="both"):
        """Returns True for values between 'left' and 'right', else False."""
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
        """The cumulative sum of values in the series."""
        return self._apply_window_op(
            agg_ops.sum_op, bigframes.core.WindowSpec(following=0)
        )

    def cummax(self) -> Series:
        """The cumulative maximum of values in the series."""
        return self._apply_window_op(
            agg_ops.max_op, bigframes.core.WindowSpec(following=0)
        )

    def cummin(self) -> Series:
        """The cumulative minimum of values in the series."""
        return self._apply_window_op(
            agg_ops.min_op, bigframes.core.WindowSpec(following=0)
        )

    def shift(self, periods: int = 1) -> Series:
        """Shift index by desired number of periods."""
        window = bigframes.core.WindowSpec(
            preceding=periods if periods > 0 else None,
            following=-periods if periods < 0 else None,
        )
        return self._apply_window_op(agg_ops.ShiftOp(periods), window)

    def diff(self) -> Series:
        """Difference between each element and previous element."""
        return self - self.shift(1)

    METHOD = typing.Literal["average", "min", "max", "first", "dense"]
    NA_OPTION = typing.Literal["keep", "top", "bottom"]

    def rank(
        self, method: METHOD = "average", *, na_option: NA_OPTION = "keep"
    ) -> Series:
        """Calculates the rank of values in the series.

        Arguments:
            method: How groups of tied elements are handled: either 'average', 'min', 'max', or 'first'
                * average: mean rank of items in each group is used as rank
                * min: minimum rank of items in each group is used as rank
                * max: maximum rank of items in each group is used as rank
                * first: original ordering is used to break ties in ranking
            na_option: How NA-values elements are interpreted: either 'keep', 'top', or 'bottom'
                * keep: NA values result in NA rank
                * top: NA values will be ranked above other values
                * bottom: NA values will be ranked below other values

        Returns:
            Series of numeric ranks.
        """
        if method not in ["average", "min", "max", "first", "dense"]:
            raise ValueError(
                "method must be one of 'average', 'min', 'max', 'first', or 'dense'"
            )
        if na_option not in ["keep", "top", "bottom"]:
            raise ValueError("na_option must be one of 'keep', 'top', or 'bottom'")
        if method == "rank":
            # Dense rank has a somewhat different implementation than all other methods
            return self._dense_rank(na_option)
        # Step 1: Calculate row numbers for each row
        block = self._block.copy()
        # Identify null values to be treated according to na_option param
        nullity_col_id = self._value_column + "_bf_internal_nullity"
        block.apply_unary_op(
            self._value_column,
            ops.isnull_op,
            output_name=nullity_col_id,
        )
        window = WindowSpec(
            # BigQuery has syntax to reorder nulls with "NULLS FIRST/LAST", but that is unavailable through ibis presently, so must order on a separate nullity expression first.
            ordering=(
                OrderingColumnReference(
                    self._value_column,
                    OrderingDirection.ASC,
                    na_last=(na_option == "bottom"),
                ),
            ),
        )
        rownum_id = f"{self._value_column}_bf_internal_rownum"
        # Count_op ignores nulls, so if na_option is "top" or "bottom", we instead count the nullity columns, where nulls have been mapped to bools
        block.apply_window_op(
            self._value_column if na_option == "keep" else nullity_col_id,
            agg_ops.count_op,
            window_spec=window,
            output_name=rownum_id,
        )
        if method == "first":
            result = Series(block, rownum_id, name=self.name)
        else:
            # Step 2: Apply aggregate to groups of like input values.
            # This step is skipped for method=='first'
            rank_id = f"{self._value_column}_bigframes_rank"
            agg_op = {
                "average": agg_ops.mean_op,
                "min": agg_ops.min_op,
                "max": agg_ops.max_op,
            }[method]
            block.apply_window_op(
                rownum_id,
                agg_op,
                window_spec=WindowSpec(grouping_keys=(self._value_column,)),
                output_name=rank_id,
            )
            result = Series(block, rank_id, name=self.name)
        if na_option == "keep":
            # For na_option "keep", null inputs must produce null outputs
            result = result.mask(self.isnull(), pandas.NA)
        if method in ["min", "max", "first"]:
            # Pandas rank always produces Float64, so must cast for aggregation types that produce ints
            result = result.astype(pandas.Float64Dtype())
        return result

    def _dense_rank(self, na_option: str):
        # TODO(tbergeron)
        raise NotImplementedError("Dense rank not implemented")

    def fillna(self, value) -> "Series":
        """Fills NULL values."""
        return self._apply_binary_op(value, ops.fillna_op)

    def head(self, n: int = 5) -> Series:
        """Limits Series to a specific number of rows."""
        return typing.cast(Series, self.iloc[0:n])

    def isnull(self) -> "Series":
        """Returns a boolean same-sized object indicating if the values are NULL/missing."""
        return self._apply_unary_op(ops.isnull_op)

    isna = isnull

    def notnull(self) -> "Series":
        """Returns a boolean same-sized object indicating if the values are not NULL/missing."""
        return self._apply_unary_op(ops.notnull_op)

    notna = notnull

    def __and__(self, other: bool | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.and_op)

    __rand__ = __and__

    def __or__(self, other: bool | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.or_op)

    __ror__ = __or__

    def __add__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.add(other)

    def __radd__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.radd(other)

    def add(self, other: float | int | Series | pandas.Series) -> Series:
        """Add series to other element-wise."""
        return self._apply_binary_op(other, ops.add_op)

    def radd(self, other: float | int | Series | pandas.Series) -> Series:
        """Add series to other element-wise."""
        return self._apply_binary_op(other, ops.reverse(ops.add_op))

    def __sub__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.sub(other)

    def __rsub__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.rsub(other)

    def sub(self, other: float | int | Series | pandas.Series) -> Series:
        """Subtract other from series element-wise."""
        return self._apply_binary_op(other, ops.sub_op)

    def rsub(self, other: float | int | Series | pandas.Series) -> Series:
        """Subtract series from other element-wise."""
        return self._apply_binary_op(other, ops.reverse(ops.sub_op))

    def __mul__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.mul(other)

    def __rmul__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.rmul(other)

    def mul(self, other: float | int | Series | pandas.Series) -> Series:
        """Multiply series and other element-wise."""
        return self._apply_binary_op(other, ops.mul_op)

    def rmul(self, other: float | int | Series | pandas.Series) -> Series:
        """Multiply series and other element-wise."""
        return self._apply_binary_op(other, ops.reverse(ops.mul_op))

    multiply = mul

    def __truediv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.truediv(other)

    def __rtruediv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.rtruediv(other)

    def truediv(self, other: float | int | Series | pandas.Series) -> Series:
        """Divide series by other element-wise."""
        return self._apply_binary_op(other, ops.div_op)

    def rtruediv(self, other: float | int | Series | pandas.Series) -> Series:
        """Divide other by series element-wise."""
        return self._apply_binary_op(other, ops.reverse(ops.div_op))

    div = truediv

    divide = truediv

    rdiv = rtruediv

    def __floordiv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.floordiv(other)

    def __rfloordiv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.rfloordiv(other)

    def floordiv(self, other: float | int | Series | pandas.Series) -> Series:
        """Divide series by other element-wise, rounding down to the next integer."""
        return self._apply_binary_op(other, ops.floordiv_op)

    def rfloordiv(self, other: float | int | Series | pandas.Series) -> Series:
        """Divide other by series element-wise, rounding down to the next integer."""
        return self._apply_binary_op(other, ops.reverse(ops.floordiv_op))

    def __lt__(self, other: float | int | Series | pandas.Series) -> Series:  # type: ignore
        return self.lt(other)

    def __le__(self, other: float | int | Series | pandas.Series) -> Series:  # type: ignore
        return self.le(other)

    def lt(self, other) -> Series:
        """Element-wise less than."""
        return self._apply_binary_op(other, ops.lt_op)

    def le(self, other) -> Series:
        """Element-wise less than or equal."""
        return self._apply_binary_op(other, ops.le_op)

    def __gt__(self, other: float | int | Series | pandas.Series) -> Series:  # type: ignore
        return self.gt(other)

    def __ge__(self, other: float | int | Series | pandas.Series) -> Series:  # type: ignore
        return self.ge(other)

    def gt(self, other) -> Series:
        """Element-wise greater than."""
        return self._apply_binary_op(other, ops.gt_op)

    def ge(self, other) -> Series:
        """Element-wise greater than or equal."""
        return self._apply_binary_op(other, ops.ge_op)

    def __mod__(self, other) -> Series:  # type: ignore
        return self.mod(other)

    def __rmod__(self, other) -> Series:  # type: ignore
        return self.rmod(other)

    def mod(self, other) -> Series:  # type: ignore
        """Calculate series modulo other element-wise."""
        return self._apply_binary_op(other, ops.mod_op)

    def rmod(self, other) -> Series:  # type: ignore
        """Calculate other modulo series element-wise."""
        return self._apply_binary_op(other, ops.reverse(ops.mod_op))

    def __matmul__(self, other: Series):
        """Calculate dot product of series and other."""
        return (self * other).sum()

    dot = __matmul__

    def abs(self) -> "Series":
        """Calculate absolute value of numbers in the Series."""
        return self._apply_unary_op(ops.abs_op)

    def round(self, decimals=0) -> "Series":
        """Round each value in a Series to the given number of decimals."""

        def round_op(x: ibis_types.Value, y: ibis_types.Value):
            return typing.cast(ibis_types.NumericValue, x).round(
                digits=typing.cast(ibis_types.IntegerValue, y)
            )

        return self._apply_binary_op(decimals, round_op)

    def all(self) -> bool:
        """Returns true if and only if all elements are True. Nulls are ignored"""
        return typing.cast(bool, self._apply_aggregation(agg_ops.all_op))

    def any(self) -> bool:
        """Returns true if and only if at least one element is True. Nulls are ignored"""
        return typing.cast(bool, self._apply_aggregation(agg_ops.any_op))

    def count(self) -> int:
        """Counts the number of values in the series. Ignores null/nan."""
        return typing.cast(int, self._apply_aggregation(agg_ops.count_op))

    def max(self) -> bigframes.scalar.Scalar:
        """Return the maximum values over the requested axis."""
        return self._apply_aggregation(agg_ops.max_op)

    def min(self) -> bigframes.scalar.Scalar:
        """Return the maximum values over the requested axis."""
        return self._apply_aggregation(agg_ops.min_op)

    def std(self) -> float:
        """Return the standard deviation of the values in the series."""
        return typing.cast(float, self._apply_aggregation(agg_ops.std_op))

    def var(self) -> float:
        """Return the variance of the values in the series."""
        return typing.cast(float, self._apply_aggregation(agg_ops.var_op))

    def mode(self) -> Series:
        """
        Return the mode(s) of the values in the series, in sorted value order.

        The mode(s) are the values(s) that occur the most times in the series.
        """
        block = self._block.copy()
        # Approach: Count each value, return each value for which count(x) == max(counts))
        value_count_col_id = self._value_column + "_bf_internal_value_count"
        block = block.aggregate(
            [self._value_column],
            ((self._value_column, agg_ops.count_op, value_count_col_id),),
        )
        max_value_count_col_id = self._value_column + "_bf_internal_max_value_count"
        block.apply_window_op(
            value_count_col_id,
            agg_ops.max_op,
            window_spec=WindowSpec(),
            output_name=max_value_count_col_id,
        )
        is_mode_col_id = self._value_column + "_bf_internal_is_mode"
        block.project_binary_op(
            value_count_col_id,
            max_value_count_col_id,
            ops.eq_op,
            output_id=is_mode_col_id,
        )
        block.filter(is_mode_col_id)
        return (
            Series(block, self._value_column, name=self.name)
            .sort_values()
            .reset_index(drop=True)
        )

    def mean(self) -> float:
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
        return typing.cast(float, self._apply_aggregation(agg_ops.mean_op))

    def sum(self) -> float:
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
        return typing.cast(float, self._apply_aggregation(agg_ops.sum_op))

    def prod(self) -> float:
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
        return typing.cast(float, self._apply_aggregation(agg_ops.product_op))

    product = prod

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
        block = blocks.Block(index._expr)
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
        block = blocks.Block(index._expr)
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
        """Keep original values where 'cond' is True, else repacle with values from 'other'"""
        return self._apply_ternary_op(
            cond, other if (other is not None) else pandas.NA, ops.where_op
        )

    def clip(self, lower, upper):
        """Clip series values to 'lower' and 'upper' bounds."""
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
            elif isinstance(other, bigframes.scalar.DeferredScalar):
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
        self, op: agg_ops.AggregateOp
    ) -> bigframes.scalar.ImmediateScalar:
        aggregation_result = typing.cast(
            ibis_types.Scalar, op._as_ibis(self[self.notnull()]._to_ibis_expr())
        )
        return bigframes.scalar.DeferredScalar(aggregation_result).compute()

    def _apply_window_op(
        self,
        op: agg_ops.WindowOp,
        window_spec: bigframes.core.WindowSpec,
    ):
        block = self._block.copy()
        block.apply_window_op(self._value_column, op, window_spec=window_spec)
        return Series(
            block,
            self._value_column,
            name=self.name,
        )

    def _apply_binary_op(
        self,
        other: typing.Any,
        op: ops.BinaryOp,
    ) -> Series:
        """Applies a binary operator to the series and other."""
        (left, right, index) = self._align(other)

        block = index._block
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
        """Applies a ternary operator to the series, other, and other2."""
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

    def find(self, sub, start=None, end=None) -> "Series":
        """Return the position of the first occurence of substring."""
        # is actually a ternary op
        class FindOp(ops.UnaryOp):
            def _as_ibis(self, x: ibis_types.Value):
                return typing.cast(ibis_types.StringValue, x).find(sub, start, end)

        return self._apply_unary_op(FindOp())

    def value_counts(self):
        """Count the number of occurences of each value in the Series. Results are sorted in decreasing order of frequency."""
        counts = self.groupby(self).count()
        block = counts._block
        block.expr = block.expr.order_by(
            [
                OrderingColumnReference(
                    counts._value_column, direction=OrderingDirection.DESC
                )
            ]
        )
        return Series(block, counts._value_column, name="count")

    def sort_values(self, axis=0, ascending=True, na_position="last") -> Series:
        """Sort series by values in ascending or descending order."""
        if na_position not in ["first", "last"]:
            raise ValueError("Param na_position must be one of 'first' or 'last'")
        block = self._viewed_block
        direction = OrderingDirection.ASC if ascending else OrderingDirection.DESC
        block.expr = block.expr.order_by(
            [
                OrderingColumnReference(
                    self._value_column,
                    direction=direction,
                    na_last=(na_position == "last"),
                )
            ]
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
        direction = OrderingDirection.ASC if ascending else OrderingDirection.DESC
        na_last = na_position == "last"
        ordering = [
            OrderingColumnReference(column, direction=direction, na_last=na_last)
            for column in block.index_columns
        ]
        block.expr = block.expr.order_by(ordering)
        return Series(
            block,
            self._value_column,
            name=self.name,
        )

    def rolling(self, window: int, *, min_periods=None) -> bigframes.core.window.Window:
        """Create a rolling window over the series

        Arguments:
            window: the fixed number of observations used for each window
            min_periods: number of observations needed to produce a non-null result for a row, defaults to window
        """
        # To get n size window, need current row and n-1 preceding rows.
        window_spec = WindowSpec(
            preceding=window - 1, following=0, min_periods=min_periods or window
        )
        return bigframes.core.window.Window(
            self._block.copy(), window_spec, self._value_column, self.name
        )

    def expanding(self, min_periods: int = 1) -> bigframes.core.window.Window:
        """Create a expanding window over the series

        Arguments:
            min_periods: number of observations needed to produce a non-null result for a row, defaults to 1
        """
        window_spec = WindowSpec(following=0, min_periods=min_periods)
        return bigframes.core.window.Window(
            self._block.copy(), window_spec, self._value_column, self.name
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
        (value, key, index) = self._align(by, "inner" if dropna else "left")
        return SeriesGroupyBy(
            index._block,
            value.get_name(),
            key.get_name(),
            value_name=self.name,
            key_name=by.name,
            dropna=dropna,
        )

    def apply(self, func) -> Series:
        """Returns a series with a user defined function applied."""
        # TODO(shobs, b/274645634): Support convert_dtype, args, **kwargs
        # is actually a ternary op
        class RemoteOp(ops.UnaryOp):
            def _as_ibis(self, x: ibis_types.Value):
                return func(x)

        return self._apply_unary_op(RemoteOp())

    def drop_duplicates(self, *, keep: str = "first") -> Series:
        """
        Creates copy of series with any duplicate values removed.

        Args:
            keep: 'first', 'last' or False. Determines which value is kept for values with duplicates. All instances discarded if set to False
        """
        if keep not in ["first", "last", False]:
            raise ValueError("keep must be one of 'first', 'last', or False'")
        block = self._block.copy()
        val_count_col_id = self._value_column + "_bf_internal_counter_before"
        if keep == "first":
            # Count how many copies occur up to current copy of value
            # Discard this value if there are copies BEFORE
            window_spec = WindowSpec(
                grouping_keys=(self._value_column,),
                following=0,
            )
        elif keep == "last":
            # Count how many copies occur up to current copy of values
            # Discard this value if there are copies AFTER
            window_spec = WindowSpec(
                grouping_keys=(self._value_column,),
                preceding=0,
            )
        else:  # keep == False
            # Count how many copies of the value occur in entire series.
            # Discard this value if there are copies ANYWHERE
            window_spec = WindowSpec(grouping_keys=(self._value_column,))
        block.apply_window_op(
            self._value_column,
            agg_ops.count_op,
            window_spec=window_spec,
            output_name=val_count_col_id,
        )
        keep_condition_col_id = self._value_column + "_bf_internal_keep_cond"
        block.apply_unary_op(
            val_count_col_id,
            ops.partial_right(ops.le_op, 1),
            output_name=keep_condition_col_id,
        )
        block.filter(keep_condition_col_id)
        return Series(block, self._value_column, name=self._name)

    def mask(self, cond, other=None) -> Series:
        """Replace values in a series where the condition is true."""
        if callable(cond):
            cond = self.apply(cond)

        if not isinstance(cond, Series):
            raise TypeError(
                f"Only bigframes series condition is supported, received {type(cond).__name__}"
            )
        return self.where(~cond, other)

    def to_frame(self) -> bigframes.DataFrame:
        """Convert Series to DataFrame."""
        block = self._viewed_block

        # To be consistent with Pandas, it assigns 0 as the column name if missing. 0 is the first element of RangeIndex.
        block.replace_column_labels([self.name] if self.name else ["0"])
        return bigframes.DataFrame(block.index)

    def to_csv(self, path_or_buf=None, **kwargs) -> typing.Optional[str]:
        """Convert series to a excel."""
        # TODO(b/280651142): Implement version that leverages bq export native csv support to bypass local pandas step.
        return self.compute().to_csv(path_or_buf, **kwargs)

    def to_dict(self, into: type[dict] = dict) -> typing.Mapping:
        """Convert series to a dictionary."""
        return self.compute().to_dict(into)

    def to_excel(self, excel_writer, sheet_name="Sheet1", **kwargs) -> None:
        """Convert series to a excel."""
        return self.compute().to_excel(excel_writer, sheet_name, **kwargs)

    def to_json(self, path_or_buf=None, **kwargs) -> typing.Optional[str]:
        """Convert series to json."""
        # TODO(b/280651142): Implement version that leverages bq export native csv support to bypass local pandas step.
        return self.compute().to_json(path_or_buf, **kwargs)

    def to_latex(
        self, buf=None, columns=None, header=True, index=True, **kwargs
    ) -> typing.Optional[str]:
        """Convert series to a latex."""
        return self.compute().to_latex(
            buf, columns=columns, header=header, index=index, **kwargs
        )

    def to_list(self) -> list:
        """Convert series to a list."""
        return self.compute().to_list()

    def to_markdown(
        self, buf=None, mode="wt", index=True, **kwargs
    ) -> typing.Optional[str]:
        """Convert series to markdown."""
        return self.compute().to_markdown(buf, mode, index, **kwargs)

    def to_numpy(
        self, dtype=None, copy=False, na_value=None, **kwargs
    ) -> numpy.ndarray:
        """Convert series to a numpy array."""
        return self.compute().to_numpy(dtype, copy, na_value, **kwargs)

    def to_pickle(self, path, **kwargs) -> None:
        """Convert series to a pickle."""
        return self.compute().to_pickle(path, **kwargs)

    def to_string(
        self,
        buf=None,
        na_rep="NaN",
        float_format=None,
        header=True,
        index=True,
        length=False,
        dtype=False,
        name=False,
        max_rows=None,
        min_rows=None,
    ) -> typing.Optional[str]:
        """Convert series to string."""
        return self.compute().to_string(
            buf,
            na_rep,
            float_format,
            header,
            index,
            length,
            dtype,
            name,
            max_rows,
            min_rows,
        )

    def to_xarray(self):
        """Convert series to an xarray."""
        return self.compute().to_xarray()

    # Keep this at the bottom of the Series class to avoid
    # confusing type checker by overriding str
    @property
    def str(self) -> strings.StringMethods:
        """Methods that act on a string Series.

        Returns:
            bigframes.operations.strings.StringMethods: Methods that act on a string Series.
        """
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

    def std(self) -> Series:
        """Return the standard deviation of the values in each group in the series."""
        return self._aggregate(agg_ops.std_op)

    def var(self) -> Series:
        """Return the variance of the values in each group in the series."""
        return self._aggregate(agg_ops.var_op)

    def prod(self) -> Series:
        """Finds the mean of the numeric values for each group in the series. Ignores null/nan."""
        return self._aggregate(agg_ops.product_op)

    def cumsum(self) -> Series:
        """Calculate the cumulative sum of values in each grouping."""
        return self._apply_window_op(
            agg_ops.sum_op,
            bigframes.core.WindowSpec(grouping_keys=(self._by,), following=0),
        )

    def cumprod(self) -> Series:
        """Calculate the cumulative product of values in each grouping."""
        return self._apply_window_op(
            agg_ops.product_op,
            bigframes.core.WindowSpec(grouping_keys=(self._by,), following=0),
        )

    def cummax(self) -> Series:
        """Calculate the cumulative maximum of values in each grouping."""
        return self._apply_window_op(
            agg_ops.max_op,
            bigframes.core.WindowSpec(grouping_keys=(self._by,), following=0),
        )

    def cummin(self) -> Series:
        """Calculate the cumulative minimum of values in each grouping."""
        return self._apply_window_op(
            agg_ops.min_op,
            bigframes.core.WindowSpec(grouping_keys=(self._by,), following=0),
        )

    def cumcount(self) -> Series:
        """Calculate the cumulative count of values within each grouping."""
        return self._apply_window_op(
            agg_ops.rank_op,
            bigframes.core.WindowSpec(grouping_keys=(self._by,), following=0),
            discard_name=True,
        )

    def shift(self, periods=1) -> Series:
        """Shift index by desired number of periods."""
        window = bigframes.core.WindowSpec(
            grouping_keys=(self._by,),
            preceding=periods if periods > 0 else None,
            following=-periods if periods < 0 else None,
        )
        return self._apply_window_op(agg_ops.ShiftOp(periods), window)

    def diff(self) -> Series:
        """Difference between each element and previous element."""
        return self._ungroup() - self.shift(1)

    def _ungroup(self) -> Series:
        """Convert back to regular series, without aggregating."""
        return Series(self._block, self._value_column, name=self._value_name)

    def _aggregate(self, aggregate_op: agg_ops.AggregateOp) -> Series:
        aggregate_col_id = self._value_column + "_bf_aggregated"
        result_block = self._block.aggregate(
            [self._by],
            ((self._value_column, aggregate_op, aggregate_col_id),),
            dropna=self._dropna,
        )
        result_block.index_columns = [self._by]
        if self._key_name:
            result_block.index.name = self._key_name
        return Series(result_block, aggregate_col_id, name=self._value_name)

    def _apply_window_op(
        self,
        op: agg_ops.WindowOp,
        window_spec: bigframes.core.WindowSpec,
        discard_name=False,
    ):
        block = self._block.copy()
        block.apply_window_op(self._value_column, op, window_spec=window_spec)
        return Series(
            block,
            self._value_column,
            name=self._value_name if not discard_name else None,
        )


def _interpret_as_ibis_literal(value: typing.Any) -> typing.Optional[ibis_types.Value]:
    if (
        isinstance(value, Series)
        or isinstance(value, bigframes.scalar.DeferredScalar)
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
