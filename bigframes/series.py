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

import ibis.expr.types as ibis_types
import numpy
import pandas
import pandas.core.dtypes.common
import typing_extensions

import bigframes.aggregations as agg_ops
import bigframes.core
from bigframes.core import WindowSpec
import bigframes.core.blocks as blocks
import bigframes.core.indexes as indexes
import bigframes.core.indexes.implicitjoiner
from bigframes.core.ordering import OrderingColumnReference, OrderingDirection
import bigframes.core.window
import bigframes.dtypes
import bigframes.indexers
import bigframes.operations as ops
import bigframes.operations.base
import bigframes.operations.datetimes as dt
import bigframes.operations.strings as strings
import bigframes.scalar
import third_party.bigframes_vendored.pandas.core.series as vendored_pandas_series


class Series(bigframes.operations.base.SeriesMethods, vendored_pandas_series.Series):
    @property
    def dt(self) -> dt.DatetimeMethods:
        return dt.DatetimeMethods(self._block)

    @property
    def dtype(self):
        return self._block.dtypes[0]

    @property
    def dtypes(self):
        return self._block.dtypes[0]

    @property
    def index(self) -> indexes.Index:
        return indexes.Index(self)

    @property
    def loc(self) -> bigframes.indexers.LocSeriesIndexer:
        return bigframes.indexers.LocSeriesIndexer(self)

    @property
    def iloc(self) -> bigframes.indexers.IlocSeriesIndexer:
        return bigframes.indexers.IlocSeriesIndexer(self)

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def shape(self) -> typing.Tuple[int]:
        return (self._block.shape()[0],)

    @property
    def size(self) -> int:
        return self._block.shape()[0]

    @property
    def empty(self) -> bool:
        return self._block.shape()[0] == 0

    def copy(self) -> Series:
        return Series(self._block)

    def rename(self, index: Optional[str], **kwargs) -> Series:
        if len(kwargs) != 0:
            raise NotImplementedError(
                "rename does not currently support any keyword arguments."
            )
        block = self._block.with_column_labels([index])
        return Series(block)

    def rename_axis(self, mapper: Optional[str], **kwargs) -> Series:
        if len(kwargs) != 0:
            raise NotImplementedError(
                "rename_axis does not currently support any keyword arguments."
            )
        block = self._block.with_index_labels([mapper])
        return Series(block)

    def reset_index(
        self,
        *,
        name: typing.Optional[str] = None,
        drop: bool = False,
    ) -> bigframes.DataFrame | Series:
        block = self._block.reset_index(drop)
        if drop:
            return Series(block)
        else:
            if name:
                block = block.assign_label(self._value_column, name)
            return bigframes.DataFrame(block)

    def __repr__(self) -> str:
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        # TODO(swast): Avoid downloading the whole series by using job
        # metadata, like we do with DataFrame.
        preview = self.compute()
        return repr(preview)

    def _to_ibis_expr(self):
        """Creates an Ibis table expression representing the Series."""
        expr = self._block.expr.projection([self._value])
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
        return self._apply_unary_op(bigframes.operations.AsTypeOp(dtype))

    def compute(self) -> pandas.Series:
        """Executes deferred operations and downloads the results."""
        df = self._block.compute((self._value_column,))
        series = df[self._value_column]
        series.name = self._name
        return series

    def drop(self, labels: blocks.Label | typing.Sequence[blocks.Label] = None):
        block = self._block
        index_column = block.index_columns[0]

        if _is_list_like(labels):
            block, inverse_condition_id = block.apply_unary_op(
                index_column, ops.partial_right(ops.isin_op, labels)
            )
            block, condition_id = block.apply_unary_op(
                inverse_condition_id, ops.invert_op
            )

        else:
            block, condition_id = block.apply_unary_op(
                index_column, ops.partial_right(ops.ne_op, labels)
            )
        block = block.filter(condition_id)
        block = block.drop_columns([condition_id])
        return Series(block.select_column(self._value_column))

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
        return self._apply_window_op(
            agg_ops.sum_op, bigframes.core.WindowSpec(following=0)
        )

    def cummax(self) -> Series:
        return self._apply_window_op(
            agg_ops.max_op, bigframes.core.WindowSpec(following=0)
        )

    def cummin(self) -> Series:
        return self._apply_window_op(
            agg_ops.min_op, bigframes.core.WindowSpec(following=0)
        )

    def shift(self, periods: int = 1) -> Series:
        window = bigframes.core.WindowSpec(
            preceding=periods if periods > 0 else None,
            following=-periods if periods < 0 else None,
        )
        return self._apply_window_op(agg_ops.ShiftOp(periods), window)

    def diff(self) -> Series:
        return self - self.shift(1)

    def rank(self, axis=0, method: str = "average", na_option: str = "keep") -> Series:
        if method not in ["average", "min", "max", "first", "dense"]:
            raise ValueError(
                "method must be one of 'average', 'min', 'max', 'first', or 'dense'"
            )
        if na_option not in ["keep", "top", "bottom"]:
            raise ValueError("na_option must be one of 'keep', 'top', or 'bottom'")
        # Step 1: Calculate row numbers for each row
        block = self._block
        # Identify null values to be treated according to na_option param
        block, nullity_col_id = block.apply_unary_op(
            self._value_column,
            ops.isnull_op,
        )
        window = WindowSpec(
            # BigQuery has syntax to reorder nulls with "NULLS FIRST/LAST", but that is unavailable through ibis presently, so must order on a separate nullity expression first.
            ordering=(
                OrderingColumnReference(
                    self._value_column,
                    OrderingDirection.ASC,
                    na_last=(na_option in ["bottom", "keep"]),
                ),
            ),
        )
        # Count_op ignores nulls, so if na_option is "top" or "bottom", we instead count the nullity columns, where nulls have been mapped to bools
        block, rownum_id = block.apply_window_op(
            self._value_column if na_option == "keep" else nullity_col_id,
            agg_ops.dense_rank_op if method == "dense" else agg_ops.count_op,
            window_spec=window,
        )
        if method in ["first", "dense"]:
            result = Series(
                block.select_column(rownum_id).assign_label(rownum_id, self.name)
            )
        else:
            # Step 2: Apply aggregate to groups of like input values.
            # This step is skipped for method=='first'
            agg_op = {
                "average": agg_ops.mean_op,
                "min": agg_ops.min_op,
                "max": agg_ops.max_op,
            }[method]
            block, rank_id = block.apply_window_op(
                rownum_id,
                agg_op,
                window_spec=WindowSpec(grouping_keys=(self._value_column,)),
            )
            result = Series(
                block.select_column(rank_id).assign_label(rank_id, self.name)
            )
        if na_option == "keep":
            # For na_option "keep", null inputs must produce null outputs
            result = result.mask(self.isnull(), pandas.NA)
        if method in ["min", "max", "first", "dense"]:
            # Pandas rank always produces Float64, so must cast for aggregation types that produce ints
            result = result.astype(pandas.Float64Dtype())
        return result

    def fillna(self, value=None) -> "Series" | None:
        return self._apply_binary_op(value, ops.fillna_op)

    def head(self, n: int = 5) -> Series:
        return typing.cast(Series, self.iloc[0:n])

    def nlargest(self, n: int = 5, keep: str = "first") -> Series:
        if keep not in ("first", "last", "all"):
            raise ValueError("'keep must be one of 'first', 'last', or 'all'")
        block = self._block
        if keep == "last":
            block = block.reversed()
        ordering = (
            OrderingColumnReference(
                self._value_column, direction=OrderingDirection.DESC
            ),
        )
        block = block.order_by(ordering, stable=True)
        if keep in ("first", "last"):
            return Series(block.slice(0, n))
        else:  # keep == "all":
            block, counter = block.apply_window_op(
                self._value_column,
                agg_ops.rank_op,
                window_spec=WindowSpec(ordering=ordering),
            )
            block, condition = block.apply_unary_op(
                counter, ops.partial_right(ops.le_op, n)
            )
            block = block.filter(condition)
            block = block.select_column(self._value_column)
            return Series(block)

    def nsmallest(self, n: int = 5, keep: str = "first") -> Series:
        if keep not in ("first", "last", "all"):
            raise ValueError("'keep must be one of 'first', 'last', or 'all'")
        block = self._block
        if keep == "last":
            block = block.reversed()
        ordering = (OrderingColumnReference(self._value_column),)
        block = block.order_by(ordering, stable=True)
        if keep in ("first", "last"):
            return Series(block.slice(0, n))
        else:  # keep == "all":
            block, counter = block.apply_window_op(
                self._value_column,
                agg_ops.rank_op,
                window_spec=WindowSpec(ordering=ordering),
            )
            block, condition = block.apply_unary_op(
                counter, ops.partial_right(ops.le_op, n)
            )
            block = block.filter(condition)
            block = block.select_column(self._value_column)
            return Series(block)

    def isna(self) -> "Series":
        return self._apply_unary_op(ops.isnull_op)

    isnull = isna

    def notna(self) -> "Series":
        return self._apply_unary_op(ops.notnull_op)

    notnull = notna

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
        return self._apply_binary_op(other, ops.add_op)

    def radd(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.reverse(ops.add_op))

    def __sub__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.sub(other)

    def __rsub__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.rsub(other)

    def sub(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.sub_op)

    def rsub(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.reverse(ops.sub_op))

    def __mul__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.mul(other)

    def __rmul__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.rmul(other)

    def mul(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.mul_op)

    def rmul(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.reverse(ops.mul_op))

    multiply = mul

    def __truediv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.truediv(other)

    def __rtruediv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.rtruediv(other)

    def truediv(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.div_op)

    def rtruediv(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.reverse(ops.div_op))

    div = truediv

    divide = truediv

    rdiv = rtruediv

    def __floordiv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.floordiv(other)

    def __rfloordiv__(self, other: float | int | Series | pandas.Series) -> Series:
        return self.rfloordiv(other)

    def floordiv(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.floordiv_op)

    def rfloordiv(self, other: float | int | Series | pandas.Series) -> Series:
        return self._apply_binary_op(other, ops.reverse(ops.floordiv_op))

    def __lt__(self, other: float | int | Series | pandas.Series) -> Series:  # type: ignore
        return self.lt(other)

    def __le__(self, other: float | int | Series | pandas.Series) -> Series:  # type: ignore
        return self.le(other)

    def lt(self, other) -> Series:
        return self._apply_binary_op(other, ops.lt_op)

    def le(self, other) -> Series:
        return self._apply_binary_op(other, ops.le_op)

    def __gt__(self, other: float | int | Series | pandas.Series) -> Series:  # type: ignore
        return self.gt(other)

    def __ge__(self, other: float | int | Series | pandas.Series) -> Series:  # type: ignore
        return self.ge(other)

    def gt(self, other) -> Series:
        return self._apply_binary_op(other, ops.gt_op)

    def ge(self, other) -> Series:
        return self._apply_binary_op(other, ops.ge_op)

    def __mod__(self, other) -> Series:  # type: ignore
        return self.mod(other)

    def __rmod__(self, other) -> Series:  # type: ignore
        return self.rmod(other)

    def mod(self, other) -> Series:  # type: ignore
        return self._apply_binary_op(other, ops.mod_op)

    def rmod(self, other) -> Series:  # type: ignore
        return self._apply_binary_op(other, ops.reverse(ops.mod_op))

    def __matmul__(self, other):
        return (self * other).sum()

    dot = __matmul__

    def abs(self) -> Series:
        return self._apply_unary_op(ops.abs_op)

    def round(self, decimals=0) -> "Series":
        def round_op(x: ibis_types.Value, y: ibis_types.Value):
            return typing.cast(ibis_types.NumericValue, x).round(
                digits=typing.cast(ibis_types.IntegerValue, y)
            )

        return self._apply_binary_op(decimals, round_op)

    def all(self) -> bool:
        return typing.cast(bool, self._apply_aggregation(agg_ops.all_op))

    def any(self) -> bool:
        return typing.cast(bool, self._apply_aggregation(agg_ops.any_op))

    def count(self) -> int:
        return typing.cast(int, self._apply_aggregation(agg_ops.count_op))

    def max(self) -> bigframes.scalar.Scalar:
        return self._apply_aggregation(agg_ops.max_op)

    def min(self) -> bigframes.scalar.Scalar:
        return self._apply_aggregation(agg_ops.min_op)

    def std(self) -> float:
        return typing.cast(float, self._apply_aggregation(agg_ops.std_op))

    def var(self) -> float:
        return typing.cast(float, self._apply_aggregation(agg_ops.var_op))

    def _central_moment(self, n: int) -> float:
        """Useful helper for calculating central moment statistics"""
        # Nth central moment is mean((x-mean(x))^n)
        # See: https://en.wikipedia.org/wiki/Moment_(mathematics)
        mean = self.mean()
        mean_deltas = self - mean
        delta_power = mean_deltas
        # TODO(tbergeron): Replace with pow once implemented
        for i in range(1, n):
            delta_power = delta_power * mean_deltas
        return delta_power.mean()

    def kurt(self) -> float:
        # TODO(tbergeron): Cache intermediate count/moment/etc. statistics at block level
        count = self.count()
        moment4 = self._central_moment(4)
        moment2 = self._central_moment(2)  # AKA: Population Variance

        # Kurtosis is often defined as the second standardize moment: moment(4)/moment(2)**2
        # Pandas however uses Fisher’s estimator, implemented below
        numerator = (count + 1) * (count - 1) * moment4
        denominator = (count - 2) * (count - 3) * moment2**2
        adjustment = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))

        return (numerator / denominator) - adjustment

    kurtosis = kurt

    def mode(self) -> Series:
        block = self._block
        # Approach: Count each value, return each value for which count(x) == max(counts))
        value_count_col_id = self._value_column + "_bf_internal_value_count"
        block = block.aggregate(
            [self._value_column],
            ((self._value_column, agg_ops.count_op, value_count_col_id),),
            as_index=False,
        )
        block, max_value_count_col_id = block.apply_window_op(
            value_count_col_id,
            agg_ops.max_op,
            window_spec=WindowSpec(),
        )
        block, is_mode_col_id = block.apply_binary_op(
            value_count_col_id,
            max_value_count_col_id,
            ops.eq_op,
        )
        block = block.filter(is_mode_col_id)
        mode_values_series = Series(
            block.select_column(self._value_column).assign_label(
                self._value_column, self.name
            )
        )
        return typing.cast(
            Series, mode_values_series.sort_values().reset_index(drop=True)
        )

    def mean(self) -> float:
        return typing.cast(float, self._apply_aggregation(agg_ops.mean_op))

    def sum(self) -> float:
        return typing.cast(float, self._apply_aggregation(agg_ops.sum_op))

    def prod(self) -> float:
        return typing.cast(float, self._apply_aggregation(agg_ops.product_op))

    product = prod

    def __eq__(self, other: object) -> Series:  # type: ignore
        return self.eq(other)

    def __ne__(self, other: object) -> Series:  # type: ignore
        return self.ne(other)

    def __invert__(self) -> Series:
        return self._apply_unary_op(ops.invert_op)

    def eq(self, other: object) -> Series:
        # TODO: enforce stricter alignment
        return self._apply_binary_op(other, ops.eq_op)

    def ne(self, other: object) -> Series:
        # TODO: enforce stricter alignment
        return self._apply_binary_op(other, ops.ne_op)

    def where(self, cond, other=None):
        return self._apply_ternary_op(
            cond, other if (other is not None) else pandas.NA, ops.where_op
        )

    def clip(self, lower, upper):
        return self._apply_ternary_op(lower, upper, ops.clip_op)

    def argmax(self) -> bigframes.scalar.Scalar:
        block, row_nums = self._block.promote_offsets()
        block = block.order_by(
            [
                OrderingColumnReference(
                    self._value_column, direction=OrderingDirection.DESC
                ),
                OrderingColumnReference(row_nums),
            ]
        )
        return typing.cast(
            bigframes.scalar.Scalar, Series(block.select_column(row_nums)).iloc[0]
        )

    def argmin(self) -> bigframes.scalar.Scalar:
        block, row_nums = self._block.promote_offsets()
        block = block.order_by(
            [
                OrderingColumnReference(self._value_column),
                OrderingColumnReference(row_nums),
            ]
        )
        return typing.cast(
            bigframes.scalar.Scalar, Series(block.select_column(row_nums)).iloc[0]
        )

    def __getitem__(self, indexer: Series):
        # TODO: enforce stricter alignment, should fail if indexer is missing any keys.
        (left, right, block) = self._align(indexer, "left")
        block = block.filter(right)
        block = block.select_column(left)
        return Series(block)

    def __getattr__(self, key: str):
        if hasattr(pandas.Series(), key):
            raise NotImplementedError(
                "BigFrames has not yet implemented an equivalent to pandas.Series.{key} . Please check https://github.com/googleapis/bigframes/issues for existing feature requests, or file your own. You may also send feedback to the bigframes team at bigframes-feedback@google.com. You may include information about your use case, as well as code snippets or other feedback.".format(
                    key=key
                )
            )
        else:
            raise AttributeError(key)

    def _align(self, other: typing.Any, how="outer") -> tuple[str, str, blocks.Block]:  # type: ignore
        """Aligns the series value with other scalar or series object. Returns new left column id, right column id and joined tabled expression."""
        values, block = self._align_n(
            [
                other,
            ],
            how,
        )
        return (values[0], values[1], block)

    def _align3(self, other1: typing.Any, other2: typing.Any, how="left") -> tuple[str, str, str, blocks.Block]:  # type: ignore
        """Aligns the series value with 2 other scalars or series objects. Returns new values and joined tabled expression."""
        values, index = self._align_n([other1, other2], how)
        return (values[0], values[1], values[2], index)

    def _align_n(
        self, others: typing.Sequence[typing.Any], how="outer"
    ) -> tuple[typing.Sequence[str], blocks.Block]:
        value_ids = [self._value_column]
        block = self._block
        for other in others:
            if isinstance(other, Series):
                combined_index, (
                    get_column_left,
                    get_column_right,
                ) = block.index.join(other._block.index, how=how)
                value_ids = [
                    *[get_column_left(value) for value in value_ids],
                    get_column_right(other._value_column),
                ]
                block = combined_index._block
            elif isinstance(other, bigframes.scalar.DeferredScalar):
                # TODO(tbereron): support deferred scalars.
                raise ValueError("Deferred scalar not supported as operand.")
            elif isinstance(other, pandas.Series):
                raise NotImplementedError(
                    "Pandas series not supported supported as operand."
                )
            else:
                # Will throw if can't interpret as scalar.
                block, constant_col_id = block.create_constant(other)
                value_ids = [*value_ids, constant_col_id]
        return (value_ids, block)

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
        block = self._block
        block, result_id = block.apply_window_op(
            self._value_column, op, window_spec=window_spec, result_label=self.name
        )
        return Series(block.select_column(result_id))

    def _apply_binary_op(
        self,
        other: typing.Any,
        op: ops.BinaryOp,
    ) -> Series:
        """Applies a binary operator to the series and other."""
        (left, right, block) = self._align(other)

        block, result_id = block.apply_binary_op(left, right, op, self._value_column)

        name = self._name
        if isinstance(other, Series) and other.name != self.name:
            name = None

        return Series(block.select_column(result_id).assign_label(result_id, name))

    def _apply_ternary_op(
        self,
        other: typing.Any,
        other2: typing.Any,
        op: ops.TernaryOp,
    ) -> Series:
        """Applies a ternary operator to the series, other, and other2."""
        (x, y, z, block) = self._align3(other, other2)

        block, result_id = block.apply_ternary_op(x, y, z, op, result_label=self.name)

        return Series(block.select_column(result_id))

    def value_counts(self):
        counts = self.groupby(self).count()
        block = counts._block
        block = block.order_by(
            [
                OrderingColumnReference(
                    counts._value_column, direction=OrderingDirection.DESC
                )
            ]
        )
        return Series(
            block.select_column(counts._value_column).assign_label(
                counts._value_column, "count"
            )
        )

    def sort_values(self, *, axis=0, ascending=True, na_position="last") -> Series:
        if na_position not in ["first", "last"]:
            raise ValueError("Param na_position must be one of 'first' or 'last'")
        direction = OrderingDirection.ASC if ascending else OrderingDirection.DESC
        block = self._block.order_by(
            [
                OrderingColumnReference(
                    self._value_column,
                    direction=direction,
                    na_last=(na_position == "last"),
                )
            ]
        )
        return Series(block)

    def sort_index(self, *, axis=0, ascending=True, na_position="last") -> Series:
        # TODO(tbergeron): Support level parameter once multi-index introduced.
        if na_position not in ["first", "last"]:
            raise ValueError("Param na_position must be one of 'first' or 'last'")
        block = self._block
        direction = OrderingDirection.ASC if ascending else OrderingDirection.DESC
        na_last = na_position == "last"
        ordering = [
            OrderingColumnReference(column, direction=direction, na_last=na_last)
            for column in block.index_columns
        ]
        block = block.order_by(ordering)
        return Series(block)

    def rolling(self, window: int, min_periods=None) -> bigframes.core.window.Window:
        # To get n size window, need current row and n-1 preceding rows.
        window_spec = WindowSpec(
            preceding=window - 1, following=0, min_periods=min_periods or window
        )
        return bigframes.core.window.Window(
            self._block, window_spec, self._value_column, self.name
        )

    def expanding(self, min_periods: int = 1) -> bigframes.core.window.Window:
        window_spec = WindowSpec(following=0, min_periods=min_periods)
        return bigframes.core.window.Window(
            self._block, window_spec, self._value_column, self.name
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
        # If all validations passed, must be grouping on the single-level index
        group_key = self._block.index_columns[0]
        return SeriesGroupBy(
            self._block,
            self._value_column,
            group_key,
            value_name=self.name,
            key_name=self.index.name,
            dropna=dropna,
        )

    def _groupby_series(
        self,
        by: Series,
        dropna: bool = True,
    ):
        (value, key, block) = self._align(by, "inner" if dropna else "left")
        return SeriesGroupBy(
            block,
            value,
            key,
            value_name=self.name,
            key_name=by.name,
            dropna=dropna,
        )

    def apply(self, func) -> Series:
        # TODO(shobs, b/274645634): Support convert_dtype, args, **kwargs
        # is actually a ternary op
        return self._apply_unary_op(ops.RemoteFunctionOp(func))

    def add_prefix(self, prefix: str, axis: int | str | None = None) -> Series:
        return Series(self._get_block().add_prefix(prefix))

    def add_suffix(self, suffix: str, axis: int | str | None = None) -> Series:
        return Series(self._get_block().add_suffix(suffix))

    def drop_duplicates(self, *, keep: str = "first") -> Series:
        if keep not in ["first", "last", False]:
            raise ValueError("keep must be one of 'first', 'last', or False'")
        block = self._block
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
        block, val_count_col_id = block.apply_window_op(
            self._value_column,
            agg_ops.count_op,
            window_spec=window_spec,
        )
        block, keep_condition_col_id = block.apply_unary_op(
            val_count_col_id,
            ops.partial_right(ops.le_op, 1),
        )
        block = block.filter(keep_condition_col_id)
        return Series(block.select_column(self._value_column))

    def mask(self, cond, other=None) -> Series:
        if callable(cond):
            cond = self.apply(cond)

        if not isinstance(cond, Series):
            raise TypeError(
                f"Only bigframes series condition is supported, received {type(cond).__name__}"
            )
        return self.where(~cond, other)

    def to_frame(self) -> bigframes.DataFrame:
        # To be consistent with Pandas, it assigns 0 as the column name if missing. 0 is the first element of RangeIndex.
        block = self._block.with_column_labels([self.name] if self.name else ["0"])
        return bigframes.DataFrame(block)

    def to_csv(self, path_or_buf=None, **kwargs) -> typing.Optional[str]:
        # TODO(b/280651142): Implement version that leverages bq export native csv support to bypass local pandas step.
        return self.compute().to_csv(path_or_buf, **kwargs)

    def to_dict(self, into: type[dict] = dict) -> typing.Mapping:
        return typing.cast(dict, self.compute().to_dict(into))

    def to_excel(self, excel_writer, sheet_name="Sheet1", **kwargs) -> None:
        return self.compute().to_excel(excel_writer, sheet_name, **kwargs)

    def to_json(
        self,
        path_or_buf=None,
        orient: typing.Literal[
            "split", "records", "index", "columns", "values", "table"
        ] = "columns",
        **kwargs,
    ) -> typing.Optional[str]:
        # TODO(b/280651142): Implement version that leverages bq export native csv support to bypass local pandas step.
        return self.compute().to_json(path_or_buf, **kwargs)

    def to_latex(
        self, buf=None, columns=None, header=True, index=True, **kwargs
    ) -> typing.Optional[str]:
        return self.compute().to_latex(
            buf, columns=columns, header=header, index=index, **kwargs
        )

    def tolist(self) -> list:
        return self.compute().to_list()

    to_list = tolist

    def to_markdown(
        self,
        buf: typing.IO[str] | None = None,
        mode: str = "wt",
        index: bool = True,
        **kwargs,
    ) -> typing.Optional[str]:
        return self.compute().to_markdown(buf, mode=mode, index=index, **kwargs)  # type: ignore

    def to_numpy(
        self, dtype=None, copy=False, na_value=None, **kwargs
    ) -> numpy.ndarray:
        return self.compute().to_numpy(dtype, copy, na_value, **kwargs)

    def to_pickle(self, path, **kwargs) -> None:
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
        return self.compute().to_xarray()

    # Keep this at the bottom of the Series class to avoid
    # confusing type checker by overriding str
    @property
    def str(self) -> strings.StringMethods:
        return strings.StringMethods(self._block)

    def _slice(
        self,
        start: typing.Optional[int] = None,
        stop: typing.Optional[int] = None,
        step: typing.Optional[int] = None,
    ) -> bigframes.Series:
        return bigframes.Series(
            self._block.slice(start=start, stop=stop, step=step).select_column(
                self._value_column
            ),
        )


class SeriesGroupBy:
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
        )._apply_unary_op(ops.partial_right(ops.sub_op, 1))

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
        return Series(self._block.select_column(self._value_column))

    def _aggregate(self, aggregate_op: agg_ops.AggregateOp) -> Series:
        aggregate_col_id = self._value_column + "_bf_aggregated"
        result_block = self._block.aggregate(
            [self._by],
            ((self._value_column, aggregate_op, aggregate_col_id),),
            dropna=self._dropna,
        )

        return Series(
            result_block.select_column(aggregate_col_id).assign_label(
                aggregate_col_id, self._value_name
            )
        )

    def _apply_window_op(
        self,
        op: agg_ops.WindowOp,
        window_spec: bigframes.core.WindowSpec,
        discard_name=False,
    ):
        label = self._value_name if not discard_name else None
        block, result_id = self._block.apply_window_op(
            self._value_column,
            op,
            result_label=label,
            window_spec=window_spec,
            skip_null_groups=self._dropna,
        )
        return Series(block.select_column(result_id))


def _is_list_like(obj: typing.Any) -> typing_extensions.TypeGuard[typing.Sequence]:
    return pandas.api.types.is_list_like(obj)
