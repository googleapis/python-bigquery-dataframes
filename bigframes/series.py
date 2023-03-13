"""Series is a 1 dimensional data structure."""

from __future__ import annotations

import typing

import ibis
import ibis.common.exceptions
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types
import pandas

import bigframes.core
import bigframes.core.blocks as blocks
import bigframes.core.indexes.implicitjoiner
import bigframes.core.indexes.index
import bigframes.scalar


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
    ):
        self._block = block
        self._value_column = value_column

    @property
    def _value(self) -> ibis_types.Value:
        """Private property to get Ibis expression for the value column."""
        return self._block.expr.get_column(self._value_column)

    @property
    def index(self) -> bigframes.core.indexes.implicitjoiner.ImplicitJoiner:
        return self._block.index

    @property
    def name(self) -> str:
        # TODO(swast): Introduce a level of indirection over Ibis to allow for
        # more accurate pandas behavior (such as allowing for unnamed or
        # non-uniquely named objects) without breaking SQL.
        return self._value_column

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
        expr = self._block.expr.projection([self._value])
        return expr.to_ibis_expr()[self._value_column]

    def compute(self) -> pandas.Series:
        """Executes deferred operations and downloads the results."""
        df = self._block.compute((self._value_column,))
        # TODO(swast): Rename Series so name doesn't have to match Ibis
        # expression.
        return df[self._value_column]

    def head(self, n: int = 5) -> Series:
        """Limits Series to a specific number of rows."""
        block = self._block.copy()
        block.expr = self._block.expr.apply_limit(n)
        return Series(
            block,
            self._value_column,
        )

    def len(self) -> "Series":
        """Compute the length of each string."""
        return Series(
            self._block.copy(
                [
                    typing.cast(ibis_types.StringValue, self._value)
                    .length()
                    .name(self._value_column),
                ]
            ),
            self._value_column,
        )

    def isnull(self) -> "Series":
        """Returns a boolean same-sized object indicating if the values are NULL/missing."""
        return Series(
            self._block.copy(
                [
                    self._value.isnull().name(self._value_column),
                ]
            ),
            self._value_column,
        )

    isna = isnull

    def notnull(self) -> "Series":
        """Returns a boolean same-sized object indicating if the values are not NULL/missing."""
        return Series(
            self._block.copy(
                [
                    self._value.notnull().name(self._value_column),
                ]
            ),
            self._value_column,
        )

    notna = notnull

    def lower(self) -> "Series":
        """Convert strings in the Series to lowercase."""
        return Series(
            self._block.copy(
                [
                    typing.cast(ibis_types.StringValue, self._value)
                    .lower()
                    .name(self._value_column),
                ]
            ),
            self._value_column,
        )

    def upper(self) -> "Series":
        """Convert strings in the Series to uppercase."""
        return Series(
            self._block.copy(
                [
                    typing.cast(ibis_types.StringValue, self._value)
                    .upper()
                    .name(self._value_column),
                ]
            ),
            self._value_column,
        )

    def __add__(self, other: float | int | Series | pandas.Series) -> Series:
        def add_op(
            x: ibis_types.Value,
            y: ibis_types.Value,
        ):
            return typing.cast(ibis_types.NumericValue, x) + typing.cast(
                ibis_types.NumericValue, y
            )

        return self._apply_binary_op(other, add_op)

    def __sub__(self, other: float | int | Series | pandas.Series) -> Series:
        def sub_op(
            x: ibis_types.Value,
            y: ibis_types.Value,
        ):
            return typing.cast(ibis_types.NumericValue, x) - typing.cast(
                ibis_types.NumericValue, y
            )

        return self._apply_binary_op(other, sub_op)

    def __mul__(self, other: float | int | Series | pandas.Series) -> Series:
        def mul_op(
            x: ibis_types.Value,
            y: ibis_types.Value,
        ):
            return typing.cast(ibis_types.NumericValue, x) * typing.cast(
                ibis_types.NumericValue, y
            )

        return self._apply_binary_op(other, mul_op)

    def __truediv__(self, other: float | int | Series | pandas.Series) -> Series:
        def div_op(
            x: ibis_types.Value,
            y: ibis_types.Value,
        ):
            return typing.cast(ibis_types.NumericValue, x) / typing.cast(
                ibis_types.NumericValue, y
            )

        return self._apply_binary_op(other, div_op, ibis_dtypes.float)

    def __lt__(self, other) -> Series:  # type: ignore
        def lt_op(
            x: ibis_types.Value,
            y: ibis_types.Value,
        ):
            return x < y

        return self._apply_binary_op(other, lt_op, ibis_dtypes.bool)

    def __le__(self, other) -> Series:  # type: ignore
        def le_op(
            x: ibis_types.Value,
            y: ibis_types.Value,
        ):
            return x <= y

        return self._apply_binary_op(other, le_op, ibis_dtypes.bool)

    def __gt__(self, other) -> Series:  # type: ignore
        def gt_op(
            x: ibis_types.Value,
            y: ibis_types.Value,
        ):
            return x > y

        return self._apply_binary_op(other, gt_op, ibis_dtypes.bool)

    def __ge__(self, other) -> Series:  # type: ignore
        def ge_op(
            x: ibis_types.Value,
            y: ibis_types.Value,
        ):
            return x >= y

        return self._apply_binary_op(other, ge_op, ibis_dtypes.bool)

    def abs(self) -> "Series":
        """Calculate absolute value of numbers in the Series."""
        return Series(
            self._block.copy(
                [
                    typing.cast(ibis_types.NumericValue, self._value)
                    .abs()
                    .name(self._value_column),
                ]
            ),
            self._value_column,
        )

    def reverse(self) -> "Series":
        """Reverse strings in the Series."""
        return Series(
            self._block.copy(
                [
                    typing.cast(ibis_types.StringValue, self._value)
                    .reverse()
                    .name(self._value_column),
                ]
            ),
            self._value_column,
        )

    def round(self, decimals=0) -> "Series":
        """Round each value in a Series to the given number of decimals."""
        return Series(
            self._block.copy(
                [
                    typing.cast(ibis_types.NumericValue, self._value)
                    .round(digits=decimals)
                    .name(self._value_column),
                ]
            ),
            self._value_column,
        )

    def mean(self) -> bigframes.scalar.Scalar:
        """Finds the mean of the numeric values in the series. Ignores null/nan."""
        return bigframes.scalar.Scalar(
            typing.cast(ibis_types.NumericColumn, self._to_ibis_expr()).mean()
        )

    def sum(self) -> bigframes.scalar.Scalar:
        """Sums the numeric values in the series. Ignores null/nan."""
        return bigframes.scalar.Scalar(
            typing.cast(ibis_types.NumericColumn, self._to_ibis_expr()).sum()
        )

    def slice(self, start=None, stop=None) -> "Series":
        """Slice substrings from each element in the Series."""
        return Series(
            self._block.copy(
                [
                    typing.cast(ibis_types.StringValue, self._value)[start:stop].name(
                        self._value_column
                    ),
                ]
            ),
            self._value_column,
        )

    def __eq__(self, other: object) -> Series:  # type: ignore
        """Element-wise equals between the series and another series or literal."""
        return self.eq(other)

    def __ne__(self, other: object) -> Series:  # type: ignore
        """Element-wise not-equals between the series and another series or literal."""
        return self.ne(other)

    def __invert__(self) -> Series:
        """Element-wise logical negation. Does not handle null or nan values."""
        return Series(
            self._block.copy(
                [
                    typing.cast(ibis_types.NumericValue, self._value)
                    .negate()
                    .name(self._value_column),
                ]
            ),
            self._value_column,
        )

    def eq(self, other: object) -> Series:
        """
        Element-wise equals between the series and another series or literal.
        None and NaN are not equal to themselves.
        This is inconsitent with Pandas eq behavior with None: https://github.com/pandas-dev/pandas/issues/20442
        """
        # TODO: enforce stricter alignment
        (left, right, index) = self._align(other)
        block = self._block.copy()
        block.index = index
        block.replace_value_columns(
            [
                (left == right).fillna(ibis.literal(False)).name(self._value_column),
            ]
        )
        return Series(
            block,
            self._value_column,
        )

    def ne(self, other: object) -> Series:
        """
        Element-wise not-equals between the series and another series or literal.
        None and NaN are not equal to themselves.
        This is inconsitent with Pandas eq behavior with None: https://github.com/pandas-dev/pandas/issues/20442
        """
        # TODO: enforce stricter alignment
        (left, right, index) = self._align(other)
        block = self._block.copy()
        block.index = index
        block.replace_value_columns(
            [
                (left != right).fillna(ibis.literal(True)).name(self._value_column),
            ]
        )
        return Series(
            block,
            self._value_column,
        )

    def __getitem__(self, indexer: Series):
        """Get items using boolean series indexer."""
        # TODO: enforce stricter alignment, should fail if indexer is missing any keys.
        (left, right, index) = self._align(indexer, "left")
        block = self._block.copy()
        block.index = index
        block.replace_value_columns(
            [
                left,
            ]
        )
        filtered_expr = block.expr.filter((right == ibis.literal(True)))
        block.expr = filtered_expr
        return Series(block, self._value_column)

    def _align(self, other: typing.Any, how="outer") -> tuple[ibis_types.Value, ibis_types.Value, bigframes.core.indexes.implicitjoiner.ImplicitJoiner]:  # type: ignore
        """Aligns the series value with other scalar or series object. Returns new left value, right value and joined tabled expression."""
        # TODO: Support deferred scalar
        if isinstance(other, Series):
            combined_index, (
                get_column_left,
                get_column_right,
            ) = self._block.index.join(other.index, how=how)
            left_value = get_column_left(self._value_column)
            right_value = get_column_right(other._value.get_name())
            return (left_value, right_value, combined_index)
        elif isinstance(other, bigframes.scalar.Scalar):
            # TODO(tbereron): support deferred scalars.
            raise ValueError("Deferred scalar not yet supported for binary operations.")
        as_literal_value = _interpret_as_ibis_literal(other)
        if as_literal_value is not None:
            combined_index = self._block.index
            left_value = self._value
            right_value = as_literal_value
            return (left_value, right_value, combined_index)
        else:
            return NotImplemented

    def _apply_binary_op(
        self,
        other: typing.Any,
        op: typing.Callable[[ibis_types.Value, ibis_types.Value], ibis_types.Value],
        expected_dtype: typing.Optional[ibis_dtypes.DataType] = None,
    ) -> Series:
        """Applies a binary operator to the series and other."""
        (left, right, index) = self._align(other)
        block = self._block.copy()
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
        return Series(
            block,
            self._value_column,
        )

    def find(self, sub, start=None, end=None) -> "Series":
        """Return the position of the first occurence of substring."""
        return Series(
            self._block.copy(
                [
                    typing.cast(ibis_types.StringValue, self._value)
                    .find(sub, start, end)
                    .name(self._value_column),
                ]
            ),
            self._value_column,
        )

    def groupby(self, by: Series, *, dropna: bool = True):
        """Group the series by a given list of column labels. Only supports grouping by values from another aligned Series."""
        # TODO: Support groupby level
        if dropna:
            by = by[by.notna()]
        (value, key, index) = self._align(by, "inner" if dropna else "left")
        block = self._block.copy()
        block.index = index
        block.replace_value_columns([key, value])
        return SeriesGroupyBy(block, value.get_name(), key.get_name())


class SeriesGroupyBy:
    """Represents a deferred series with a grouping expression."""

    def __init__(self, block: blocks.Block, value_column: str, by: str):
        # TODO(tbergeron): Support more group-by expression types
        self._block = block
        self._value_column = value_column
        self._by = by

    def sum(self) -> Series:
        """Sums the numeric values for each group in the series. Ignores null/nan."""
        return self._aggregate(
            typing.cast(
                ibis_types.NumericColumn,
                self._block.expr.get_column(self._value_column),
            )
            .sum()
            .name(self._value_column + "_sum")
        )

    def mean(self) -> Series:
        """Finds the mean of the numeric values for each group in the series. Ignores null/nan."""
        return self._aggregate(
            typing.cast(
                ibis_types.NumericColumn,
                self._block.expr.get_column(self._value_column),
            )
            .mean()
            .name(self._value_column + "_mean")
        )

    def _aggregate(self, metric: ibis_types.Scalar) -> Series:
        group_expr = bigframes.core.BigFramesGroupByExpr(self._block.expr, self._by)
        block = blocks.Block(group_expr.aggregate((metric,)), (self._by,))
        return Series(block, metric.get_name())


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
