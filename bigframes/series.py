"""Series is a 1 dimensional data structure."""

from __future__ import annotations

import typing
from typing import Optional

import ibis
import ibis.expr.types as ibis_types
import pandas

import bigframes.core
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
        expr: bigframes.core.BigFramesExpr,
        value: ibis_types.Value,
        index: Optional[bigframes.core.indexes.implicitjoiner.ImplicitJoiner] = None,
    ):
        if index is None:
            index = bigframes.core.indexes.implicitjoiner.ImplicitJoiner(expr)

        self._expr = expr
        self._index = index
        self._value = value

    @property
    def index(self) -> bigframes.core.indexes.implicitjoiner.ImplicitJoiner:
        return self._index

    @property
    def name(self) -> str:
        # TODO(swast): Introduce a level of indirection over Ibis to allow for
        # more accurate pandas behavior (such as allowing for unnamed or
        # non-uniquely named objects) without breaking SQL.
        return self._value.get_name()

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
        expr = self._expr.projection([self._value])
        return expr.to_ibis_expr()[self._value.get_name()]

    def compute(self) -> pandas.Series:
        """Executes deferred operations and downloads the results."""
        # TODO(swast): Cache the job ID so we can look it up again if they ask
        # for the results? We'd need a way to invalidate the cache if DataFrame
        # becomes mutable, though. Or move this method to the immutable
        # expression class.
        # TODO(swast): We might want to move this method to Session and/or
        # provide our own minimal metadata class. Tight coupling to the
        # BigQuery client library isn't ideal, especially if we want to support
        # a LocalSession for unit testing.
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        # TODO(swast): How to handle MultiIndex?
        if isinstance(self._index, bigframes.core.indexes.index.Index):
            # TODO(swast): We need something that encapsulates data + index
            # column(s) so that Series and DataFrame don't need to duplicate
            # Index logic.
            expr = self._expr.projection(
                [self._value, self._expr.get_column(self._index._index_column)]
            )
            df = expr.to_ibis_expr().execute()
            df = df.set_index(self._index._index_column)
            df.index.name = self._index.name
            return df[self.name]
        else:
            return self._to_ibis_expr().execute()

    def head(self, n: int = 5) -> Series:
        """Limits Series to a specific number of rows."""
        expr = self._expr.apply_limit(n)
        index = self._index.copy()
        index._expr = expr
        return Series(
            expr,
            self._value,
            index=index,
        )

    def len(self) -> "Series":
        """Compute the length of each string."""
        return Series(
            self._expr,
            typing.cast(ibis_types.StringValue, self._value)
            .length()
            .name(self._value.get_name()),
            index=self._index,
        )

    def lower(self) -> "Series":
        """Convert strings in the Series to lowercase."""
        return Series(
            self._expr,
            typing.cast(ibis_types.StringValue, self._value)
            .lower()
            .name(self._value.get_name()),
            index=self._index,
        )

    def upper(self) -> "Series":
        """Convert strings in the Series to uppercase."""
        return Series(
            self._expr,
            typing.cast(ibis_types.StringValue, self._value)
            .upper()
            .name(self._value.get_name()),
            index=self._index,
        )

    def __add__(self, other: float | int | Series | pandas.Series) -> Series:
        (left, right, expr, index) = self._align(other)
        return Series(
            expr,
            (
                typing.cast(ibis_types.NumericValue, left)
                + typing.cast(ibis_types.NumericValue, right)
            ).name(  # type: ignore
                self._value.get_name()
            ),
            index=index,
        )

    def abs(self) -> "Series":
        """Calculate absolute value of numbers in the Series."""
        return Series(
            self._expr,
            typing.cast(ibis_types.NumericValue, self._value)
            .abs()
            .name(self._value.get_name()),
            index=self._index,
        )

    def reverse(self) -> "Series":
        """Reverse strings in the Series."""
        return Series(
            self._expr,
            typing.cast(ibis_types.StringValue, self._value)
            .reverse()
            .name(self._value.get_name()),
            index=self._index,
        )

    def round(self, decimals=0) -> "Series":
        """Round each value in a Series to the given number of decimals."""
        return Series(
            self._expr,
            typing.cast(ibis_types.NumericValue, self._value)
            .round(digits=decimals)
            .name(self._value.get_name()),
            index=self._index,
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
            self._expr,
            typing.cast(ibis_types.StringValue, self._value)[start:stop].name(
                self._value.get_name()
            ),
            index=self._index,
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
            self._expr,
            typing.cast(ibis_types.NumericValue, self._value)
            .negate()
            .name(self._value.get_name()),
            index=self._index,
        )

    def eq(self, other: object) -> Series:
        """
        Element-wise equals between the series and another series or literal.
        None and NaN are not equal to themselves.
        This is inconsitent with Pandas eq behavior with None: https://github.com/pandas-dev/pandas/issues/20442
        """
        # TODO: enforce stricter alignment
        (left, right, expr, index) = self._align(other)
        return Series(
            expr,
            (left == right).fillna(ibis.literal(False)).name(self._value.get_name()),
            index=index,
        )

    def ne(self, other: object) -> Series:
        """
        Element-wise not-equals between the series and another series or literal.
        None and NaN are not equal to themselves.
        This is inconsitent with Pandas eq behavior with None: https://github.com/pandas-dev/pandas/issues/20442
        """
        # TODO: enforce stricter alignment
        (left, right, expr, index) = self._align(other)
        return Series(
            expr,
            (left != right).fillna(ibis.literal(True)).name(self._value.get_name()),
            index=index,
        )

    def __getitem__(self, indexer: Series):
        """Get items using index. Only works with a boolean series derived from the base table."""
        # TODO: enforce stricter alignment
        if indexer._expr._table != self._expr._table:
            raise ValueError(
                "BigFrames expressions can only be filtered with expressions referencing the same table."
            )
        filtered_expr = self._expr.filter(
            typing.cast(ibis_types.BooleanValue, indexer._value)
        )
        index = self._index.copy()
        index._expr = filtered_expr
        return Series(filtered_expr, self._value, index=index)

    def _align(self, other: typing.Any) -> tuple[ibis_types.Value, ibis_types.Value, bigframes.core.BigFramesExpr, bigframes.core.indexes.implicitjoiner.ImplicitJoiner]:  # type: ignore
        """Aligns the series value with other scalar or series object. Returns new left value, right value and joined tabled expression."""
        # TODO: Support deferred scalar
        if isinstance(other, Series):
            combined_index, (get_column_left, get_column_right) = self._index.join(
                other.index, how="outer"
            )
            left_value = get_column_left(self._value.get_name())
            right_value = get_column_right(other._value.get_name())
            expr = combined_index._expr
        elif isinstance(other, bigframes.scalar.Scalar):
            # TODO(tbereron): support deferred scalars.
            raise ValueError("Deferred scalar not yet supported for binary operations.")
        else:
            combined_index = self._index
            left_value = self._value
            right_value = other
            expr = self._expr
        return (left_value, right_value, expr, combined_index)

    def find(self, sub, start=None, end=None) -> "Series":
        """Return the position of the first occurence of substring."""
        return Series(
            self._expr,
            typing.cast(ibis_types.StringValue, self._value)
            .find(sub, start, end)
            .name(self._value.get_name()),
            index=self._index,
        )

    def groupby(self, by: Series):
        """Group the series by a given list of column labels. Only supports grouping by values from another aligned Series."""
        # TODO(tbegeron): support other grouping expressions and options
        return SeriesGroupyBy(self._expr, self._value, by._value)


class SeriesGroupyBy:
    """Represents a deferred series with a grouping expression."""

    def __init__(
        self,
        expr: bigframes.core.BigFramesExpr,
        series_expr: ibis_types.Value,
        grouping_expr: ibis_types.Value,
    ):
        # TODO(tbergeron): Validate alignment (potentially using new index abstraction)
        self._expr = bigframes.core.BigFramesGroupByExpr(expr, grouping_expr)
        self._series_expr = series_expr

    def sum(self) -> Series:
        """Sums the numeric values for each group in the series. Ignores null/nan."""
        # Would be unnamed in pandas, but bigframes needs identifier for now.
        result_name = (
            self._series_expr.get_name() + "_sum"
        )  # Would be unnamed in pandas, but bigframes needs identifier for now.
        new_table_expr = self._expr.aggregate(
            [
                typing.cast(ibis_types.NumericColumn, self._series_expr)
                .sum()
                .fillna(ibis.literal(0))
                .name(result_name)
            ]
        )
        index = bigframes.core.indexes.index.Index(
            new_table_expr, self._expr._by.get_name()
        )
        return Series(
            new_table_expr, new_table_expr.get_column(result_name), index=index
        )

    def mean(self) -> Series:
        """Finds the mean of the numeric values for each group in the series. Ignores null/nan."""
        result_name = (
            self._series_expr.get_name() + "_mean"
        )  # Would be unnamed in pandas, but bigframes needs identifier for now.
        new_table_expr = self._expr.aggregate(
            [
                typing.cast(ibis_types.NumericColumn, self._series_expr)
                .mean()
                .name(result_name)
            ]
        )
        index = bigframes.core.indexes.index.Index(
            new_table_expr, self._expr._by.get_name()
        )
        return Series(
            new_table_expr, new_table_expr.get_column(result_name), index=index
        )
