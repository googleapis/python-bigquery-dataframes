"""Series is a 1 dimensional data structure."""

from __future__ import annotations

import typing

import ibis
import ibis.expr.types as ibis_types
import pandas
from google.cloud import bigquery

import bigframes.core
import bigframes.core.indexes.implicitjoiner
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
    ):
        self._expr = expr
        # TODO(swast): How can we consolidate BigFramesExpr and ImplicitJoiner?
        self._index = bigframes.core.indexes.implicitjoiner.ImplicitJoiner(expr)
        self._value = value

    @property
    def index(self) -> bigframes.core.indexes.implicitjoiner.ImplicitJoiner:
        return self._index

    def __repr__(self) -> str:
        """Converts a Series to a string."""
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        job = self._execute_query().result(max_results=10)
        rows = job.total_rows
        preview = job.to_dataframe()[job.schema[0].name]

        # TODO(swast): Print the SQL too?
        lines = [repr(preview)]
        if rows > len(preview.index):
            lines.append("...")

        lines.append("")
        lines.append(f"[{rows} rows]")
        return "\n".join(lines)

    def _to_ibis_expr(self):
        """Creates an Ibis table expression representing the Series."""
        expr = self._expr.projection([self._value])
        return expr.to_ibis_expr()[self._value.get_name()]

    def _execute_query(self) -> bigquery.QueryJob:
        """Execute a query and return metadata about the results."""
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
        return self._to_ibis_expr().execute()

    def compute(self) -> pandas.Series:
        """Executes deferred operations and downloads the results."""
        value = self._to_ibis_expr()
        return value.execute()

    def head(self, n: int = 5) -> Series:
        """Limits Series to a specific number of rows."""
        return Series(
            self._expr.apply_limit(n),
            self._value,
        )

    def lower(self) -> "Series":
        """Convert strings in the Series to lowercase."""
        return Series(
            self._expr,
            typing.cast(ibis_types.StringValue, self._value)
            .lower()
            .name(self._value.get_name()),
        )

    def upper(self) -> "Series":
        """Convert strings in the Series to uppercase."""
        return Series(
            self._expr,
            typing.cast(ibis_types.StringValue, self._value)
            .upper()
            .name(self._value.get_name()),
        )

    def __add__(self, other: float | int | Series | pandas.Series) -> Series:
        if isinstance(other, Series):
            combined_index, (left_has_value, right_has_value) = self._index.join(
                other.index, how="outer"
            )
            left_value = (
                ibis.case().when(left_has_value, self._value).else_(ibis.null()).end()
            )
            right_value = (
                ibis.case().when(right_has_value, other._value).else_(ibis.null()).end()
            )

            return Series(
                combined_index._expr,
                (
                    typing.cast(ibis_types.NumericValue, left_value)
                    + typing.cast(ibis_types.NumericValue, right_value)
                ).name(self._value.get_name()),
            )
        else:
            return Series(
                self._expr,
                typing.cast(
                    ibis_types.NumericValue, self._value + other  # type: ignore
                ).name(self._value.get_name()),
            )

    def abs(self) -> "Series":
        """Calculate absolute value of numbers in the Series."""
        return Series(
            self._expr,
            typing.cast(ibis_types.NumericValue, self._value)
            .abs()
            .name(self._value.get_name()),
        )

    def reverse(self) -> "Series":
        """Reverse strings in the Series."""
        return Series(
            self._expr,
            typing.cast(ibis_types.StringValue, self._value)
            .reverse()
            .name(self._value.get_name()),
        )

    def round(self, decimals=0) -> "Series":
        """Round each value in a Series to the given number of decimals."""
        return Series(
            self._expr,
            typing.cast(ibis_types.NumericValue, self._value)
            .round(digits=decimals)
            .name(self._value.get_name()),
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
        )
