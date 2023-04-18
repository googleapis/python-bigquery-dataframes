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

from __future__ import annotations

import typing

import ibis
import ibis.expr.types as ibis_types

T = typing.TypeVar("T", bound=ibis_types.Value)


class AggregateOp:
    _window = None

    def with_window(self, ibis_window):
        self._window = ibis_window
        return self

    def _windowize(self, value: T) -> T:
        return value.over(self._window) if self._window else value


def numeric_op(operation):
    def constrined_op(op, column: ibis_types.Column):
        if column.type().is_numeric():
            return operation(op, column)
        else:
            raise ValueError(
                f"Numeric operation cannot be applied to type {column.type()}"
            )

    return constrined_op


class SumOp(AggregateOp):
    @numeric_op
    def __call__(self, column: ibis_types.NumericColumn) -> ibis_types.NumericValue:
        return self._windowize(column.sum())


class MeanOp(AggregateOp):
    @numeric_op
    def __call__(self, column: ibis_types.NumericColumn) -> ibis_types.NumericValue:
        return self._windowize(column.mean())


class ProductOp(AggregateOp):
    @numeric_op
    def __call__(self, column: ibis_types.NumericColumn) -> ibis_types.NumericValue:
        # Need to short-circuit as log with zeroes is illegal sql
        is_zero = typing.cast(ibis_types.BooleanColumn, (column == 0))

        # There is no product sql aggregate function, so must implement as a sum of logs, and then
        # apply power after. Note, log and power base must be equal! This impl uses base 2.
        logs = typing.cast(
            ibis_types.NumericColumn,
            ibis.case().when(is_zero, 0).else_(column.abs().log2()).end(),
        )
        logs_sum = self._windowize(logs.sum())
        magnitude = typing.cast(ibis_types.NumericValue, ibis_types.literal(2)).pow(
            logs_sum
        )

        # Can't determine sign from logs, so have to determine parity of count of negative inputs
        is_negative = typing.cast(
            ibis_types.NumericColumn,
            ibis.case().when(column.sign() == -1, 1).else_(0).end(),
        )
        negative_count = self._windowize(is_negative.sum())
        negative_count_parity = negative_count % typing.cast(
            ibis_types.NumericValue, ibis.literal(2)
        )  # 1 if result should be negative, otherwise 0

        any_zeroes = self._windowize(is_zero.any())
        float_result = (
            ibis.case()
            .when(any_zeroes, ibis_types.literal(0))
            .else_(magnitude * pow(-1, negative_count_parity))
            .end()
        )
        return float_result.cast(column.type())


class MaxOp(AggregateOp):
    def __call__(self, column: ibis_types.Column) -> ibis_types.Value:
        return self._windowize(column.max())


class MinOp(AggregateOp):
    def __call__(self, column: ibis_types.Column) -> ibis_types.Value:
        return self._windowize(column.min())


class CountOp(AggregateOp):
    def __call__(self, column: ibis_types.Column) -> ibis_types.IntegerValue:
        return self._windowize(column.count())


class RankOp(AggregateOp):
    def __call__(self, column: ibis_types.Column) -> ibis_types.IntegerValue:
        return self._windowize(column.rank())


class AllOp(AggregateOp):
    def __call__(self, column: ibis_types.Column) -> ibis_types.BooleanValue:
        # BQ will return null for empty column, result would be true in pandas.
        result = typing.cast(ibis_types.BooleanColumn, column != 0).all()
        return typing.cast(
            ibis_types.BooleanScalar,
            self._windowize(result).fillna(ibis_types.literal(True)),
        )


class AnyOp(AggregateOp):
    def __call__(self, column: ibis_types.Column) -> ibis_types.BooleanValue:
        # BQ will return null for empty column, result would be false in pandas.
        result = typing.cast(ibis_types.BooleanColumn, column != 0).any()
        return typing.cast(
            ibis_types.BooleanScalar,
            self._windowize(result).fillna(ibis_types.literal(True)),
        )


sum_op = SumOp()
mean_op = MeanOp()
product_op = ProductOp()
max_op = MaxOp()
min_op = MinOp()
count_op = CountOp()
rank_op = RankOp()
all_op = AllOp()
any_op = AnyOp()
