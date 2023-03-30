from __future__ import annotations

import typing

import ibis.expr.types as ibis_types


def numeric_op(operation):
    def constrined_op(column: ibis_types.Column):
        if column.type().is_numeric():
            return operation(column)
        else:
            raise ValueError(
                "Numeric operation cannot be applied to type {}".format(column.type())
            )

    return constrined_op


@numeric_op
def sum_op(column: ibis_types.NumericColumn) -> ibis_types.NumericValue:
    return column.sum()


@numeric_op
def mean_op(column: ibis_types.NumericColumn) -> ibis_types.NumericValue:
    return column.mean()


def max_op(column: ibis_types.Column) -> ibis_types.Value:
    return column.max()


def min_op(column: ibis_types.Column) -> ibis_types.Value:
    return column.min()


def count_op(column: ibis_types.Column) -> ibis_types.IntegerValue:
    return column.count()


def rank(column: ibis_types.Column) -> ibis_types.IntegerValue:
    return column.rank()


def all_op(column: ibis_types.Column) -> ibis_types.BooleanValue:
    # BQ will return null for empty column, result would be true in pandas.
    return typing.cast(
        ibis_types.BooleanScalar,
        typing.cast(ibis_types.BooleanColumn, column != 0)
        .all()
        .fillna(ibis_types.literal(True)),
    )


def any_op(column: ibis_types.Column) -> ibis_types.BooleanValue:
    # BQ will return null for empty column, result would be false in pandas.
    return typing.cast(
        ibis_types.BooleanScalar,
        (
            typing.cast(ibis_types.BooleanColumn, column != 0)
            .any()
            .fillna(ibis_types.literal(False))
        ),
    )
