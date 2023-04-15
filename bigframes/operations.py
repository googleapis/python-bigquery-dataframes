from __future__ import annotations

import typing

import ibis
import ibis.common.exceptions
import ibis.expr.operations.generic
import ibis.expr.types as ibis_types

_ZERO = typing.cast(ibis_types.NumericValue, ibis_types.literal(0))

BinaryOp = typing.Callable[[ibis_types.Value, ibis_types.Value], ibis_types.Value]
TernaryOp = typing.Callable[
    [ibis_types.Value, ibis_types.Value, ibis_types.Value], ibis_types.Value
]

### Unary Ops


def abs_op(x: ibis_types.Value):
    return typing.cast(ibis_types.NumericValue, x).abs()


def invert_op(x: ibis_types.Value):
    return typing.cast(ibis_types.NumericValue, x).negate()


def isnull_op(x: ibis_types.Value):
    return x.isnull()


def len_op(x: ibis_types.Value):
    return typing.cast(ibis_types.StringValue, x).length()


def notnull_op(x: ibis_types.Value):
    return x.notnull()


def reverse_op(x: ibis_types.Value):
    return typing.cast(ibis_types.StringValue, x).reverse()


### Binary Ops
def and_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return typing.cast(ibis_types.BooleanValue, x) & typing.cast(
        ibis_types.BooleanValue, y
    )


def or_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return typing.cast(ibis_types.BooleanValue, x) | typing.cast(
        ibis_types.BooleanValue, y
    )


def add_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return typing.cast(ibis_types.NumericValue, x) + typing.cast(
        ibis_types.NumericValue, y
    )


def sub_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return typing.cast(ibis_types.NumericValue, x) - typing.cast(
        ibis_types.NumericValue, y
    )


def mul_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return typing.cast(ibis_types.NumericValue, x) * typing.cast(
        ibis_types.NumericValue, y
    )


def div_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return typing.cast(ibis_types.NumericValue, x) / typing.cast(
        ibis_types.NumericValue, y
    )


def lt_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return x < y


def le_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return x <= y


def gt_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return x > y


def ge_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    return x >= y


def floordiv_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    x_numeric = typing.cast(ibis_types.NumericValue, x)
    y_numeric = typing.cast(ibis_types.NumericValue, y)
    floordiv_expr = x_numeric // y_numeric
    # MOD(N, 0) will error in bigquery, but needs to return 0 in BQ so we short-circuit in this case.
    # Multiplying left by zero propogates nulls.
    return (
        ibis.case()
        .when(y_numeric == _ZERO, _ZERO * x_numeric)
        .else_(floordiv_expr)
        .end()
    )


def mod_op(
    x: ibis_types.Value,
    y: ibis_types.Value,
):
    x_numeric = typing.cast(ibis_types.NumericValue, x)
    y_numeric = typing.cast(ibis_types.NumericValue, y)
    # Hacky short-circuit to avoid passing zero-literal to sql backend, evaluate locally instead to 0.
    op = y.op()
    if isinstance(op, ibis.expr.operations.generic.Literal) and op.value == 0:
        return _ZERO * x_numeric  # Dummy op to propogate nulls and type from x arg

    bq_mod = x_numeric % y_numeric  # Bigquery will maintain x sign here
    # In BigQuery returned value has the same sign as X. In pandas, the sign of y is used, so we need to flip the result if sign(x) != sign(y)
    return (
        ibis.case()
        .when(
            y_numeric == _ZERO, _ZERO * x_numeric
        )  # Dummy op to propogate nulls and type from x arg
        .when(
            (y_numeric < _ZERO) & (bq_mod > _ZERO), (y_numeric + bq_mod)
        )  # Convert positive result to negative
        .when(
            (y_numeric > _ZERO) & (bq_mod < _ZERO), (y_numeric + bq_mod)
        )  # Convert negative result to positive
        .else_(bq_mod)
        .end()
    )


def reverse(op):
    return lambda x, y: op(y, x)


# Ternary ops


def where_op(
    original: ibis_types.Value,
    condition: ibis_types.Value,
    replacement: ibis_types.Value,
) -> ibis_types.Value:
    """Returns x if y is true, otherwise returns z."""
    return ibis.case().when(condition, original).else_(replacement).end()


def clip_op(
    original: ibis_types.Value,
    lower: ibis_types.Value,
    upper: ibis_types.Value,
) -> ibis_types.Value:
    """Clips value to lower and upper bounds."""
    if isinstance(lower, ibis_types.NullScalar) and (
        not isinstance(upper, ibis_types.NullScalar)
    ):
        return (
            ibis.case()
            .when(upper.isnull() | (original > upper), upper)
            .else_(original)
            .end()
        )
    elif (not isinstance(lower, ibis_types.NullScalar)) and isinstance(
        upper, ibis_types.NullScalar
    ):
        return (
            ibis.case()
            .when(lower.isnull() | (original < lower), lower)
            .else_(original)
            .end()
        )
    elif isinstance(lower, ibis_types.NullScalar) and (
        isinstance(upper, ibis_types.NullScalar)
    ):
        return original
    else:
        # Note: Pandas has unchanged behavior when upper bound and lower bound are flipped. This implementation requires that lower_bound < upper_bound
        return (
            ibis.case()
            .when(lower.isnull() | (original < lower), lower)
            .when(upper.isnull() | (original > upper), upper)
            .else_(original)
            .end()
        )
