"""Mappings for Pandas dtypes supported by BigFrames"""

from typing import Any, Dict, Iterable, Tuple, Union

import db_dtypes  # type: ignore
import ibis
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types
import numpy as np
import pandas as pd

# Type hints for Pandas dtypes supported by BigFrames
BigFramesDtype = Union[
    pd.BooleanDtype,
    pd.Float64Dtype,
    pd.Int64Dtype,
    pd.StringDtype,
    db_dtypes.DateDtype,
    db_dtypes.TimeDtype,
    np.dtype[np.datetime64],
    pd.DatetimeTZDtype,
]

# Type hints for Ibis data types supported by BigFrames
IbisDtype = Union[
    ibis_dtypes.Boolean,
    ibis_dtypes.Float64,
    ibis_dtypes.Int64,
    ibis_dtypes.String,
    ibis_dtypes.Date,
    ibis_dtypes.Time,
    ibis_dtypes.Timestamp,
]

# Type hints for Ibis data types that can be read to Python objects by BigFrames
ReadOnlyIbisDtype = Union[
    ibis_dtypes.Binary, ibis_dtypes.JSON, ibis_dtypes.Decimal, ibis_dtypes.GeoSpatial
]

BIDIRECTIONAL_MAPPINGS: Iterable[Tuple[IbisDtype, BigFramesDtype]] = (
    (ibis_dtypes.boolean, pd.BooleanDtype()),
    (ibis_dtypes.float64, pd.Float64Dtype()),
    (ibis_dtypes.int64, pd.Int64Dtype()),
    (ibis_dtypes.string, pd.StringDtype(storage="pyarrow")),
    (ibis_dtypes.date, db_dtypes.DateDtype()),
    (ibis_dtypes.time, db_dtypes.TimeDtype()),
    (ibis_dtypes.timestamp, np.dtype("datetime64[us]")),
    (ibis_dtypes.Timestamp(timezone="UTC"), pd.DatetimeTZDtype(unit="us", tz="UTC")),  # type: ignore
)

BIGFRAMES_TO_IBIS: Dict[BigFramesDtype, IbisDtype] = {
    pandas: ibis for ibis, pandas in BIDIRECTIONAL_MAPPINGS
}

IBIS_TO_BIGFRAMES: Dict[
    Union[IbisDtype, ReadOnlyIbisDtype], Union[BigFramesDtype, np.dtype[Any]]
] = {ibis: pandas for ibis, pandas in BIDIRECTIONAL_MAPPINGS}
IBIS_TO_BIGFRAMES.update(
    {
        ibis_dtypes.binary: np.dtype("O"),
        ibis_dtypes.json: np.dtype("O"),
        ibis_dtypes.Decimal(precision=38, scale=9, nullable=True): np.dtype("O"),
        ibis_dtypes.Decimal(precision=76, scale=38, nullable=True): np.dtype("O"),
        ibis_dtypes.GeoSpatial(
            geotype="geography", srid=None, nullable=True
        ): pd.StringDtype(storage="pyarrow")
        # TODO: Interval, Array, Struct
    }
)


def ibis_dtype_to_bigframes_dtype(
    ibis_dtype: Union[IbisDtype, ReadOnlyIbisDtype]
) -> Union[BigFramesDtype, np.dtype[Any]]:
    """Converts an Ibis dtype to a BigFrames dtype

    Args:
        ibis_dtype: The ibis dtype used to represent this type, which
        should in turn correspond to an underlying BigQuery type

    Returns:
        The supported BigFrames dtype, which may be provided by pandas,
        numpy, or db_types

    Raises:
        ValueError: if passed an unexpected type
    """
    if ibis_dtype in IBIS_TO_BIGFRAMES:
        return IBIS_TO_BIGFRAMES[ibis_dtype]
    else:
        raise ValueError(f"Unexpected Ibis data type {type(ibis_dtype)}")


def bigframes_dtype_to_ibis_dtype(bigframes_dtype: BigFramesDtype) -> IbisDtype:
    """Converts a BigFrames supported dtype to an Ibis dtype

    Args:
        bigframes_dtype: A dtype supported by BigFrames

    Returns:
        The corresponding Ibis type

    Raises:
        ValueError: if passed a dtype not supported by BigFrames"""
    if bigframes_dtype in BIGFRAMES_TO_IBIS:
        return BIGFRAMES_TO_IBIS[bigframes_dtype]
    else:
        raise ValueError(f"Unexpected data type {type(bigframes_dtype)}")


def literal_to_ibis_scalar(literal) -> ibis.expr.types.Scalar:
    """Accept any literal and, if possible, return an Ibis Scalar
    expression with a BigFrames compatible data type

    Args:
        literal: any value accepted by Ibis

    Returns:
        An ibis Scalar supported by BigFrames

    Raises:
        ValueError: if passed literal cannot be coerced to a
        BigFrames compatible scalar
    """
    scalar_expr = ibis.literal(literal)
    if scalar_expr.type().is_floating():
        scalar_expr = ibis.literal(literal, ibis_dtypes.float64)
    elif scalar_expr.type().is_integer():
        scalar_expr = ibis.literal(literal, ibis_dtypes.int64)

    # TODO(bmil): support other literals that can be coerced to compatible types
    if scalar_expr.type() not in BIGFRAMES_TO_IBIS.values():
        raise ValueError(f"Literal did not coerce to a supported data type: {literal}")

    return scalar_expr
