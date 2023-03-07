"""Mappings for Pandas dtypes supported by BigFrames"""

import db_dtypes  # type: ignore
import ibis.expr.datatypes as ibis_dtypes
import numpy as np
import pandas as pd

BIDIRECTIONAL_MAPPINGS = (
    (ibis_dtypes.boolean, pd.BooleanDtype()),
    (ibis_dtypes.float64, pd.Float64Dtype()),
    (ibis_dtypes.int64, pd.Int64Dtype()),
    (ibis_dtypes.string, pd.StringDtype()),
    (ibis_dtypes.date, db_dtypes.DateDtype()),
    (ibis_dtypes.time, db_dtypes.TimeDtype()),
    (ibis_dtypes.timestamp, np.dtype("datetime64[us]")),
    (ibis_dtypes.Timestamp(timezone="UTC"), pd.DatetimeTZDtype(unit="us", tz="UTC")),  # type: ignore
)

BIGFRAMES_TO_IBIS = {pandas: ibis for ibis, pandas in BIDIRECTIONAL_MAPPINGS}

IBIS_TO_BIGFRAMES = {ibis: pandas for ibis, pandas in BIDIRECTIONAL_MAPPINGS}
IBIS_TO_BIGFRAMES.update(
    {
        ibis_dtypes.binary: np.dtype("O"),
        ibis_dtypes.json: np.dtype("O"),
        ibis_dtypes.Decimal(precision=38, scale=9, nullable=True): np.dtype("O"),
        ibis_dtypes.Decimal(precision=76, scale=38, nullable=True): np.dtype("O"),
        ibis_dtypes.GeoSpatial(geotype="geography", srid=None, nullable=True): np.dtype(
            "O"
        )
        # TODO: Interval, Array, Struct
    }
)


def ibis_dtype_to_bigframes_dtype(ibis_dtype: ibis_dtypes.DataType):
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
        raise ValueError("Unexpected Ibis data type")


def bigframes_dtype_to_ibis_dtype(bigframes_dtype):
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
        raise ValueError("Unexpected data type")
