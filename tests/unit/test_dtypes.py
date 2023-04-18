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

import db_dtypes  # type: ignore
import ibis
import ibis.expr.datatypes as ibis_dtypes
import numpy as np
import pandas as pd
import pytest

import bigframes.dtypes


@pytest.mark.parametrize(
    ["ibis_dtype", "bigframes_dtype"],
    [
        # This test should cover all the standard BigQuery data types as they
        # appear in Ibis
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types
        # corresponding to BigQuery ARRAY
        # TODO(bmil)
        # corresponding to BigQuery BIGNUMERIC
        (ibis_dtypes.Decimal(precision=76, scale=38, nullable=True), np.dtype("O")),
        # corresponding to BigQuery BOOL
        (ibis_dtypes.boolean, pd.BooleanDtype()),
        # corresponding to BigQuery BYTES
        (ibis_dtypes.binary, np.dtype("O")),
        # corresponding to BigQuery DATE
        (ibis_dtypes.date, db_dtypes.DateDtype()),
        # corresponding to BigQuery DATETIME
        (ibis_dtypes.Timestamp(), np.dtype("datetime64[us]")),
        # corresponding to BigQuery FLOAT64
        (ibis_dtypes.float64, pd.Float64Dtype()),
        # corresponding to BigQuery GEOGRAPHY
        (
            ibis_dtypes.GeoSpatial(geotype="geography", srid=None, nullable=True),
            pd.StringDtype(storage="pyarrow"),
        ),
        # corresponding to BigQuery INT64
        (ibis_dtypes.int64, pd.Int64Dtype()),
        # corresponding to BigQuery INTERVAL
        # TODO(bmil)
        # corresponding to BigQuery JSON
        (ibis_dtypes.json, np.dtype("O")),
        # corresponding to BigQuery NUMERIC
        (ibis_dtypes.Decimal(precision=38, scale=9, nullable=True), np.dtype("O")),
        # corresponding to BigQuery STRING
        (ibis_dtypes.string, pd.StringDtype(storage="pyarrow")),
        # corresponding to BigQuery STRUCT
        # TODO(bmil)
        # corresponding to BigQuery TIME
        (ibis_dtypes.time, db_dtypes.TimeDtype()),
        # corresponding to BigQuery TIMESTAMP
        (
            ibis_dtypes.Timestamp(timezone="UTC"),
            pd.DatetimeTZDtype(unit="us", tz="UTC"),  # type: ignore
        ),
    ],
)
def test_ibis_dtype_converts(ibis_dtype, bigframes_dtype):
    """Test all the Ibis data types needed to read BigQuery tables"""
    result = bigframes.dtypes.ibis_dtype_to_bigframes_dtype(ibis_dtype)
    assert result == bigframes_dtype


def test_ibis_timestamp_pst_raises_unexpected_datatype():
    """BigQuery timestamp only supports UTC time"""
    with pytest.raises(ValueError, match="Unexpected Ibis data type"):
        bigframes.dtypes.ibis_dtype_to_bigframes_dtype(
            ibis_dtypes.Timestamp(timezone="PST")
        )


def test_ibis_float32_raises_unexpected_datatype():
    """Other Ibis types not read from BigQuery are not expected"""
    with pytest.raises(ValueError, match="Unexpected Ibis data type"):
        bigframes.dtypes.ibis_dtype_to_bigframes_dtype(ibis_dtypes.float32)


@pytest.mark.parametrize(
    ["bigframes_dtype", "ibis_dtype"],
    [
        # This test covers all dtypes that BigFrames can exactly map to Ibis
        (pd.BooleanDtype(), ibis_dtypes.boolean),
        (db_dtypes.DateDtype(), ibis_dtypes.date),
        (np.dtype("datetime64[us]"), ibis_dtypes.Timestamp()),
        (pd.Float64Dtype(), ibis_dtypes.float64),
        (pd.Int64Dtype(), ibis_dtypes.int64),
        (pd.StringDtype(storage="pyarrow"), ibis_dtypes.string),
        (db_dtypes.TimeDtype(), ibis_dtypes.time),
        (
            pd.DatetimeTZDtype(unit="us", tz="UTC"),  # type: ignore
            ibis_dtypes.Timestamp(timezone="UTC"),
        ),
    ],
)
def test_bigframes_dtype_converts(ibis_dtype, bigframes_dtype):
    """Test all the Ibis data types needed to read BigQuery tables"""
    result = bigframes.dtypes.bigframes_dtype_to_ibis_dtype(bigframes_dtype)
    assert result == ibis_dtype


def test_numpy_float32_raises_unexpected_datatype():
    """Incompatible dtypes should fail when passed into BigFrames"""
    with pytest.raises(ValueError, match="Unexpected data type"):
        bigframes.dtypes.bigframes_dtype_to_ibis_dtype(np.float32)


@pytest.mark.parametrize(
    ["literal", "ibis_scalar"],
    [
        (True, ibis.literal(True, ibis_dtypes.boolean)),
        (5, ibis.literal(5, ibis_dtypes.int64)),
        (-33.2, ibis.literal(-33.2, ibis_dtypes.float64)),
    ],
)
def test_literal_to_ibis_scalar_converts(literal, ibis_scalar):
    assert bigframes.dtypes.literal_to_ibis_scalar(literal).equals(ibis_scalar)


def test_literal_to_ibis_scalar_throws_on_incompatible_literal():
    with pytest.raises(
        ValueError,
        match="Literal did not coerce to a supported data type: {'mykey': 'myval'}",
    ):
        bigframes.dtypes.literal_to_ibis_scalar({"mykey": "myval"})
