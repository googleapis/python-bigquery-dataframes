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

"""Mappings for Pandas dtypes supported by BigQuery DataFrames package"""

from dataclasses import dataclass
import datetime
import decimal
import typing
from typing import Dict, Literal, Union

import geopandas as gpd  # type: ignore
import google.cloud.bigquery
import numpy as np
import pandas as pd
import pyarrow as pa

import bigframes.constants as constants

# Type hints for Pandas dtypes supported by BigQuery DataFrame
Dtype = Union[
    pd.BooleanDtype,
    pd.Float64Dtype,
    pd.Int64Dtype,
    pd.StringDtype,
    pd.ArrowDtype,
    gpd.array.GeometryDtype,
]
# Represents both column types (dtypes) and local-only types
# None represents the type of a None scalar.
ExpressionType = typing.Optional[Dtype]

# Convert to arrow when in array or struct
INT_DTYPE = pd.Int64Dtype()
FLOAT_DTYPE = pd.Float64Dtype()
BOOL_DTYPE = pd.BooleanDtype()
# Wrapped arrow dtypes
STRING_DTYPE = pd.StringDtype(storage="pyarrow")
BYTES_DTYPE = pd.ArrowDtype(pa.binary())
DATE_DTYPE = pd.ArrowDtype(pa.date32())
TIME_DTYPE = pd.ArrowDtype(pa.time64("us"))
DATETIME_DTYPE = pd.ArrowDtype(pa.timestamp("us"))
TIMESTAMP_DTYPE = pd.ArrowDtype(pa.timestamp("us", tz="UTC"))
NUMERIC_DTYPE = pd.ArrowDtype(pa.decimal128(38, 9))
BIGNUMERIC_DTYPE = pd.ArrowDtype(pa.decimal256(76, 38))
# No arrow equivalent
GEO_DTYPE = gpd.array.GeometryDtype()

# Used when storing Null expressions
DEFAULT_DTYPE = FLOAT_DTYPE


# Will have a few dtype variants: simple(eg. int, string, bool), complex (eg. list, struct), and virtual (eg. micro intervals, categorical)
@dataclass(frozen=True)
class SimpleDtypeInfo:
    """
    A simple dtype maps 1:1 with a database type and is not parameterized.
    """

    dtype: Dtype
    arrow_dtype: typing.Optional[pa.DataType]
    type_kind: typing.Tuple[str, ...]  # Should all correspond to the same db type
    logical_bytes: int = (
        8  # this is approximate only, some types are variably sized, also, compression
    )
    orderable: bool = False
    clusterable: bool = False


# TODO: Missing BQ types: INTERVAL, JSON, RANGE
# TODO: Add mappings to python types
SIMPLE_TYPES = (
    SimpleDtypeInfo(
        dtype=INT_DTYPE,
        arrow_dtype=pa.int64(),
        type_kind=("INT64", "INTEGER"),
        orderable=True,
        clusterable=True,
    ),
    SimpleDtypeInfo(
        dtype=FLOAT_DTYPE,
        arrow_dtype=pa.float64(),
        type_kind=("FLOAT64", "FLOAT"),
        orderable=True,
    ),
    SimpleDtypeInfo(
        dtype=BOOL_DTYPE,
        arrow_dtype=pa.bool_(),
        type_kind=("BOOL", "BOOLEAN"),
        logical_bytes=1,
        orderable=True,
        clusterable=True,
    ),
    SimpleDtypeInfo(
        dtype=STRING_DTYPE,
        arrow_dtype=pa.string(),
        type_kind=("STRING",),
        orderable=True,
        clusterable=True,
    ),
    SimpleDtypeInfo(
        dtype=DATE_DTYPE,
        arrow_dtype=pa.date32(),
        type_kind=("DATE",),
        logical_bytes=4,
        orderable=True,
        clusterable=True,
    ),
    SimpleDtypeInfo(
        dtype=TIME_DTYPE,
        arrow_dtype=pa.time64("us"),
        type_kind=("TIME",),
        orderable=True,
    ),
    SimpleDtypeInfo(
        dtype=DATETIME_DTYPE,
        arrow_dtype=pa.timestamp("us"),
        type_kind=("DATETIME",),
        orderable=True,
        clusterable=True,
    ),
    SimpleDtypeInfo(
        dtype=TIMESTAMP_DTYPE,
        arrow_dtype=pa.timestamp("us", tz="UTC"),
        type_kind=("TIMESTAMP",),
        orderable=True,
        clusterable=True,
    ),
    SimpleDtypeInfo(
        dtype=BYTES_DTYPE, arrow_dtype=pa.binary(), type_kind=("BYTES",), orderable=True
    ),
    SimpleDtypeInfo(
        dtype=NUMERIC_DTYPE,
        arrow_dtype=pa.decimal128(38, 9),
        type_kind=("NUMERIC",),
        logical_bytes=16,
        orderable=True,
        clusterable=True,
    ),
    SimpleDtypeInfo(
        dtype=BIGNUMERIC_DTYPE,
        arrow_dtype=pa.decimal256(76, 38),
        type_kind=("BIGNUMERIC",),
        logical_bytes=32,
        orderable=True,
        clusterable=True,
    ),
    # Geo has no corresponding arrow dtype
    SimpleDtypeInfo(
        dtype=GEO_DTYPE,
        arrow_dtype=None,
        type_kind=("GEOGRAPHY",),
        logical_bytes=40,
        clusterable=True,
    ),
)


# Type hints for dtype strings supported by BigQuery DataFrame
DtypeString = Literal[
    "boolean",
    "Float64",
    "Int64",
    "int64[pyarrow]",
    "string",
    "string[pyarrow]",
    "timestamp[us, tz=UTC][pyarrow]",
    "timestamp[us][pyarrow]",
    "date32[day][pyarrow]",
    "time64[us][pyarrow]",
    "decimal128(38, 9)[pyarrow]",
    "decimal256(76, 38)[pyarrow]",
    "binary[pyarrow]",
]

BOOL_BIGFRAMES_TYPES = [pd.BooleanDtype()]

# Corresponds to the pandas concept of numeric type (such as when 'numeric_only' is specified in an operation)
# Pandas is inconsistent, so two definitions are provided, each used in different contexts
NUMERIC_BIGFRAMES_TYPES_RESTRICTIVE = [
    pd.Float64Dtype(),
    pd.Int64Dtype(),
]
NUMERIC_BIGFRAMES_TYPES_PERMISSIVE = NUMERIC_BIGFRAMES_TYPES_RESTRICTIVE + [
    pd.BooleanDtype(),
    pd.ArrowDtype(pa.decimal128(38, 9)),
    pd.ArrowDtype(pa.decimal256(76, 38)),
]


## dtype predicates - use these to maintain consistency
def is_datetime_like(type: ExpressionType) -> bool:
    return type in (DATETIME_DTYPE, TIMESTAMP_DTYPE)


def is_date_like(type: ExpressionType) -> bool:
    return type in (DATETIME_DTYPE, TIMESTAMP_DTYPE, DATE_DTYPE)


def is_time_like(type: ExpressionType) -> bool:
    return type in (DATETIME_DTYPE, TIMESTAMP_DTYPE, TIME_DTYPE)


def is_binary_like(type: ExpressionType) -> bool:
    return type in (BOOL_DTYPE, BYTES_DTYPE, INT_DTYPE)


def is_string_like(type: ExpressionType) -> bool:
    return type in (STRING_DTYPE, BYTES_DTYPE)


def is_array_like(type: ExpressionType) -> bool:
    return isinstance(type, pd.ArrowDtype) and isinstance(
        type.pyarrow_dtype, pa.ListType
    )


def is_array_string_like(type: ExpressionType) -> bool:
    return (
        isinstance(type, pd.ArrowDtype)
        and isinstance(type.pyarrow_dtype, pa.ListType)
        and pa.types.is_string(type.pyarrow_dtype.value_type)
    )


def is_struct_like(type: ExpressionType) -> bool:
    return isinstance(type, pd.ArrowDtype) and isinstance(
        type.pyarrow_dtype, pa.StructType
    )


def is_json_like(type: ExpressionType) -> bool:
    # TODO: Add JSON type support
    return type == STRING_DTYPE


def is_json_encoding_type(type: ExpressionType) -> bool:
    # Types can be converted into JSON.
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions#json_encodings
    return type != GEO_DTYPE


def is_numeric(type: ExpressionType) -> bool:
    return type in NUMERIC_BIGFRAMES_TYPES_PERMISSIVE


def is_iterable(type: ExpressionType) -> bool:
    return type in (STRING_DTYPE, BYTES_DTYPE) or is_array_like(type)


def is_comparable(type: ExpressionType) -> bool:
    return (type is not None) and is_orderable(type)


_ORDERABLE_SIMPLE_TYPES = set(
    mapping.dtype for mapping in SIMPLE_TYPES if mapping.orderable
)


def is_orderable(type: ExpressionType) -> bool:
    # On BQ side, ARRAY, STRUCT, GEOGRAPHY, JSON are not orderable
    return type in _ORDERABLE_SIMPLE_TYPES


_CLUSTERABLE_SIMPLE_TYPES = set(
    mapping.dtype for mapping in SIMPLE_TYPES if mapping.clusterable
)


def is_clusterable(type: ExpressionType) -> bool:
    # https://cloud.google.com/bigquery/docs/clustered-tables#cluster_column_types
    # This is based on default database type mapping, could in theory represent in non-default bq type to cluster.
    return type in _CLUSTERABLE_SIMPLE_TYPES


def is_bool_coercable(type: ExpressionType) -> bool:
    # TODO: Implement more bool coercions
    return (type is None) or is_numeric(type) or is_string_like(type)


BIGFRAMES_STRING_TO_BIGFRAMES: Dict[DtypeString, Dtype] = {
    typing.cast(DtypeString, mapping.dtype.name): mapping.dtype
    for mapping in SIMPLE_TYPES
}

# special case - string[pyarrow] doesn't include the storage in its name, and both
# "string" and "string[pyarrow]" are accepted
BIGFRAMES_STRING_TO_BIGFRAMES["string[pyarrow]"] = pd.StringDtype(storage="pyarrow")

# special case - both "Int64" and "int64[pyarrow]" are accepted
BIGFRAMES_STRING_TO_BIGFRAMES["int64[pyarrow]"] = pd.Int64Dtype()

# For the purposes of dataframe.memory_usage
DTYPE_BYTE_SIZES = {
    type_info.dtype: type_info.logical_bytes for type_info in SIMPLE_TYPES
}

### Conversion Functions


def dtype_for_etype(etype: ExpressionType) -> Dtype:
    if etype is None:
        return DEFAULT_DTYPE
    else:
        return etype


# Mapping between arrow and bigframes types are necessary because arrow types are used for structured types, but not all primitive types,
# so conversion are needed when data is nested or unnested. Also, sometimes local data is stored as arrow.
_ARROW_TO_BIGFRAMES = {
    mapping.arrow_dtype: mapping.dtype
    for mapping in SIMPLE_TYPES
    if mapping.arrow_dtype is not None
}


def arrow_dtype_to_bigframes_dtype(arrow_dtype: pa.DataType) -> Dtype:
    if arrow_dtype in _ARROW_TO_BIGFRAMES:
        return _ARROW_TO_BIGFRAMES[arrow_dtype]
    if pa.types.is_list(arrow_dtype):
        return pd.ArrowDtype(arrow_dtype)
    if pa.types.is_struct(arrow_dtype):
        return pd.ArrowDtype(arrow_dtype)
    if arrow_dtype == pa.null():
        return DEFAULT_DTYPE
    else:
        raise ValueError(
            f"Unexpected Arrow data type {arrow_dtype}. {constants.FEEDBACK_LINK}"
        )


_BIGFRAMES_TO_ARROW = {
    mapping.dtype: mapping.arrow_dtype
    for mapping in SIMPLE_TYPES
    if mapping.arrow_dtype is not None
}


def bigframes_dtype_to_arrow_dtype(
    bigframes_dtype: Dtype,
) -> pa.DataType:
    if bigframes_dtype in _BIGFRAMES_TO_ARROW:
        return _BIGFRAMES_TO_ARROW[bigframes_dtype]
    if isinstance(bigframes_dtype, pd.ArrowDtype):
        if pa.types.is_list(bigframes_dtype.pyarrow_dtype):
            return bigframes_dtype.pyarrow_dtype
        if pa.types.is_struct(bigframes_dtype.pyarrow_dtype):
            return bigframes_dtype.pyarrow_dtype
    else:
        raise ValueError(
            f"No arrow conversion for {bigframes_dtype}. {constants.FEEDBACK_LINK}"
        )


def infer_literal_type(literal) -> typing.Optional[Dtype]:
    # Maybe also normalize literal to canonical python representation to remove this burden from compilers?
    if pd.api.types.is_list_like(literal):
        element_types = [infer_literal_type(i) for i in literal]
        common_type = lcd_type(*element_types)
        as_arrow = bigframes_dtype_to_arrow_dtype(common_type)
        return pd.ArrowDtype(as_arrow)
    if pd.api.types.is_dict_like(literal):
        fields = [
            (key, bigframes_dtype_to_arrow_dtype(infer_literal_type(literal[key])))
            for key in literal.keys()
        ]
        return pd.ArrowDtype(pa.struct(fields))
    if pd.isna(literal):
        return None  # Null value without a definite type
    if isinstance(literal, (bool, np.bool_)):
        return BOOL_DTYPE
    if isinstance(literal, (int, np.integer)):
        return INT_DTYPE
    if isinstance(literal, (float, np.floating)):
        return FLOAT_DTYPE
    if isinstance(literal, decimal.Decimal):
        return NUMERIC_DTYPE
    if isinstance(literal, (str, np.str_)):
        return STRING_DTYPE
    if isinstance(literal, (bytes, np.bytes_)):
        return BYTES_DTYPE
    # Make sure to check datetime before date as datetimes are also dates
    if isinstance(literal, (datetime.datetime, pd.Timestamp)):
        if literal.tzinfo is not None:
            return TIMESTAMP_DTYPE
        else:
            return DATETIME_DTYPE
    if isinstance(literal, datetime.date):
        return DATE_DTYPE
    if isinstance(literal, datetime.time):
        return TIME_DTYPE
    else:
        raise ValueError(f"Unable to infer type for value: {literal}")


def infer_literal_arrow_type(literal) -> typing.Optional[pa.DataType]:
    if pd.isna(literal):
        return None  # Null value without a definite type
    return bigframes_dtype_to_arrow_dtype(infer_literal_type(literal))


# Don't have dtype for json, so just end up interpreting as STRING
_REMAPPED_TYPEKINDS = {"JSON": "STRING"}
_TK_TO_BIGFRAMES = {
    type_kind: mapping.dtype
    for mapping in SIMPLE_TYPES
    for type_kind in mapping.type_kind
}


def convert_schema_field(
    field: google.cloud.bigquery.SchemaField,
) -> typing.Tuple[str, Dtype]:
    is_repeated = field.mode == "REPEATED"
    if field.field_type == "RECORD":
        mapped_fields = map(convert_schema_field, field.fields)
        pa_struct = pa.struct(
            (name, bigframes_dtype_to_arrow_dtype(dtype))
            for name, dtype in mapped_fields
        )
        pa_type = pa.list_(pa_struct) if is_repeated else pa_struct
        return field.name, pd.ArrowDtype(pa_type)
    elif (
        field.field_type in _TK_TO_BIGFRAMES or field.field_type in _REMAPPED_TYPEKINDS
    ):
        singular_type = _TK_TO_BIGFRAMES[
            _REMAPPED_TYPEKINDS.get(field.field_type, field.field_type)
        ]
        if is_repeated:
            pa_type = pa.list_(bigframes_dtype_to_arrow_dtype(singular_type))
            return field.name, pd.ArrowDtype(pa_type)
        else:
            return field.name, singular_type
    else:
        raise ValueError(f"Cannot handle type: {field.field_type}")


def bf_type_from_type_kind(
    bq_schema: list[google.cloud.bigquery.SchemaField],
) -> typing.Dict[str, Dtype]:
    """Converts bigquery sql type to the default bigframes dtype."""
    return {name: dtype for name, dtype in map(convert_schema_field, bq_schema)}


def is_dtype(scalar: typing.Any, dtype: Dtype) -> bool:
    """Captures whether a scalar can be losslessly represented by a dtype."""
    if scalar is None:
        return True
    if pd.api.types.is_bool_dtype(dtype):
        return pd.api.types.is_bool(scalar)
    if pd.api.types.is_float_dtype(dtype):
        return pd.api.types.is_float(scalar)
    if pd.api.types.is_integer_dtype(dtype):
        return pd.api.types.is_integer(scalar)
    if isinstance(dtype, pd.StringDtype):
        return isinstance(scalar, str)
    if isinstance(dtype, pd.ArrowDtype):
        pa_type = dtype.pyarrow_dtype
        return is_patype(scalar, pa_type)
    return False


# string is binary
def is_patype(scalar: typing.Any, pa_type: pa.DataType) -> bool:
    """Determine whether a scalar's type matches a given pyarrow type."""
    if pa_type == pa.time64("us"):
        return isinstance(scalar, datetime.time)
    elif pa_type == pa.timestamp("us"):
        if isinstance(scalar, datetime.datetime):
            return not scalar.tzinfo
        if isinstance(scalar, pd.Timestamp):
            return not scalar.tzinfo
    elif pa_type == pa.timestamp("us", tz="UTC"):
        if isinstance(scalar, datetime.datetime):
            return scalar.tzinfo == datetime.timezone.utc
        if isinstance(scalar, pd.Timestamp):
            return scalar.tzinfo == datetime.timezone.utc
    elif pa_type == pa.date32():
        return isinstance(scalar, datetime.date)
    elif pa_type == pa.binary():
        return isinstance(scalar, bytes)
    elif pa_type == pa.decimal128(38, 9):
        # decimal.Decimal is a superset, but ibis performs out-of-bounds and loss-of-precision checks
        return isinstance(scalar, decimal.Decimal)
    elif pa_type == pa.decimal256(76, 38):
        # decimal.Decimal is a superset, but ibis performs out-of-bounds and loss-of-precision checks
        return isinstance(scalar, decimal.Decimal)
    return False


# Utilities for type coercion, and compatibility
def is_compatible(scalar: typing.Any, dtype: Dtype) -> typing.Optional[Dtype]:
    """Whether scalar can be compare to items of dtype (though maybe requiring coercion). Returns the datatype that must be used for the comparison"""
    if is_dtype(scalar, dtype):
        return dtype
    elif pd.api.types.is_numeric_dtype(dtype):
        # Implicit conversion currently only supported for numeric types
        if pd.api.types.is_bool(scalar):
            return lcd_type(pd.BooleanDtype(), dtype)
        if pd.api.types.is_float(scalar):
            return lcd_type(pd.Float64Dtype(), dtype)
        if pd.api.types.is_integer(scalar):
            return lcd_type(pd.Int64Dtype(), dtype)
        if isinstance(scalar, decimal.Decimal):
            # TODO: Check context to see if can use NUMERIC instead of BIGNUMERIC
            return lcd_type(pd.ArrowDtype(pa.decimal256(76, 38)), dtype)
    return None


def lcd_type(*dtypes: Dtype) -> Dtype:
    if len(dtypes) < 1:
        raise ValueError("at least one dypes should be provided")
    if len(dtypes) == 1:
        return dtypes[0]
    unique_dtypes = set(dtypes)
    if len(unique_dtypes) == 1:
        return unique_dtypes.pop()
    # Implicit conversion currently only supported for numeric types
    hierarchy: list[Dtype] = [
        pd.BooleanDtype(),
        pd.Int64Dtype(),
        pd.ArrowDtype(pa.decimal128(38, 9)),
        pd.ArrowDtype(pa.decimal256(76, 38)),
        pd.Float64Dtype(),
    ]
    if any([dtype not in hierarchy for dtype in dtypes]):
        return None
    lcd_index = max([hierarchy.index(dtype) for dtype in dtypes])
    return hierarchy[lcd_index]


def coerce_to_common(etype1: ExpressionType, etype2: ExpressionType) -> ExpressionType:
    """Coerce types to a common type or throw a TypeError"""
    if etype1 is not None and etype2 is not None:
        common_supertype = lcd_type(etype1, etype2)
        if common_supertype is not None:
            return common_supertype
    if can_coerce(etype1, etype2):
        return etype2
    if can_coerce(etype2, etype1):
        return etype1
    raise TypeError(f"Cannot coerce {etype1} and {etype2} to a common type.")


def can_coerce(source_type: ExpressionType, target_type: ExpressionType) -> bool:
    if source_type is None:
        return True  # None can be coerced to any supported type
    else:
        return (source_type == STRING_DTYPE) and (
            target_type in (DATETIME_DTYPE, TIMESTAMP_DTYPE, TIME_DTYPE, DATE_DTYPE)
        )


def lcd_type_or_throw(dtype1: Dtype, dtype2: Dtype) -> Dtype:
    result = lcd_type(dtype1, dtype2)
    if result is None:
        raise NotImplementedError(
            f"BigFrames cannot upcast {dtype1} and {dtype2} to common type. {constants.FEEDBACK_LINK}"
        )
    return result


### Remote functions use only
# TODO: Refactor into remote function module

# Input and output types supported by BigQuery DataFrames remote functions.
# TODO(shobs): Extend the support to all types supported by BQ remote functions
# https://cloud.google.com/bigquery/docs/remote-functions#limitations
RF_SUPPORTED_IO_PYTHON_TYPES = {bool, bytes, float, int, str}

RF_SUPPORTED_IO_BIGQUERY_TYPEKINDS = {
    "BOOLEAN",
    "BOOL",
    "BYTES",
    "FLOAT",
    "FLOAT64",
    "INT64",
    "INTEGER",
    "STRING",
}
