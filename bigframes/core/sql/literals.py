# Copyright 2026 Google LLC
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

import datetime
import decimal
import math
from typing import Mapping, Union

import shapely.geometry.base  # type: ignore

import bigframes.core.compile.googlesql as googlesql

# shapely.wkt.dumps was moved to shapely.io.to_wkt in 2.0.
try:
    from shapely.io import to_wkt  # type: ignore
except ImportError:
    from shapely.wkt import dumps  # type: ignore

    to_wkt = dumps


SIMPLE_LITERAL_TYPES = Union[
    bytes,
    str,
    int,
    bool,
    float,
    datetime.datetime,
    datetime.date,
    datetime.time,
    decimal.Decimal,
    list,
]


def simple_literal(value: Union[SIMPLE_LITERAL_TYPES, None]) -> str:
    """Return quoted input string."""

    # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#literals
    if value is None:
        return "NULL"
    elif isinstance(value, str):
        # Single quoting seems to work nicer with ibis than double quoting
        return f"'{googlesql._escape_chars(value)}'"
    elif isinstance(value, bytes):
        return repr(value)
    elif isinstance(value, (bool, int)):
        return str(value)
    elif isinstance(value, float):
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#floating_point_literals
        if math.isnan(value):
            return 'CAST("nan" as FLOAT)'
        if value == math.inf:
            return 'CAST("+inf" as FLOAT)'
        if value == -math.inf:
            return 'CAST("-inf" as FLOAT)'
        return str(value)
    # Check datetime first as it is a subclass of date
    elif isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            return f"DATETIME('{value.isoformat()}')"
        else:
            return f"TIMESTAMP('{value.isoformat()}')"
    elif isinstance(value, datetime.date):
        return f"DATE('{value.isoformat()}')"
    elif isinstance(value, datetime.time):
        return f"TIME(DATETIME('1970-01-01 {value.isoformat()}'))"
    elif isinstance(value, shapely.geometry.base.BaseGeometry):
        return f"ST_GEOGFROMTEXT({simple_literal(to_wkt(value))})"
    elif isinstance(value, decimal.Decimal):
        # TODO: disambiguate BIGNUMERIC based on scale and/or precision
        return f"CAST('{str(value)}' AS NUMERIC)"
    elif isinstance(value, list):
        simple_literals = [simple_literal(i) for i in value]
        return f"[{', '.join(simple_literals)}]"

    else:
        raise ValueError(f"Cannot produce literal for {value}")


def struct_literal(struct_options: Mapping[str, SIMPLE_LITERAL_TYPES]) -> str:
    rendered_options = []
    for option_name, option_value in struct_options.items():
        rendered_val = simple_literal(option_value)
        rendered_options.append(f"{rendered_val} AS {option_name}")
    return f"STRUCT({', '.join(rendered_options)})"
