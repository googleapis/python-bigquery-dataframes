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

import math
import typing

import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types

# TODO(tbergeron): Encode more efficiently
ORDERING_ID_STRING_BASE: int = 10
# Sufficient to store any value up to 2^63
DEFAULT_ORDERING_ID_LENGTH: int = math.ceil(63 * math.log(2, ORDERING_ID_STRING_BASE))


def stringify_order_id(
    order_id: ibis_types.Value, length: int = DEFAULT_ORDERING_ID_LENGTH
) -> ibis_types.StringValue:
    """Converts an order id value to string if it is not already a string. MUST produced fixed-length strings."""
    if order_id.type().is_int64():
        # This is very inefficient encoding base-10 string uses only 10 characters per byte(out of 256 bit combinations)
        # Furthermore, if know tighter bounds on order id are known, can produce smaller strings.
        # 19 characters chosen as it can represent any positive Int64 in base-10
        # For missing values, ":" * 19 is used as it is larger than any other value this function produces, so null values will be last.
        string_order_id = (
            typing.cast(
                ibis_types.StringValue,
                typing.cast(ibis_types.IntegerValue, order_id).cast(ibis_dtypes.string),
            )
            .lpad(length, "0")
            .fillna(ibis_types.literal(":" * length))
        )
    else:
        string_order_id = (
            typing.cast(ibis_types.StringValue, order_id)
            .lpad(length, "0")
            .fillna(ibis_types.literal(":" * length))
        )
    return typing.cast(ibis_types.StringValue, string_order_id)
