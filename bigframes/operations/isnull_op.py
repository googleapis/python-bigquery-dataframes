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

# Direct imports from bigframes
from bigframes import dtypes
from bigframes.operations import base_ops
import bigframes.operations.type as op_typing

# Imports for Ibis compilation
from bigframes_vendored.ibis.expr import types as ibis_types

# Imports for Polars compilation
try:
    import polars as pl
except ImportError:
    # Polars is optional, error will be raised elsewhere if user tries to use it.
    pass


# Definitions of IsNullOp and NotNullOp operations
IsNullOp = base_ops.create_unary_op(
    name="isnull",
    type_signature=op_typing.FixedOutputType(
        lambda x: True, dtypes.BOOL_DTYPE, description="nullable"
    ),
)
isnull_op = IsNullOp()

NotNullOp = base_ops.create_unary_op(
    name="notnull",
    type_signature=op_typing.FixedOutputType(
        lambda x: True, dtypes.BOOL_DTYPE, description="nullable"
    ),
)
notnull_op = NotNullOp()

# Ibis Scalar Op Implementations
def _ibis_isnull_op_impl(x: ibis_types.Value):
    return x.isnull()

def _ibis_notnull_op_impl(x: ibis_types.Value):
    return x.notnull()


# Polars Expression Implementations
def _polars_isnull_op_impl(op: IsNullOp, input: pl.Expr) -> pl.Expr:
    return input.is_null()

def _polars_notnull_op_impl(op: NotNullOp, input: pl.Expr) -> pl.Expr:
    return input.is_not_null()

__all__ = [
    "IsNullOp",
    "isnull_op",
    "NotNullOp",
    "notnull_op",
    "_ibis_isnull_op_impl",
    "_ibis_notnull_op_impl",
    "_polars_isnull_op_impl",
    "_polars_notnull_op_impl",
]
