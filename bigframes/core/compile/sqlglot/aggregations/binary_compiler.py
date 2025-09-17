# Copyright 2025 Google LLC
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

import sqlglot.expressions as sge

from bigframes.core import window_spec
import bigframes.core.compile.sqlglot.aggregations.op_registration as reg
import bigframes.core.compile.sqlglot.expressions.typed_expr as typed_expr
from bigframes.operations import aggregations as agg_ops

BINARY_OP_REGISTRATION = reg.OpRegistration()


def compile(
    op: agg_ops.WindowOp,
    left: typed_expr.TypedExpr,
    right: typed_expr.TypedExpr,
    window: typing.Optional[window_spec.WindowSpec] = None,
) -> sge.Expression:
    return BINARY_OP_REGISTRATION[op](op, left, right, window=window)
